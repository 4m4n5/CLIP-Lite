import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda import amp

# from torch.utils.tensorboard import SummaryWriter

import os
import wandb
import numpy as np
import argparse
import json
from fvcore.common.config import CfgNode as CN
from collections import Counter
from typing import Any
from loguru import logger
from tqdm import tqdm

# fmt: off
from factories import (
    PretrainingDatasetFactory, PretrainingModelFactory, OptimizerFactory,
    LRSchedulerFactory, NegativeSamplingDatasetFactory
)
from config import Config
from utils.common import common_parser, common_setup, cycle
import utils.distributed as dist
from utils.checkpointing import CheckpointManager
from utils.base import Timer, make_directory, StatefulDistributedSampler

import warnings
warnings.filterwarnings('ignore')

parser = common_parser(
    description="Train the VLInfo model on COCO Captions."
)
group = parser.add_argument_group("Checkpointing and Logging")
group.add_argument(
    "--resume-from", default=None,
    help="Path to a checkpoint to resume training from (if provided)."
)
group.add_argument(
    "--checkpoint-every", type=int, default=10000,
    help="Serialize model to a checkpoint after every these many iterations.",
)
group.add_argument(
    "--log-every", type=int, default=500,
    help="""Log training curves to wandb after every these many iterations
    only master process logs averaged loss values across processes.""",
)
group.add_argument(
    "--climax-freq", type=int, default=1000,
    help="""Frequency to checkpoint at during climax (last 20% training)""",
)
# fmt: on


def init_dataloaders(_C, _A, type="normal"):
    # Initialize dataset
    if type == "normal":
        train_dataset = PretrainingDatasetFactory.from_config(
            _C, split="train")
        val_dataset = PretrainingDatasetFactory.from_config(_C, split="val")
        batch_size = _C.OPTIM.BATCH_SIZE // dist.get_world_size()
    elif type == "clusters":
        # Initialize dataset, datasampler, dataloader
        train_dataset = NegativeSamplingDatasetFactory.from_config(
            _C, split="train")
        val_dataset = NegativeSamplingDatasetFactory.from_config(
            _C, split="val")
        batch_size = (_C.OPTIM.BATCH_SIZE // dist.get_world_size()) // 2

    # Initialize datasamplers
    train_sampler = (
        DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        if _A.num_gpus_per_machine > 0
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        if _A.num_gpus_per_machine > 0
        else None
    )

    # Initialize dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )

    return train_dataloader, val_dataloader


def main(_A: argparse.Namespace):
    if _A.num_gpus_per_machine == 0:
        device: Any = torch.device("cpu")
    else:
        device = torch.cuda.current_device()

    # Create a config object (this will be immutable) and perform common setup
    # such as logging and setting up serialization directory.
    _C = Config(_A.config, _A.config_override)
    common_setup(_C, _A)

    if dist.is_master_process():
        wandb.init(config=_C)
        make_directory(_A.checkpoints_dir + _C.RUN_ID)

    # Initialize and load model, optimizer, scheduler, scaler
    model = PretrainingModelFactory.from_config(_C).to(device)
    # TODO: .named_parameters() vs .parameters()
    optimizer = OptimizerFactory.from_config(_C, model.named_parameters())
    scheduler = LRSchedulerFactory.from_config(_C, optimizer)
    scaler = amp.GradScaler(enabled=_C.AMP)

    # Continue training
    if _A.resume_from is not None:
        start_iteration = CheckpointManager(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        ).load(_A.resume_from)
    else:
        start_iteration = 0

    # Load Clustered Negative sampling dataloaders
    if (start_iteration >= _C.DATA.NEGATIVE_SAMPLING_START_ITERATION) and (
        "clusters" in _C.DATA.NEGATIVE_SAMPLING
    ):
        logger.info(
            f"Starting clustered negative sampling, loading new dataloaders...")
        # Initialize clustered dataloaders
        train_dataloader, val_dataloader = init_dataloaders(
            _C, _A, type="clusters")
        # Initialize an iterator to sample batches infinitely
        train_dataloader_iter = cycle(
            train_dataloader, device, start_iteration, type="clusters"
        )

    else:
        # Initialize Normal dataloaders
        train_dataloader, val_dataloader = init_dataloaders(
            _C, _A, type="normal")

        # Initialize an iterator to sample batches infinitely
        train_dataloader_iter = cycle(
            train_dataloader, device, start_iteration, type="normal"
        )

    if dist.get_world_size() > 1:
        dist.synchronize()
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )

    # Keep track of time per iteration and ETA.
    timer = Timer(
        start_from=start_iteration + 1, total_iterations=_C.OPTIM.NUM_ITERATIONS
    )

    if dist.is_master_process():
        checkpoint_manager = CheckpointManager(
            _A.checkpoints_dir + _C.RUN_ID,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    # Training Loop
    for iteration in range(start_iteration + 1, _C.OPTIM.NUM_ITERATIONS + 1):
        if (iteration == _C.DATA.NEGATIVE_SAMPLING_START_ITERATION) and (
            "clusters" in _C.DATA.NEGATIVE_SAMPLING
        ):
            logger.info(
                f"Starting clustered negative sampling, loading new dataloaders..."
            )
            # Initialize clustered dataloaders
            train_dataloader, val_dataloader = init_dataloaders(
                _C, _A, type="clusters")
            # Initialize an iterator to sample batches infinitely
            train_dataloader_iter = cycle(
                train_dataloader, device, iteration, type="clusters"
            )

        timer.tic()
        optimizer.zero_grad()
        batch = next(train_dataloader_iter)

        with amp.autocast(enabled=_C.AMP):
            output_dict = model(batch)
            loss = output_dict["loss"]

        scaler.scale(loss).backward()

        # Clip norm of gradients, and then do optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), _C.OPTIM.CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        timer.toc()

        # Logging
        if iteration % _A.log_every == 0:
            logger.info(
                f"{timer.stats} [Loss {loss:.3f}] [GPU {dist.gpu_mem_usage()} MB]"
            )
            if dist.is_master_process():
                wandb.log(
                    {
                        "info_loss_train": output_dict["loss_components"]["total_loss"],
                        "cross_modal_loss_train": output_dict["loss_components"][
                            "cross_modal_loss"
                        ],
                        "visual_loss_train": output_dict["loss_components"][
                            "visual_loss"
                        ],
                        "textual_loss_train": output_dict["loss_components"][
                            "textual_loss"
                        ],
                    }
                )

        # checkpointing
        if iteration % _A.checkpoint_every == 0:
            if dist.is_master_process():
                checkpoint_manager.step(iteration)
            # All processes will wait till master process is done serializing.
            dist.synchronize()

            torch.set_grad_enabled(False)
            model.eval()

            # Accumulate different val loss components according to the type of
            # pretraining model.
            val_loss_counter: Counter = Counter()

            for val_iteration, val_batch in enumerate(val_dataloader, start=1):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)

                output_dict = model(val_batch)
                val_loss_counter.update(output_dict["loss_components"])

            # Divide each loss component by number of val batches per GPU.
            val_loss_dict = {
                k: v / val_iteration for k, v in dict(val_loss_counter).items()
            }
            dist.average_across_processes(val_loss_dict)

            torch.set_grad_enabled(True)
            model.train()

            if dist.is_master_process():
                wandb.log(
                    {
                        "info_loss_val": val_loss_dict["total_loss"],
                        "cross_modal_loss_val": val_loss_dict["cross_modal_loss"],
                        "visual_loss_val": val_loss_dict["visual_loss"],
                        "textual_loss_val": val_loss_dict["textual_loss"],
                    }
                )

        # Crazy checkpointing
        climax_checkpoint_freq = _A.climax_freq
        if (iteration / _C.OPTIM.NUM_ITERATIONS) > 0.8:
            if iteration % climax_checkpoint_freq == 0:
                if dist.is_master_process():
                    checkpoint_manager.climax_step(iteration)
                dist.synchronize()


if __name__ == "__main__":
    _A = parser.parse_args()

    if _A.num_gpus_per_machine == 0:
        main(_A)

    else:
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus_per_machine,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            args=(_A,),
        )
