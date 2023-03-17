import argparse
import os
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from tqdm import tqdm

from factories import (
    DownstreamDatasetFactory,
    PretrainingModelFactory,
    OptimizerFactory,
    LRSchedulerFactory,
)

from config import Config
from utils.checkpointing import CheckpointManager
from utils.common import common_parser, common_setup, cycle
import utils.distributed as dist
from utils.metrics import TopkAccuracy
from utils.base import Timer
from clip import build_model, tokenize

parser = common_parser(
    description="Train SVMs for VOC2007 classification on a pretrained model."
)
group = parser.add_argument_group("Downstream config arguments.")
group.add_argument(
    "--down-config", metavar="FILE", help="Path to a downstream config file."
)
group.add_argument(
    "--down-config-override",
    nargs="*",
    default=[],
    help="A list of key-value pairs to modify downstream config params.",
)

parser.add_argument_group("Checkpointing")
parser.add_argument(
    "--weight-init",
    choices=["random", "imagenet", "torchvision", "vlinfo", "clip"],
    default="vlinfo",
    help="""How to initialize weights:
        1. 'random' initializes all weights randomly
        2. 'imagenet' initializes backbone weights from torchvision model zoo
        3. 'vlinfo' load state dict from --checkpoint-path""",
)

parser.add_argument(
    "--checkpoint-path",
    help="Path to load checkpoint and run downstream task evaluation.",
    required=True,
)


@torch.no_grad()
def evaluation(arch, dataloader, tokenizer, device, _A):
    if _A.weight_init == "vlinfo":
        # Put model in eval mode
        text_encoder = arch.text_encoder.to(device).eval()
        text_projector = arch.loss.global_d.text_block.to(device).eval()

        image_encoder = arch.image_encoder.to(device).eval()
        image_projector = arch.loss.global_d.img_block.to(device).eval()

        del arch

    if _A.weight_init == "clip":
        arch.to(device)

    # Extract text features
    texts = dataloader.dataset.text
    num_text = len(texts)
    text_bs = 128

    text_embeds = []

    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i: min(num_text, i + text_bs)]

        if _A.weight_init == "vlinfo":
            # Tokenize the text
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            )

            input_ids = text_input["input_ids"].to(device)
            attention_mask = text_input["attention_mask"].to(device)

            text_feat = text_encoder(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
            text_embed = F.normalize(text_projector(text_feat), p=2, dim=-1)

            text_embeds.append(text_embed)

        elif _A.weight_init == "clip":
            text_input = tokenize(text)
            text_embed = arch.encode_text(text_input.to(device))
            text_embed = F.normalize(text_embed, p=2, dim=-1)
            text_embeds.append(text_embed)

    text_embeds = torch.cat(text_embeds, dim=0)

    image_feats = []
    image_embeds = []
    image_ids = []
    for image, img_id in tqdm(dataloader):
        image = image.to(device)

        if _A.weight_init == "vlinfo":
            image_feat = image_encoder(image)
            image_embed = F.normalize(image_projector(image_feat), p=2, dim=-1)

        elif _A.weight_init == "clip":
            image_feat = arch.encode_image(image)
            image_embed = F.normalize(image_feat, p=2, dim=-1)

        image_embeds.append(image_embed)
        image_ids.append(img_id)

    image_embeds = torch.cat(image_embeds, dim=0)
    image_ids = torch.cat(image_ids, dim=0)

    # Create similarity matrix
    sims_matrix = image_embeds @ text_embeds.t()

    score_matrix_i2t = sims_matrix
    score_matrix_t2i = sims_matrix.t()

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), image_ids


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, image_ids):

    idx2img = {}
    img2idx = {}

    for idx, img_id in enumerate(image_ids):
        idx2img[idx] = img_id.item()
        img2idx[img_id.item()] = idx

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        image_id = image_ids[index].item()
        for i in img2txt[image_id]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]

        img_id = txt2img[index]
        img_idx = img2idx[img_id]
        ranks[index] = np.where(inds == img_idx)[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    }
    return eval_result


def main(_A: argparse.Namespace):
    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device (this will be zero here by default).
        device = torch.cuda.current_device()

    # Create a downstream config object (this will be immutable) and perform
    # common setup such as logging and setting up serialization directory.
    _DOWNC = Config(_A.down_config, _A.down_config_override)
    common_setup(_DOWNC, _A, job_type="downstream")

    dataset = DownstreamDatasetFactory.from_config(_DOWNC, split="val")
    dataloader = DataLoader(
        dataset,
        batch_size=_DOWNC.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        shuffle=False,
    )

    if _A.weight_init == "vlinfo":
        # Create a (pretraining) config object and backup in serialization directory.
        _C = Config(_A.config, _A.config_override)
        # Initialize from a checkpoint, but only keep the visual module.
        arch = PretrainingModelFactory.from_config(_C)
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

        # Load weights according to the init method, do nothing for `random`, and
        # `imagenet` is already taken care of.
        checkpoint_path = os.path.join(_A.checkpoint_path)
        if _A.weight_init == "vlinfo":
            _ = CheckpointManager(model=arch).load(checkpoint_path)

    if _A.weight_init == "clip":
        state_dict = torch.load(_A.checkpoint_path, map_location="cpu")
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        arch = build_model(sd)
        tokenizer = None

    score_val_i2t, score_val_t2i, image_ids = evaluation(
        arch, dataloader, tokenizer, device, _A
    )

    val_result = itm_eval(
        score_val_i2t,
        score_val_t2i,
        dataloader.dataset.txt2img,
        dataloader.dataset.img2txt,
        image_ids,
    )
    print(val_result)

    log_stats = {
        **{f"val_{k}": v for k, v in val_result.items()},
    }

    print(log_stats)


if __name__ == "__main__":
    _A = parser.parse_args()
    _A.num_gpus_per_machine = 1

    # No distributed training here, just a single process.
    main(_A)
