import argparse
import multiprocessing as mp
import os
from typing import Any, List

from loguru import logger
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizer

import torchvision
import torchvision.transforms as transforms

from config import Config
from factories import PretrainingModelFactory, DownstreamDatasetFactory
from utils.checkpointing import CheckpointManager
from utils.common import common_parser, common_setup
from encoder import ImageEncoder
import json
from clip import build_model, tokenize

parser = common_parser(
    description="Train SVMs for VOC2007 classification on a pretrained model."
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


def main(_A: argparse.Namespace):
    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device (this will be zero here by default).
        device = torch.cuda.current_device()

    # Create a (pretraining) config object and backup in serialization directory.
    _C = Config(_A.config, _A.config_override)

    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    batch_size = 128

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)

    class_captions = [
        "a picture of a plane.",
        "a picture of a car.",
        "a picture of a bird.",
        "a picture of a cat.",
        "a picture of a deer.",
        "a picture of a dog.",
        "a picture of a frog.",
        "a picture of a horse.",
        "a picture of a ship.",
        "a picture of a truck.",
    ]

    # Load weights according to the init method, do nothing for `random`, and
    # `imagenet` is already taken care of.
    if _A.weight_init == "vlinfo":
        # Initialize from a checkpoint, but only keep the visual module.
        arch = PretrainingModelFactory.from_config(_C)

        checkpoint_manager = CheckpointManager(model=arch)
        _ = checkpoint_manager.load(_A.checkpoint_path)

        # Put model in eval mode
        text_encoder = arch.text_encoder.to(device).eval()
        text_projector = arch.loss.global_d.text_block.to(device).eval()

        image_encoder = arch.image_encoder.to(device).eval()
        image_projector = arch.loss.global_d.img_block.to(device).eval()

        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

        # encode class prompts
        text_input = tokenizer(
            class_captions,
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
        prompt_features = F.normalize(text_projector(text_feat), p=2, dim=-1)

    if _A.weight_init == "clip":
        state_dict = torch.load(_A.checkpoint_path, map_location="cpu")
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        arch = build_model(sd)
        arch.to(device)

        text_input = tokenize(class_captions)
        text_embed = arch.encode_text(text_input.to(device))
        prompt_features = F.normalize(text_embed, p=2, dim=-1)

    # Extract image features
    features_test: List[torch.Tensor] = []
    targets_test: List[torch.Tensor] = []

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            if _A.weight_init == "vlinfo":
                features = F.normalize(image_projector(
                    image_encoder(images)), p=2, dim=-1)
            if _A.weight_init == "clip":
                features = F.normalize(arch.encode_image(images), p=2, dim=-1)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(features @ prompt_features.t(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print((100 * correct / total))


if __name__ == "__main__":
    _A = parser.parse_args()
    _A.num_gpus_per_machine = 1

    # No distributed training here, just a single process.
    main(_A)
