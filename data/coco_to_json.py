import argparse
import os
import pickle
import platform
from typing import Any, List


from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import pandas as pd
import json

# fmt: off
# insert at 1, 0 is the script path (or '' in REPL)
from readers import CocoCaptionsReader

parser = argparse.ArgumentParser("Serialize a COCO Captions split to json.")
parser.add_argument(
    "-d",
    "--data-root",
    default="/bigtemp/as3ek/p/vlinfo/datasets/coco",
    help="Path to the root directory of COCO dataset.",
)
parser.add_argument(
    "-s",
    "--split",
    choices=["train", "val"],
    required=True,
    help="Which split to process, either `train` or `val`.",
)
parser.add_argument(
    "-t",
    "--save-type",
    choices=["csv", "json"],
    required=True,
    help="Which ouput type to process, either `csv` or `json`.",
)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=128,
    help="Batch size to process and serialize data. Set as per CPU memory.",
)
parser.add_argument(
    "-j",
    "--cpu-workers",
    type=int,
    default=4,
    help="Number of CPU workers for data loading.",
)
parser.add_argument(
    "-e",
    "--short-edge-size",
    type=int,
    default=None,
    help="""Resize shorter edge to this size (keeping aspect ratio constant)
    before serializing. Useful for saving disk memory, and faster read.
    If None, no images are resized.""",
)

parser.add_argument(
    "-o",
    "--output-dir",
    default="csvs/",
    required=True,
    help="Path to to the folder which stores the file containing serialized dataset.",
)
# fmt: on


def collate_fn(instances: List[Any]):
    r"""Collate function for data loader to return list of instances as-is."""
    return instances


if __name__ == "__main__":

    _A = parser.parse_args()
    os.makedirs(os.path.dirname(_A.output_dir), exist_ok=True)

    dloader = DataLoader(
        CocoCaptionsReader(_A.data_root, _A.split),
        batch_size=_A.batch_size,
        num_workers=_A.cpu_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    d = []
    i = 0

    for idx, batch in enumerate(tqdm(dloader)):
        for instance in batch:
            image_id = instance["image_id"]
            filename = instance["filename"]
            captions = instance["captions"]

            for caption in captions:
                # add a dictionary entry to the final dictionary
                d.append({"image": filename, "caption": caption})
                # increment the counter
                i = i + 1

    if _A.save_type == "csv":
        data = pd.DataFrame.from_dict(d, "index")
        output_path = os.path.join(_A.output_dir, _A.split + "_coco.csv")
        data.to_csv(output_path)

    elif _A.save_type == "json":
        output_path = os.path.join(_A.output_dir, _A.split + "_coco.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)
