import argparse
import pickle
import faiss
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
import csv
from sentence_transformers import SentenceTransformer, util
import cv2
import json
import os
import random
from collections import defaultdict
import glob
from typing import Callable, Dict, List, Tuple
import albumentations as alb
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import sys

# fmt: off
sys.path.insert(1, "/u/as3ek/github/vlinfo/data/")
from readers import LmdbReader
# fmt: on


class CaptionClustering:
    def __init__(
        self,
        data_root: str = "/bigtemp/as3ek/p/vlinfo/datasets/serialized/",
        cluster_root: str = "/bigtemp/as3ek/p/vlinfo/datasets/clusters/",
        coco_root: str = "/bigtemp/as3ek/p/vlinfo/datasets/coco/",
        split: str = "train",
        mode: str = "train_sbert",
    ):
        lmdb_path = os.path.join(data_root, f"coco_{split}_{mode}2017.lmdb")
        self.reader = LmdbReader(lmdb_path, percentage=100)
        self.num_samples = len(self.reader)
        self.encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.niter = 200
        self.split = split
        self.cluster_root = cluster_root
        self.coco_root = coco_root

        # Maps
        self.img_id_caption_map = {}
        self.img_id_caption_map_path = os.path.join(
            self.cluster_root, f"img_id_caption_map_{split}.pkl"
        )
        self.img_id_encoding_map = {}
        self.img_id_encoding_map_path = os.path.join(
            self.cluster_root, f"img_id_encoding_map_{split}.pkl"
        )
        self.img_id_filename_map = {}
        self.img_id_filename_map_path = os.path.join(
            self.cluster_root, f"img_id_filename_map_{split}.pkl"
        )

    def save_pickle(self, data, path):
        with open(path, "wb") as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        return data

    def extract_embeddings(self):
        split = self.split
        # Load {image_id: filename}
        try:
            self.img_id_filename_map = self.load_pickle(
                self.img_id_filename_map_path)

            for image_id, filepath in self.img_id_filename_map.items():
                self.img_id_filename_map[image_id] = filepath.replace(
                    self.coco_root, "")
            print("Load and clean image_id > filename map")
            self.save_pickle(self.img_id_filename_map,
                             self.img_id_filename_map_path)
        except:
            image_dir = os.path.join(self.coco_root, f"{split}2017")
            # Create tuple (image_id, filename) using COCO2017 format
            image_filenames = glob.glob(os.path.join(image_dir, "*.jpg"))
            id_filename = [
                (int(os.path.basename(name)[:-4]), name) for name in image_filenames
            ]
            for (image_id, filepath) in id_filename:
                self.img_id_filename_map[image_id] = filepath.replace(
                    self.coco_root, "")
            self.save_pickle(self.img_id_filename_map,
                             self.img_id_filename_map_path)

        # Load {image_id: captions}
        try:
            self.img_id_caption_map = self.load_pickle(
                self.img_id_caption_map_path)
            print("Load image_id > caption map")
        except:
            for idx in tqdm(range(self.num_samples)):
                img_id, img, captions = self.reader[idx]
                self.img_id_caption_map[img_id] = captions
            # Save the caption map as pickle
            self.save_pickle(self.img_id_caption_map,
                             self.img_id_caption_map_path)

        # Load {image_id: encoding}
        try:
            self.img_id_encoding_map = self.load_pickle(
                self.img_id_encoding_map_path)
            print("Load image_id > encoding map")
        except:
            img_ids = list(self.img_id_caption_map.keys())
            sentences = []
            for img_id in img_ids:
                sentences.append(self.img_id_caption_map[img_id][0])
            encodings = self.encoder.encode(
                sentences,
                batch_size=64,
                show_progress_bar=True,
                convert_to_tensor=False,
            )
            for i in range(self.num_samples):
                self.img_id_encoding_map[img_ids[i]] = encodings[i]
            # Save the encoding map as pickle
            self.save_pickle(self.img_id_encoding_map,
                             self.img_id_encoding_map_path)

    def cluster(self, num_clusters, num_gpus=4):
        img_ids = list(self.img_id_encoding_map.keys())
        encodings = np.array(list(self.img_id_encoding_map.values()))

        d = encodings.shape[1]
        kmeans = faiss.Kmeans(
            d, num_clusters, niter=self.niter, verbose=True, gpu=num_gpus
        )
        kmeans.train(encodings)
        _, cluster_assignment = kmeans.index.search(encodings, 1)

        img_id_cluster_map = {}
        for i in tqdm(range(self.num_samples)):
            img_id_cluster_map[img_ids[i]] = cluster_assignment[i][0]

        split = str(self.split)
        img_id_cluster_map_path = os.path.join(
            self.cluster_root, f"img_id_cluster_map_{split}_{num_clusters}.pkl"
        )

        self.kmeans = kmeans
        self.save_pickle(img_id_cluster_map, img_id_cluster_map_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering for VLInfo")

    parser.add_argument(
        "--cluster-root",
        type=str,
        default="/bigtemp/as3ek/p/vlinfo/datasets/clusters/",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/bigtemp/as3ek/p/vlinfo/datasets/serialized/",
    )
    parser.add_argument(
        "--coco-root",
        type=str,
        default="/bigtemp/as3ek/p/vlinfo/datasets/coco/",
    )
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    folder_name = "clusters_" + str(args.max_clusters)
    cluster_root = os.path.join(args.cluster_root, folder_name)
    if not os.path.exists(cluster_root):
        os.makedirs(cluster_root)

    clust = CaptionClustering(
        split=args.split,
        cluster_root=cluster_root,
        data_root=args.data_root,
        coco_root=args.coco_root,
    )
    clust.extract_embeddings()

    for i in range(args.min_clusters, args.max_clusters + 1):
        clust.cluster(i)
