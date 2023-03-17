import os
import random
import pickle

from collections import defaultdict
import glob
from typing import Callable, Dict, List, Tuple
from PIL import Image

import albumentations as alb
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPTokenizer, BertTokenizer

from data.readers import LmdbReader
from data.tokenizers import SentencePieceBPETokenizer, GloveTokenizer
from transformers import AutoTokenizer, AutoModel, BertTokenizer
import data.transforms as T
import json
import glob
import re

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class RandomDataset(Dataset):
    r"""
    A dataset which provides randomly generated tensors
    Parameters
    ----------
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    """

    def __init__(
        self,
        data_root: str = "/bigtemp/as3ek/p/vlinfo/datasets/serialized2/",
        split: str = "train",
        mode: str = "train_sbert",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
        tokenizer_name: str = "bert-base-uncased",
    ):

        # Basic settings
        self.mode = mode
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.padding_idx = self.tokenizer.pad_token_id

        # Other settings
        self.max_caption_length = 30

    def __len__(self):
        return 118000

    def __getitem__(self, idx: int):

        captions = [
            "test caption",
            "test caption 2",
            "this is a caption",
            "these pretzels are making me thirsty",
        ]
        image = torch.rand(3, 224, 224)

        caption = random.choice(captions)

        encoded_input = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        )
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "input_ids": encoded_input["input_ids"][0],
            "attention_mask": encoded_input["attention_mask"][0],
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        # Pad `input_ids` and `attention_mask` up to max length.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [d["attention_mask"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class JsonDataset(Dataset):
    def __init__(
        self,
        json_files,
        data_root: str = "data/",
        split: str = "train",
        mode: str = "train_sbert",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
        tokenizer_name: str = "bert-base-uncased",
        visual_self_supervised: bool = False,
        textual_self_supervised: bool = False,
    ):
        # Basic settings
        self.ann = []
        for f in json_files:
            self.ann += json.load(open(f, 'r'))

        random.shuffle(self.ann)
        if percentage < 100.0:
            to_remove = int(((100.0 - percentage) / 100) * len(self.ann))
            self.ann = self.ann[to_remove:]

        if "bert" in tokenizer_name:
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
            self.padding_idx = self.tokenizer.pad_token_id

        # Initialize transforms
        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
            ]
        )

        # Other settings
        self.max_caption_length = max_caption_length
        self.use_single_caption = use_single_caption

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        if type(ann['caption']) == list:
            if self.use_single_caption:
                caption = caption[0]
            else:
                caption = random.choice(ann['caption'])
        else:
            caption = ann['caption']

        # Check if file exists at the image path
        if not os.path.isfile(ann['image']):
            print('Image not found at ' + str(ann['image']))

        # Open image from path and apply transformation, convert to CHW format.
        # # Alternative 1
        # image = cv2.imread(ann['image'])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # Image transforms
        # image_caption = self.image_transform(image=image, caption=caption)
        # image, caption = image_caption["image"], image_caption["caption"]
        # image = np.transpose(image, (2, 0, 1))

        # Alternative 2
        image = Image.open(ann['image']).convert('RGB')
        # Convert PIL image to numpy array
        image = np.array(image)
        # Apply transformations
        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))

        # Caption transforms
        caption = self.caption_transform(caption=caption)["caption"]
        encoded_input = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        )

        return_dict = {
            "image_id": torch.tensor(index, dtype=torch.long),
            "image": torch.tensor(image, dtype=torch.float),
            "input_ids": encoded_input["input_ids"][0],
            "attention_mask": encoded_input["attention_mask"][0]
        }

        return return_dict

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Pad `input_ids` and `attention_mask` up to max length.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [d["attention_mask"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        return_dict = {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return return_dict


class CocoCaptionsDataset(Dataset):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a serialized LMDB file (COCO Captions in this codebase). This is used for
    pretraining tasks which use captions - bicaptioning, forward captioning and
    token classification.
    This dataset also supports training on a randomly selected subset of the
    full dataset.
    Parameters
    ----------
    data_root: str, optional (default = "datasets/coco")
        Path to the dataset root directory. This must contain the serialized
        LMDB files (for COCO ``train2017`` and ``val2017`` splits).
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    tokenizer_name: Name of the tokenizer to be used with the text encoder
    image_transform: Callable, optional (default = data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`
        to be applied on the image.
    max_caption_length: int, optional (default = 30)
        Maximum number of tokens to keep in output caption tokens. Extra tokens
        will be trimmed from the right end of the token list.
    use_single_caption: bool, optional (default = False)
        COCO Captions provides five captions per image. If this is True, only
        one fixed caption per image is use fo training (used for an ablation).
    percentage: float, optional (default = 100.0)
        Randomly sample this much percentage of full dataset for training.
    """

    def __init__(
        self,
        data_root: str = "/bigtemp/as3ek/p/vlinfo/datasets/serialized2/",
        split: str = "train",
        mode: str = "train_sbert",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
        tokenizer_name: str = "bert-base-uncased",
        visual_self_supervised: bool = False,
        textual_self_supervised: bool = False,
    ):

        # Basic settings
        self.mode = mode
        self.split = split

        # Initialize reader that loads image, caption from lmdb
        lmdb_path = os.path.join(data_root, f"coco_{split}_{mode}2017.lmdb")
        self.reader = LmdbReader(lmdb_path, percentage=percentage)

        # Initialize tokenizer and padding_idx
        if self.mode == "glove":
            self.tokenizer = GloveTokenizer(
                "/bigtemp/as3ek/p/vlinfo/datasets/vocab/word_dict.json"
            )
            self.padding_idx = self.tokenizer.token_to_id("<pad>")

        elif self.mode == "train_sbert":
            if "bert" in tokenizer_name:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased')
                self.padding_idx = self.tokenizer.pad_token_id

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.padding_idx = self.tokenizer.pad_token_id

        # Initialize transforms
        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(max_caption_length),
            ]
        )

        # Other settings
        self.use_single_caption = use_single_caption
        self.max_caption_length = max_caption_length
        self.visual_self_supervised = visual_self_supervised
        self.textual_self_supervised = textual_self_supervised

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int):
        image_id, image, captions = self.reader[idx]

        # Choose one caption based on use single caption
        if self.use_single_caption:
            caption = captions[0]
        else:
            caption = random.choice(captions)

        # Placeholders that will be required later
        aug_image = image
        aug_caption = caption
        # Get another caption from the list
        while aug_caption == caption:
            aug_caption = random.choice(captions)

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))

        return_dict = {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": torch.tensor(image, dtype=torch.float),
        }

        # Return based on mode
        if self.mode == "glove":
            caption_tokens = self.caption_transform(caption=caption)["caption"]
            # Caption tokens
            return_dict["caption_tokens"] = torch.tensor(
                caption_tokens, dtype=torch.long
            )

            # Flipped caption tokens
            return_dict["noitpac_tokens"] = torch.tensor(
                caption_tokens, dtype=torch.long
            ).flip(0)

            # Caption lengths
            return_dict["caption_lengths"] = torch.tensor(
                len(caption_tokens), dtype=torch.long
            )

        elif self.mode == "sbert":
            return_dict["caption_encodings"] = torch.tensor(
                caption, dtype=torch.float)

        elif self.mode == "train_sbert":
            # Transform
            caption = self.caption_transform(caption=caption)["caption"]
            # Tokenize
            encoded_input = self.tokenizer(
                caption,
                padding=False,
                truncation=True,
                max_length=self.max_caption_length,
                return_tensors="pt",
            )
            return_dict["input_ids"] = encoded_input["input_ids"][0]
            return_dict["attention_mask"] = encoded_input["attention_mask"][0]

            # If data for text supervision is required
            if self.textual_self_supervised:
                # Transform
                aug_caption = self.caption_transform(
                    caption=aug_caption)["caption"]
                # Tokenize the augmented caption
                aug_encoded_input = self.tokenizer(
                    aug_caption,
                    padding=False,
                    truncation=True,
                    max_length=self.max_caption_length,
                    return_tensors="pt",
                )

                # Things to be returned
                return_dict["aug_input_ids"] = aug_encoded_input["input_ids"][0]
                return_dict["aug_attention_mask"] = aug_encoded_input["attention_mask"][
                    0
                ]

            # If data for visual self supervsion is required
            if self.visual_self_supervised:
                aug_image_caption = self.image_transform(
                    image=aug_image, caption=aug_caption
                )
                aug_image, aug_caption = (
                    aug_image_caption["image"],
                    aug_image_caption["caption"],
                )
                aug_image = np.transpose(aug_image, (2, 0, 1))

                # Things to be returned
                return_dict["aug_image"] = torch.tensor(
                    aug_image, dtype=torch.float)

        return return_dict

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        if self.mode == "glove":
            # Pad `caption_tokens` and `masked_labels` up to this length.
            caption_tokens = torch.nn.utils.rnn.pad_sequence(
                [d["caption_tokens"] for d in data],
                batch_first=True,
                padding_value=self.padding_idx,
            )
            noitpac_tokens = torch.nn.utils.rnn.pad_sequence(
                [d["noitpac_tokens"] for d in data],
                batch_first=True,
                padding_value=self.padding_idx,
            )

            return {
                "image_id": torch.stack([d["image_id"] for d in data], dim=0),
                "image": torch.stack([d["image"] for d in data], dim=0),
                "caption_tokens": caption_tokens,
                "noitpac_tokens": noitpac_tokens,
                "caption_lengths": torch.stack([d["caption_lengths"] for d in data]),
            }

        elif self.mode == "train_sbert":
            # Pad `input_ids` and `attention_mask` up to max length.
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [d["input_ids"] for d in data],
                batch_first=True,
                padding_value=self.padding_idx,
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [d["attention_mask"] for d in data],
                batch_first=True,
                padding_value=self.padding_idx,
            )

            return_dict = {
                "image_id": torch.stack([d["image_id"] for d in data], dim=0),
                "image": torch.stack([d["image"] for d in data], dim=0),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if self.visual_self_supervised:
                return_dict["aug_image"] = torch.stack(
                    [d["aug_image"] for d in data], dim=0
                )

            if self.textual_self_supervised:
                aug_input_ids = torch.nn.utils.rnn.pad_sequence(
                    [d["aug_input_ids"] for d in data],
                    batch_first=True,
                    padding_value=self.padding_idx,
                )
                aug_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    [d["aug_attention_mask"] for d in data],
                    batch_first=True,
                    padding_value=self.padding_idx,
                )

                return_dict["aug_input_ids"] = aug_input_ids
                return_dict["aug_attention_mask"] = aug_attention_mask

            return return_dict


class CocoCaptionsClusteredDataset(Dataset):
    r"""
    A dataset which provides image-caption-neg_caption-neg_image elements from
    a serialized LMDB file (COCO Captions in this codebase). This is used for
    pretraining tasks which use captions - bicaptioning, forward captioning and
    token classification.
    This dataset also supports training on a randomly selected subset of the
    full dataset.
    Parameters
    ----------
    data_root: str, optional (default = "datasets/coco")
        Path to the dataset root directory. This must contain the serialized
        LMDB files (for COCO ``train2017`` and ``val2017`` splits).
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    tokenizer: data.tokenizers.SentencePieceBPETokenizer
        A tokenizer which has the mapping between word tokens and their
        integer IDs.
    image_transform: Callable, optional (default = data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`
        to be applied on the image.
    max_caption_length: int, optional (default = 30)
        Maximum number of tokens to keep in output caption tokens. Extra tokens
        will be trimmed from the right end of the token list.
    use_single_caption: bool, optional (default = False)
        COCO Captions provides five captions per image. If this is True, only
        one fixed caption per image is use fo training (used for an ablation).
    percentage: float, optional (default = 100.0)
        Randomly sample this much percentage of full dataset for training.
    """

    def __init__(
        self,
        data_root: str = "/bigtemp/as3ek/p/vlinfo/datasets/serialized/",
        split: str = "train",
        mode: str = "train_sbert",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_caption_length: int = 30,
        use_single_caption: bool = False,
        percentage: float = 100.0,
        tokenizer_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        negative_sampling: str = "clusters",
        total_iters: int = 500000,
        negative_sampling_start_iter: int = 250000,
        cluster_path: str = "/bigtemp/as3ek/p/vlinfo/datasets/clusters/",
        coco_root: str = "/bigtemp/as3ek/p/vlinfo/datasets/coco/",
    ):
        # Basic settings
        self.mode = mode
        self.split = split
        self.coco_root = coco_root

        # Initialize negative_sampling settings
        self.cluster_path = cluster_path
        # Cluster options is a list of options for the number of clusters
        # ex: [2, 3, 4, 5, 10]
        self.cluster_options = self.get_cluster_options(cluster_path)
        self.iter_num = 0
        self.total_iters = total_iters
        self.negative_sampling_start_iter = negative_sampling_start_iter
        self.current_cluster_num = -1
        self.negative_sampling = negative_sampling

        # Initialize reader that loads image, caption from lmdb
        lmdb_path = os.path.join(data_root, f"coco_{split}_{mode}2017.lmdb")
        self.reader = LmdbReader(lmdb_path, percentage=percentage)

        # Initialize tokenizer and padding_idx
        if self.mode == "glove":
            self.tokenizer = GloveTokenizer(
                "/u/as3ek/github/vlinfo/data/datasets/vocab/word_dict.json"
            )
            self.padding_idx = self.tokenizer.token_to_id("<pad>")

        elif self.mode == "train_sbert":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.padding_idx = self.tokenizer.pad_token_id

        # Initialize transforms
        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption()
            ]
        )

        # Other settings
        self.use_single_caption = use_single_caption
        self.max_caption_length = max_caption_length

    def update_iter(self, iter_num):
        self.iter_num = iter_num

    def save_pickle(self, data, path):
        with open(path, "wb") as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        return data

    def get_cluster_options(self, cluster_path):
        cluster_options = []
        split = self.split

        all_files = os.listdir(cluster_path)
        for file in all_files:
            if f"img_id_cluster_map_{split}" in file:
                cluster_num = int(file.split("_")[-1].replace(".pkl", ""))
                cluster_options.append(cluster_num)

        return cluster_options

    def get_coco_maps(self):
        """Returns {img_id: caption} and {img_id: img_filename}"""
        split = self.split
        # Load map between image_id and captions
        img_id_caption_map_path = os.path.join(
            self.cluster_path,
            f"img_id_caption_map_{split}.pkl",
        )
        img_id_caption_map = self.load_pickle(img_id_caption_map_path)

        # Load map between image_id and filename,
        # to be used to load the negative image
        img_id_filename_map_path = os.path.join(
            self.cluster_path,
            f"img_id_filename_map_{split}.pkl",
        )
        img_id_filename_map = self.load_pickle(img_id_filename_map_path)

        return img_id_caption_map, img_id_filename_map

    def get_coco_item(self, image_id):
        split = self.split
        filename = self.img_id_filename_map[image_id]
        filepath = os.path.join(self.coco_root, filename)
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        captions = self.img_id_caption_map[image_id]
        caption = random.choice(captions)

        return {"image_id": image_id, "image": image, "caption": caption}

    def get_cluster_maps(self, num_clusters):
        split = self.split
        # Dictionary map between img_id and its cluster id
        # {image_id: clutser_id}
        img_id_cluster_map_path = os.path.join(
            self.cluster_path, f"img_id_cluster_map_{split}_{num_clusters}.pkl"
        )
        img_id_cluster_map = self.load_pickle(img_id_cluster_map_path)

        # Dictionary map between cluster_id and all the img_ids in the cluster
        # {cluster_id: [img_id1, img_id2, ...]}
        cluster_img_ids_map = {}
        for img_id, cluster in img_id_cluster_map.items():
            # If cluster is not in map, initialize it
            if cluster not in cluster_img_ids_map:
                cluster_img_ids_map[cluster] = []
            cluster_img_ids_map[cluster].append(img_id)

        return img_id_cluster_map, cluster_img_ids_map

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int):
        # Get data from reader
        image_id, image, captions = self.reader[idx]
        # Choose one caption based on use single caption
        if self.use_single_caption:
            caption = captions[0]
        else:
            caption = random.choice(captions)
        # Estimante the required number of clusters
        # Considering linear shrinking clusters
        pred_num_cluster = int(max(self.cluster_options)) * (
            (self.iter_num - self.negative_sampling_start_iter)
            / (self.total_iters - self.negative_sampling_start_iter)
        )
        # Select the correct number of clusters from the options
        # closest to the predicted number cosidering linear shrinking
        num_clusters = min(
            self.cluster_options, key=lambda x: abs(x - pred_num_cluster)
        )
        # Check if a new img_id_cluster_map needs to be loaded
        if self.current_cluster_num != num_clusters:
            # If performing clustering 1st time,
            # load image_id -> payload{filename, caption} maps
            if self.current_cluster_num == -1:
                (
                    self.img_id_caption_map,
                    self.img_id_filename_map,
                ) = self.get_coco_maps()

            (
                self.img_id_cluster_map,
                self.cluster_img_ids_map,
            ) = self.get_cluster_maps(num_clusters)

            # Update current cluster number
            self.current_cluster_num = num_clusters

        # Get cluster id for the current image-caption
        cluster_id = self.img_id_cluster_map[image_id]
        # Get a random image id from the above cluster
        # this will be our negative sample
        neg_image_id = random.choice(self.cluster_img_ids_map[cluster_id])
        while neg_image_id == image_id:
            neg_image_id = random.choice(self.cluster_img_ids_map[cluster_id])

        # Get the image and the caption for the neg_image_id
        neg_data = self.get_coco_item(neg_image_id)
        neg_image_id, neg_image, neg_caption = (
            neg_data["image_id"],
            neg_data["image"],
            neg_data["caption"],
        )

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        # 1) For the positive sample
        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))
        caption = self.caption_transform(caption=caption)["caption"]

        encoded_input = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        )

        # 2) For the negative sample
        neg_image_caption = self.image_transform(
            image=neg_image, caption=neg_caption)
        neg_image, neg_caption = (
            neg_image_caption["image"],
            neg_image_caption["caption"],
        )
        neg_image = np.transpose(neg_image, (2, 0, 1))
        neg_caption = self.caption_transform(caption=neg_caption)["caption"]

        neg_encoded_input = self.tokenizer(
            neg_caption,
            padding=False,
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        )

        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": torch.tensor(image, dtype=torch.float),
            "input_ids": encoded_input["input_ids"][0],
            "attention_mask": encoded_input["attention_mask"][0],
            "neg_image": torch.tensor(neg_image, dtype=torch.float),
            "neg_input_ids": neg_encoded_input["input_ids"][0],
            "neg_attention_mask": neg_encoded_input["attention_mask"][0],
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        # Pad `input_ids` and `attention_mask` up to max length.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [d["attention_mask"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        neg_input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["neg_input_ids"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        neg_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [d["neg_attention_mask"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        return {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "neg_image": torch.stack([d["neg_image"] for d in data], dim=0),
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
        }


class VOC07ClassificationDataset(Dataset):
    r"""
    A dataset which provides image-label pairs from the PASCAL VOC 2007 dataset.
    Parameters
    ----------
    data_root: str, optional (default = "datasets/VOC2007")
        Path to the dataset root directory. This must contain directories
        ``Annotations``, ``ImageSets`` and ``JPEGImages``.
    split: str, optional (default = "trainval")
        Which split to read from. One of ``{"trainval", "test"}``.
    image_transform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    """

    def __init__(
        self,
        data_root: str = "/u/as3ek/github/vlinfo/data/datasets/voc07/",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        self.split = split
        self.image_transform = image_transform

        ann_paths = sorted(
            glob.glob(os.path.join(data_root, "ImageSets",
                      "Main", f"*_{split}.txt"))
        )
        # A list like; ["aeroplane", "bicycle", "bird", ...]
        self.class_names = [os.path.basename(path).split("_")[
            0] for path in ann_paths]

        # We will construct a map for image name to a list of
        # shape: (num_classes, ) and values as one of {-1, 0, 1}.
        # 1: present, -1: not present, 0: ignore.
        image_names_to_labels: Dict[str, torch.Tensor] = defaultdict(
            lambda: -torch.ones(len(self.class_names), dtype=torch.int32)
        )
        for cls_num, ann_path in enumerate(ann_paths):
            with open(ann_path, "r") as fopen:
                for line in fopen:
                    img_name, orig_label_str = line.strip().split()
                    orig_label = int(orig_label_str)

                    # In VOC data, -1 (not present): set to 0 as train target
                    # In VOC data, 0 (ignore): set to -1 as train target.
                    orig_label = 0 if orig_label == -1 else -1 if orig_label == 0 else 1
                    image_names_to_labels[img_name][cls_num] = orig_label

        # Convert the dict to a list of tuples for easy indexing.
        # Replace image name with full image path.
        self.instances: List[Tuple[str, torch.Tensor]] = [
            (
                os.path.join(data_root, "JPEGImages", f"{image_name}.jpg"),
                label.tolist(),
            )
            for image_name, label in image_names_to_labels.items()
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_path, label = self.instances[idx]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "label": torch.stack([d["label"] for d in data], dim=0),
        }


class INaturalist2018Dataset(Dataset):
    r"""
    A dataset which provides image-label pairs from the iNaturalist 2018 dataset.
    Parameters
    ----------
    data_root: str, optional (default = "datasets/inaturalist")
        Path to the dataset root directory. This must contain images and
        annotations (``train2018``, ``val2018`` and ``annotations`` directories).
    split: str, optional (default = "train")
        Which split to read from. One of ``{"train", "val"}``.
    image_transform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    """

    def __init__(
        self,
        data_root: str = "datasets/inaturalist",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        self.split = split
        self.image_transform = image_transform

        annotations = json.load(
            open(os.path.join(data_root, "annotations", f"{split}2018.json"))
        )
        # Make a list of image IDs to file paths.
        self.image_id_to_file_path = {
            ann["id"]: os.path.join(data_root, ann["file_name"])
            for ann in annotations["images"]
        }
        # For a list of instances: (image_id, category_id) tuples.
        self.instances = [
            (ann["image_id"], ann["category_id"]) for ann in annotations["annotations"]
        ]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        image_id, label = self.instances[idx]
        image_path = self.image_id_to_file_path[image_id]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "label": torch.stack([d["label"] for d in data], dim=0),
        }


class ImageNetDataset(ImageNet):
    r"""
    Simple wrapper over torchvision's ImageNet dataset with a feature to support
    restricting dataset size for semi-supervised learning setup (data-efficiency
    ablations).
    We also handle image transform here instead of passing to super class.
    Parameters
    ----------
    data_root: str, optional (default = "datasets/imagenet")
        Path to the dataset root directory. This must contain directories
        ``train``, ``val`` with per-category sub-directories.
    split: str, optional (default = "train")
        Which split to read from. One of ``{"train", "val"}``.
    image_transform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    percentage: int, optional (default = 100)
        Percentage of dataset to keep. This dataset retains first K% of images
        per class to retain same class label distribution. This is 100% by
        default, and will be ignored if ``split`` is ``val``.
    """

    def __init__(
        self,
        data_root: str = "datasets/imagenet",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        percentage: float = 100,
    ):
        super().__init__(data_root, split)
        assert percentage > 0, "Cannot load dataset with 0 percent original size."

        self.image_transform = image_transform

        # Super class has `imgs` list and `targets` list. Make a dict of
        # class ID to index of instances in these lists and pick first K%.
        if split == "train" and percentage < 100:
            label_to_indices: Dict[int, List[int]] = defaultdict(list)
            for index, target in enumerate(self.targets):
                label_to_indices[target].append(index)

            # Trim list of indices per label.
            for label in label_to_indices:
                retain = int(len(label_to_indices[label]) * (percentage / 100))
                label_to_indices[label] = label_to_indices[label][:retain]

            # Trim `self.imgs` and `self.targets` as per indices we have.
            retained_indices: List[int] = [
                index
                for indices_per_label in label_to_indices.values()
                for index in indices_per_label
            ]
            # Shorter dataset with size K% of original dataset, but almost same
            # class label distribution. super class will handle the rest.
            self.imgs = [self.imgs[i] for i in retained_indices]
            self.targets = [self.targets[i] for i in retained_indices]
            self.samples = self.imgs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, label = super().__getitem__(idx)

        # Apply transformation to  image and convert to CHW format.
        image = self.image_transform(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1))
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "label": torch.stack([d["label"] for d in data], dim=0),
        }


def pre_caption(caption, max_words):
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


class ReEvalDataset(Dataset):
    def __init__(
        self,
        data_root: str = "datasets/coco",
        ann_file: str = "",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        max_words: int = 30,
    ):

        self.transform = image_transform
        self.data_root = data_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        # create img2file
        root = data_root
        image_dir = os.path.join(root, f"{split}2017")
        # Create tuple (image_id, filename) using COCO2017 format
        image_filenames = glob.glob(os.path.join(image_dir, "*.jpg"))

        self.id_filename = [
            (int(os.path.basename(name)[:-4]), name) for name in image_filenames
        ]

        # Create mapping between image and captions
        captions = json.load(
            open(os.path.join(root, "annotations",
                 f"captions_{split}2017.json"))
        )
        self.id_to_captions = defaultdict(list)
        for annotation in captions["annotations"]:
            self.id_to_captions[annotation["image_id"]].append(
                annotation["caption"])

        txt_id = 0
        for img_id, img_path in self.id_filename:
            self.image.append(img_path)
            captions = self.id_to_captions[img_id]
            self.img2txt[img_id] = []
            for i, caption in enumerate(captions):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def save_pickle(self, data, path):
        with open(path, "wb") as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        return data

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_id, filename = self.id_filename[index]

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)

        return image, image_id


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, image_transform, data_root, max_words=30, split='val'):

        self.ann = json.load(open(ann_file, 'r'))
        self.transform = image_transform
        self.image_root = data_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)

        return image, index


class CocoObjectGender(Dataset):
    def __init__(
        self,
        data_root="/bigtemp/as3ek/p/vlinfo/datasets/coco/",
        annotation_dir="/bigtemp/as3ek/p/vlinfo/datasets/coco/",
        gender_annotation_dir="/bigtemp/as3ek/p/vlinfo/datasets/coco_gender/",
        image_dir="/bigtemp/as3ek/p/vlinfo/datasets/coco/",
        split="train",
        image_transform=None,
        balanced_train=False,
        balanced_val=False,
        balanced_test=True,
        ratio=1,
        num_object=79,
        gender_balanced=False,
        blackout=False,
        blackout_box=False,
        blur=False,
        grayscale=False,
        edges=False,
        blackout_face=False,
        tokenizer_name="bert-base-uncased",
    ):

        self.split = split
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_transform = image_transform
        self.balanced_train = balanced_train
        self.balanced_val = balanced_val
        self.balanced_test = balanced_test
        self.ratio = ratio
        self.num_object = num_object
        self.gender_balanced = gender_balanced
        self.blackout = blackout
        self.blackout_box = blackout_box
        self.blur = blur
        self.grayscale = grayscale
        self.edges = edges
        self.blackout_face = blackout_face
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self.padding_idx = self.tokenizer.pad_token_id

        print("loading %s annotations.........." % self.split)
        self.ann_data = self.load_pickle(
            os.path.join(gender_annotation_dir, split + ".data")
        )

        if self.balanced_train and split == "train":
            path = os.path.join(
                gender_annotation_dir, "{}_ratio_{}.ids".format(
                    split, self.ratio)
            )
            balanced_subset = self.load_pickle(path)
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_val and split == "val":
            path = os.path.join(
                gender_annotation_dir, "{}_ratio_{}.ids".format(
                    split, self.ratio)
            )
            balanced_subset = self.load_pickle(path)
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_test and split == "test":
            path = os.path.join(
                gender_annotation_dir, "{}_ratio_{}.ids".format(
                    split, self.ratio)
            )
            balanced_subset = self.load_pickle(path)
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        self.object_ann = np.zeros((len(self.ann_data), self.num_object))
        self.gender_ann = np.zeros((len(self.ann_data), 2), dtype=int)
        for index, ann in enumerate(self.ann_data):
            self.object_ann[index] = np.asarray(ann["objects"])
            self.gender_ann[index] = np.asarray(ann["gender"])

        if self.gender_balanced:
            man_idxs = np.nonzero(self.gender_ann[:, 0])[0]
            woman_idxs = np.nonzero(self.gender_ann[:, 1])[0]
            random.shuffle(man_idxs)  # need to do random sample every time
            random.shuffle(woman_idxs)
            min_len = 30000 if split == "train" else 1500
            selected_idxs = list(
                man_idxs[:min_len]) + list(woman_idxs[:min_len])

            self.ann_data = [self.ann_data[idx] for idx in selected_idxs]
            self.object_ann = np.take(self.object_ann, selected_idxs, axis=0)
            self.gender_ann = np.take(self.gender_ann, selected_idxs, axis=0)

        print(
            "man size : {} and woman size: {}".format(
                len(np.nonzero(self.gender_ann[:, 0])[0]),
                len(np.nonzero(self.gender_ann[:, 1])[0]),
            )
        )
        # Load captions
        split = self.split
        img_id_caption_map_path = os.path.join(
            "/bigtemp/as3ek/p/vlinfo/datasets/clusters/clusters_3",
            f"img_id_caption_map_{split}.pkl",
        )
        self.img_id2caption = self.load_pickle(img_id_caption_map_path)

        # load mask annotations
        if (
            self.blackout
            or self.blackout_box
            or self.blur
            or self.grayscale
            or self.edges
        ):
            self.cocoAnnDir = os.path.join(self.annotation_dir, "annotations")
            if self.split == "train":
                self.root = os.path.join(self.image_dir, "train2017")
                self.captionFile = os.path.join(
                    self.cocoAnnDir, "captions_train2017.json"
                )
                self.annFile = os.path.join(
                    self.cocoAnnDir, "instances_train2017.json")
            else:
                self.root = os.path.join(self.image_dir, "val2017")
                self.captionFile = os.path.join(
                    self.cocoAnnDir, "captions_val2017.json"
                )
                self.annFile = os.path.join(
                    self.cocoAnnDir, "instances_val2017.json")

            self.cocoAPI = COCO(self.annFile)

    def __getitem__(self, index):
        img = self.ann_data[index]
        img_id = img["image_id"]
        img_file_name = img["file_name"]
        captions = self.img_id2caption[img_id]
        caption = random.choice(captions)

        if self.split == "train":
            image_path_ = os.path.join(
                self.image_dir, "train2017", img_file_name.split("_")[-1]
            )
        else:
            image_path_ = os.path.join(
                self.image_dir, "val2017", img_file_name.split("_")[-1]
            )

        img_ = cv2.imread(image_path_)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        if self.blackout:
            ann_ids = self.cocoAPI.getAnnIds(imgIds=img_id)
            img_ = self.do_blackout(img_, ann_ids, "people")
        elif self.blackout_box:
            ann_ids = self.cocoAPI.getAnnIds(imgIds=img_id)
            img_ = self.do_blackout(img_, ann_ids, "people_box")
        elif self.blur:
            ann_ids = self.cocoAPI.getAnnIds(imgIds=img_id)
            img_ = self.do_blur(img_, ann_ids, "people")
        elif self.grayscale:
            ann_ids = self.cocoAPI.getAnnIds(imgIds=img_id)
            img_ = self.do_grey(img_, ann_ids)
        elif self.edges:
            ann_ids = self.cocoAPI.getAnnIds(imgIds=img_id)
            img_ = self.do_find_edges(img_, ann_ids)
        elif self.blackout_face:
            img_ = self.do_blackout_face(img_, img_id)

        if self.image_transform is not None:
            img_caption = self.image_transform(image=img_, caption=caption)
            img_ = img_caption["image"]
            caption = img_caption["caption"]

        img_ = np.transpose(img_, (2, 0, 1))

        encoded_input = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=30,
            return_tensors="pt",
        )

        return {
            "image_id": torch.tensor(img_id, dtype=torch.long),
            "image": torch.tensor(img_, dtype=torch.float),
            "gender": torch.tensor(self.gender_ann[index], dtype=torch.long),
            "input_ids": encoded_input["input_ids"][0],
            "attention_mask": encoded_input["attention_mask"][0],
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Pad `input_ids` and `attention_mask` up to max length.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [d["input_ids"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [d["attention_mask"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        return {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "gender": torch.stack([d["gender"] for d in data], dim=0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def save_pickle(self, data, path):
        with open(path, "wb") as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        return data

    def getGenderWeights(self):
        return (self.gender_ann == 0).sum(axis=0) / (
            1e-15 + (self.gender_ann.sum(axis=0) +
                     (self.gender_ann == 0).sum(axis=0))
        )

    def getObjectWeights(self):
        return (self.object_ann == 0).sum(axis=0) / (
            1e-15 + self.object_ann.sum(axis=0)
        )

    def __len__(self):
        return len(self.ann_data)

    def do_find_edges(self, img, ann_ids):
        anns = [
            ann for ann in self.cocoAPI.loadAnns(ann_ids) if ann["category_id"] == 1
        ]
        if len(anns) > 0:
            outlined_img = img.convert(mode="L").filter(ImageFilter.CONTOUR)
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            img_mask = Image.fromarray(255 * (mask > 0).astype("uint8"))
            return Image.composite(outlined_img, img, img_mask)
        return img

    def do_grey(self, img, ann_ids):
        anns = [
            ann for ann in self.cocoAPI.loadAnns(ann_ids) if ann["category_id"] == 1
        ]
        if len(anns) > 0:
            grey_img = img.convert(mode="L")
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            img_mask = Image.fromarray(255 * (mask > 0).astype("uint8"))
            return Image.composite(grey_img, img, img_mask)
        return img

    def do_blur(self, img, ann_ids, processed_area):
        # Only people category, category_id == 1.
        anns = [
            ann for ann in self.cocoAPI.loadAnns(ann_ids) if ann["category_id"] == 1
        ]
        if len(anns) > 0:
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=10))
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            if processed_area == "people":
                img_mask = Image.fromarray(255 * (mask > 0).astype("uint8"))
            elif processed_area == "background":
                img_mask = Image.fromarray(255 * (mask == 0).astype("uint8"))
            else:
                print("Please specify blur people or background")
            return Image.composite(blurred_img, img, img_mask)
        return img

    def do_blackout(self, img, ann_ids, processed_area):
        # Only people category, category_id == 1.
        anns = [
            ann for ann in self.cocoAPI.loadAnns(ann_ids) if ann["category_id"] == 1
        ]
        if len(anns) > 0:
            black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
            mask = self.cocoAPI.annToMask(anns[0])
            for ann in anns[1:]:
                mask = mask + self.cocoAPI.annToMask(ann)
            if processed_area == "people_box":
                self.box_mask(mask)
            if processed_area == "people" or processed_area == "people_box":
                img_mask = Image.fromarray(255 * (mask > 0).astype("uint8"))
            elif processed_area == "background":
                img_mask = Image.fromarray(255 * (mask == 0).astype("uint8"))
            else:
                print("Please specify blackout people or background")
            return Image.composite(black_img, img, img_mask)
        return img

    def do_box_mask(self, mask):
        x_limits = np.nonzero(mask.sum(axis=0))
        xmin = x_limits[0][0]
        xmax = x_limits[0][-1]
        y_limits = np.nonzero(mask.sum(axis=1))
        ymin = y_limits[0][0]
        ymax = y_limits[0][-1]
        mask[ymin:ymax, xmin:xmax] = 1

    def do_blackout_face(self, img, img_name):
        try:
            vertices = self.faces[int(img_name)]
        except:
            return img

        width = img.size[1]
        height = img.size[0]

        black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
        mask = np.zeros((width, height))
        for poly in vertices:
            xmin, ymin = poly[0].strip("()").split(",")
            xmax, ymax = poly[2].strip("()").split(",")
            for i in range(int(xmin), int(xmax)):
                for j in range(int(ymin), int(ymax)):
                    mask[j][i] = 1
        img_mask = Image.fromarray(255 * (mask > 0).astype("uint8")).resize(
            (img.size[0], img.size[1]), Image.ANTIALIAS
        )

        return Image.composite(black_img, img, img_mask)
