import random
import pandas as pd
import numpy as np
import argparse
import json
import os
import unicodedata
import tempfile

from typing import List
from collections import Counter
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
nltk.download('punkt')

parser = argparse.ArgumentParser(
    description="""Build a vocabulary out of captions corpus. This vocabulary
    would be a file which our tokenizer can understand.
    """
)
parser.add_argument(
    "-c", "--captions", default="/u/as3ek/github/vlinfo/data/datasets/coco/annotations/captions_train2017.json",
    help="Path to caption annotations file in COCO format.",
)
parser.add_argument(
    "-g", "--glove_path", default="/u/as3ek/github/vlinfo/data/datasets/glove/glove.42B.300d.txt",
    help="Path to the downloaded glove encodings file.",
)
## Not doing anything
parser.add_argument(
    "-s", "--vocab-size", type=int, default=10000,
    help="Total desired size of our vocabulary.",
)
parser.add_argument(
    "-o", "--output-folder", default="/u/as3ek/github/vlinfo/data/datasets/vocab/",
    help="Location where the word_dict.json will be saved",
)
parser.add_argument(
    "-l", "--do-lower-case", action="store_true",
    help="Whether to lower case the captions before forming vocabulary.",
)
parser.add_argument(
    "-a", "--keep-accents", action="store_true",
    help="Whether to keep accents before forming vocabulary (dropped by default).",
)

def _read_captions(annotations_path: str) -> List[str]:
    r"""
    Given a path to annotation file, read it and return a list of captions.
    These are not processed by any means, returned from the file as-is.
    Parameters
    ----------
    annotations_path: str
        Path to an annotations file containing captions.
    Returns
    -------
    List[str]
        List of captions from this annotation file.
    """

    _annotations = json.load(open(annotations_path))

    captions: List[str] = []
    for ann in _annotations["annotations"]:
        captions.append(ann["caption"])

    return captions

def load_glove_model(glove_path):
    print("Loading Glove Model")
    f = open(glove_path, 'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel, len(wordEmbedding)


if __name__ == "__main__":
    args = parser.parse_args()
    captions: List[str] = _read_captions(args.captions)

    # Initialize word counter for vocabulary
    word_counter = Counter()

    for i, caption in enumerate(captions):
        caption = caption.lower() if args.do_lower_case else caption

        if not args.keep_accents:
            caption = unicodedata.normalize("NFKD", caption)
            caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])

        tokens = word_tokenize(caption)
        word_counter.update(tokens)

    words = [word for word in word_counter.keys()]

    # Remove words that do not exist in glove embeddings
    glove_model, _ = load_glove_model(args.glove_path)
    words_glove = []
    for word in words:
        if word in glove_model:
            words_glove.append(word)

    word_dict = {word: idx + 4 for idx, word in enumerate(words_glove)}

    word_dict['<start>'] = 0
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2
    word_dict['<pad>'] = 3

    with open(args.output_folder + 'word_dict.json', 'w') as f:
        json.dump(word_dict, f)
