from __future__ import print_function, division
import re
import sys
import numpy as np
import scipy.sparse
import torch
from sklearn.decomposition import PCA

if sys.version_info[0] < 3:
    import io

    open = io.open
else:
    unicode = str
"""
Tools for debiasing word embeddings
Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

DEFAULT_NUM_texts = 27000
FILENAMES = {
    "g_wiki": "glove.6B.300d.small.txt",
    "g_twitter": "glove.twitter.27B.200d.small.txt",
    "g_crawl": "glove.840B.300d.small.txt",
    "w2v": "GoogleNews-word2vec.small.txt",
    "w2v_large": "GoogleNews-word2vec.txt",
}


def dedup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def safe_text(w):
    # ignore texts with numbers, etc.
    # [a-zA-Z\.'_\- :;\(\)\]] for emoticons
    return re.match(r"^[a-z_]*$", w) and len(w) < 20 and not re.match(r"^_*$", w)


def to_utf8(text, errors="strict", encoding="utf8"):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode("utf8")
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode("utf8")


class Embedding:
    def __init__(
        self,
        text_encoder,
        tokenizer,
        device,
        text_projector=None,
        loss_type="dot",
    ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_projector = text_projector
        self.loss_type = loss_type
        self.max_texts = None
        self.vecs = {}
        self.texts = []
        self.num_texts = 0
        self.index = {}
        self.device = device

    def reindex(self):
        self.index = {w: i for i, w in enumerate(self.texts)}

    def encode_text(self, text):
        with torch.no_grad():
            # Tokenize the text
            encoded_input = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=30,
                return_tensors="pt",
            )
            input_ids = encoded_input["input_ids"].to(self.device)
            attention_mask = encoded_input["attention_mask"].to(self.device)

            # Encode the prompt text
            features = self.text_encoder(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

            if self.loss_type == "dot":
                features = self.text_projector(features)

            return features.cpu().numpy()[0]

    def v(self, text):
        if text in self.index:
            return self.vecs[self.index[text]]
        else:
            vec = self.encode_text(text)
            # Set index of word to num_words
            self.texts.append(text)
            self.index[text] = len(self.texts)
            self.vecs[len(self.texts)] = vec
            self.num_texts = len(self.texts)

            return vec

    def diff(self, text1, text2):
        v = self.vecs[self.index[text1]] - self.vecs[self.index[text2]]
        return v / np.linalg.norm(v)

    def normalize(self):
        self.desc += ", normalize"
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
        self.reindex()

    def shrink(self, numtexts):
        self.desc += ", shrink " + str(numtexts)
        self.filter_texts(lambda w: self.index[w] < numtexts)

    def filter_texts(self, test):
        """
        Keep some texts based on test, e.g. lambda x: x.lower()==x
        """
        self.desc += ", filter"
        kept_indices, texts = zip(
            *[[i, w] for i, w in enumerate(self.texts) if test(w)]
        )
        self.texts = list(texts)
        self.vecs = self.vecs[kept_indices, :]
        self.reindex()

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(
                "\n".join(
                    [
                        w + " " + " ".join([str(x) for x in v])
                        for w, v in zip(self.texts, self.vecs)
                    ]
                )
            )
        print("Wrote", self.n, "texts to", filename)

    def save_w2v(self, filename, binary=True):
        with open(filename, "wb") as fout:
            fout.write(to_utf8("%s %s\n" % self.vecs.shape))
            # store in sorted order: most frequent texts at the top
            for i, text in enumerate(self.texts):
                row = self.vecs[i]
                if binary:
                    fout.write(to_utf8(text) + b" " + row.tostring())
                else:
                    fout.write(
                        to_utf8("%s %s\n" % (text, " ".join("%f" % val for val in row)))
                    )

    def remove_directions(self, directions):  # directions better be orthogonal
        self.desc += ", removed"
        for direction in directions:
            self.desc += " "
            if type(direction) is np.ndarray:
                v = direction / np.linalg.norm(direction)
                self.desc += "vector "
            else:
                w1, w2 = direction
                v = self.diff(w1, w2)
                self.desc += w1 + "-" + w2
            self.vecs = self.vecs - self.vecs.dot(v)[:, np.newaxis].dot(
                v[np.newaxis, :]
            )
        self.normalize()

    def compute_neighbors_if_necessary(self, thresh, max_texts):
        thresh = float(thresh)  # dang python 2.7!
        if (
            self._neighbors is not None
            and self.thresh == thresh
            and self.max_texts == max_texts
        ):
            return
        print("Computing neighbors")
        self.thresh = thresh
        self.max_texts = max_texts
        vecs = self.vecs[:max_texts]
        dots = vecs.dot(vecs.T)
        dots = scipy.sparse.csr_matrix(dots * (dots >= 1 - thresh / 2))
        from collections import Counter

        rows, cols = dots.nonzero()
        nums = list(Counter(rows).values())
        print("Mean:", np.mean(nums) - 1)
        print("Median:", np.median(nums) - 1)
        rows, cols, vecs = zip(
            *[
                (i, j, vecs[i] - vecs[j])
                for i, j, x in zip(rows, cols, dots.data)
                if i < j
            ]
        )
        self._neighbors = rows, cols, np.array([v / np.linalg.norm(v) for v in vecs])

    def neighbors(self, text, thresh=1):
        dots = self.vecs.dot(self.v(text))
        return [self.texts[i] for i, dot in enumerate(dots) if dot >= 1 - thresh / 2]

    def more_texts_like_these(self, texts, topn=50, max_freq=100000):
        v = sum(self.v(w) for w in texts)
        dots = self.vecs[:max_freq].dot(v)
        thresh = sorted(dots)[-topn]
        texts = [w for w, dot in zip(self.texts, dots) if dot >= thresh]
        return sorted(texts, key=lambda w: self.v(w).dot(v))[-topn:][::-1]

    def best_analogies_dist_thresh(self, v, thresh=1, topn=500, max_texts=50000):
        """Metric is cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0"""
        vecs, vocab = self.vecs[:max_texts], self.texts[:max_texts]
        self.compute_neighbors_if_necessary(thresh, max_texts)
        rows, cols, vecs = self._neighbors
        scores = vecs.dot(v / np.linalg.norm(v))
        pi = np.argsort(-abs(scores))

        ans = []
        usedL = set()
        usedR = set()
        for i in pi:
            if abs(scores[i]) < 0.001:
                break
            row = rows[i] if scores[i] > 0 else cols[i]
            col = cols[i] if scores[i] > 0 else rows[i]
            if row in usedL or col in usedR:
                continue
            usedL.add(row)
            usedR.add(col)
            ans.append((vocab[row], vocab[col], abs(scores[i])))
            if len(ans) == topn:
                break

        return ans


def viz(analogies):
    print(
        "\n".join(
            str(i).rjust(4) + a[0].rjust(29) + " | " + a[1].ljust(29) + (str(a[2]))[:4]
            for i, a in enumerate(analogies)
        )
    )


def text_plot_texts(xs, ys, texts, width=90, height=40, filename=None):
    PADDING = 10  # num chars on left and right in case texts spill over
    res = [[" " for i in range(width)] for j in range(height)]

    def rescale(nums):
        a = min(nums)
        b = max(nums)
        return [(x - a) / (b - a) for x in nums]

    print("x:", (min(xs), max(xs)), "y:", (min(ys), max(ys)))
    xs = rescale(xs)
    ys = rescale(ys)
    for (x, y, text) in zip(xs, ys, texts):
        i = int(x * (width - 1 - PADDING))
        j = int(y * (height - 1))
        row = res[j]
        z = list(
            row[i2] != " " for i2 in range(max(i - 1, 0), min(width, i + len(text) + 1))
        )
        if any(z):
            continue
        for k in range(len(text)):
            if i + k >= width:
                break
            row[i + k] = text[k]
    string = "\n".join("".join(r) for r in res)
    #     return string
    if filename:
        with open(filename, "w", encoding="utf8") as f:
            f.write(string)
        print("Wrote to", filename)
    else:
        print(string)


def doPCA(pairs, embedding, num_components=10):
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b)) / 2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    return pca


def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)
