import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, vgg19, resnet50, resnet101, resnet152
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig
import torch
import torchvision
import transformers
from typing import Any, Dict, Union


class ImageEncoder(nn.Module):
    r"""
    An image encoder from `Torchvision model zoo
    <https://pytorch.org/docs/stable/torchvision/models.html>`_. Any model can
    be specified using corresponding method name from the model zoo.
    Parameters
    ----------
    img_enc_net: str, optional (default = "resnet50")
        Name of the model from Torchvision model zoo.
    pretrained: bool, optional (default = False)
        Whether to load ImageNet pretrained weights from Torchvision.
    frozen: float, optional (default = False)
        Whether to keep all weights frozen during training.
    """

    def __init__(
        self,
        img_enc_net: str = "resnet50",
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super(ImageEncoder, self).__init__()

        self.img_encoder = getattr(torchvision.models, img_enc_net)(
            pretrained, zero_init_residual=False
        )

        # Do nothing at the last layer
        self.img_encoder.fc = nn.Identity()

        # Freeze all weights if specified.
        if frozen:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
            self.img_encoder.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""
        Compute visual features for a batch of input images.
        Parameters
        ----------
        image: torch.Tensor
            Batch of input images. A tensor of shape
            ``(batch_size, 3, height, width)``.
        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, fc_feature_size)``, for
            example it will be ``(batch_size, 2048)`` for ResNet-50.
        """
        x = self.img_encoder(image)

        return x.view(x.size(0), x.size(1))

    def detectron2_backbone_state_dict(self) -> Dict[str, Any]:
        r"""
        Return state dict of visual backbone which can be loaded with
        `Detectron2 <https://github.com/facebookresearch/detectron2>`_.
        This is useful for downstream tasks based on Detectron2 (such as
        object detection and instance segmentation). This method renames
        certain parameters from Torchvision-style to Detectron2-style.
        Returns
        -------
        Dict[str, Any]
            A dict with three keys: ``{"model", "author", "matching_heuristics"}``.
            These are necessary keys for loading this state dict properly with
            Detectron2.
        """
        # Detectron2 backbones have slightly different module names, this mapping
        # lists substrings of module names required to be renamed for loading a
        # torchvision model into Detectron2.
        DETECTRON2_RENAME_MAPPING: Dict[str, str] = {
            "layer1": "res2",
            "layer2": "res3",
            "layer3": "res4",
            "layer4": "res5",
            "bn1": "conv1.norm",
            "bn2": "conv2.norm",
            "bn3": "conv3.norm",
            "downsample.0": "shortcut",
            "downsample.1": "shortcut.norm",
        }
        # Populate this dict by renaming module names.
        d2_backbone_dict: Dict[str, torch.Tensor] = {}

        for name, param in self.img_encoder.state_dict().items():
            for old, new in DETECTRON2_RENAME_MAPPING.items():
                name = name.replace(old, new)

            # First conv and bn module parameters are prefixed with "stem.".
            if not name.startswith("res"):
                name = f"stem.{name}"

            d2_backbone_dict[name] = param

        return {
            "model": d2_backbone_dict,
            "__author__": "VLInfo",
            "matching_heuristics": True,
        }


class TextEncoder(nn.Module):
    r"""
    A sentence transformers model from
    <https://huggingface.co/sentence-transformers>`_. Any model can
    be specified using corresponding name from the model zoo.
    """

    def __init__(
        self,
        word_dict,
        mode="train_sbert",
        transform_embedding=False,
        txt_enc_dim=512,
        glove_path="/u/as3ek/github/vlinfo/data/datasets/glove/glove.42B.300d.txt",
        train_enc=False,
        load_glove=True,
        model_name="bert-base-uncased",
        pretrained=False,
        num_hidden_layers=12,
    ):
        super(TextEncoder, self).__init__()

        self.transform_embedding = transform_embedding
        self.txt_enc_dim = txt_enc_dim
        self.mode = mode
        self.model_name = model_name
        self.num_hidden_layers = num_hidden_layers

        if mode == "glove":
            if load_glove:
                (
                    self.txt_enc_layer,
                    self.vocab_size,
                    self.glove_dim,
                ) = self.get_text_encoding_layer(glove_path, word_dict, train_enc)
                in_dim = self.glove_dim

            else:
                self.vocab_size = len(word_dict)
                self.txt_enc_layer = nn.Embedding(self.vocab_size, 300)
                in_dim = 300

        elif mode == "sbert":
            in_dim = 768

        elif mode == "train_sbert":
            if pretrained:
                print("Using pre-trained bert model")
                self.strans = BertModel.from_pretrained(model_name)
            else:
                if "bert" in model_name:
                    print("Using bert model with layers: " +
                          str(self.num_hidden_layers))
                    configuration = BertConfig(
                        num_hidden_layers=self.num_hidden_layers)
                    self.strans = BertModel(configuration)
                else:
                    print("Using mpnet model" +
                          str(self.num_hidden_layers))
                    self.strans = AutoModel.from_config(
                        transformers.MPNetConfig())
            in_dim = 768

        elif mode == "finetune_sbert":
            self.strans = AutoModel.from_pretrained(model_name)
            in_dim = 768

        if transform_embedding:
            self.fc1 = nn.Linear(in_dim, self.txt_enc_dim)
            self.fc2 = nn.Linear(self.txt_enc_dim, self.txt_enc_dim)
            self.relu = nn.ReLU()

    def forward(self, x):
        if self.mode == "glove":
            x = self.txt_enc_layer(x)
            x = torch.mean(x, dim=1)

        if self.mode == "train_sbert" or self.mode == "finetune_sbert":
            model_output = self.strans(**x)

            if "bert" in self.model_name:
                x = model_output.pooler_output
            else:
                x = self.mean_pooling(model_output, x["attention_mask"])

        if self.transform_embedding:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

        return x

    def train_enc(self):
        for param in self.strans.parameters():
            param.requires_grad = True

    def dont_train_enc(self):
        for param in self.strans.parameters():
            param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def get_weights_matrix(self, glove_path, word_dict):
        # Load glove model using the given path
        glove, glove_dim = self.load_glove_model(glove_path)
        # Get vocab size
        matrix_len = len(word_dict)
        # Initialize empty matrix
        weights_matrix = np.zeros((matrix_len, glove_dim))
        words_found = 0

        # If word is present in glove, add the embedding else random
        for word, idx in word_dict.items():
            try:
                weights_matrix[word_dict[word]] = glove[word]
                words_found += 1

            except KeyError:
                weights_matrix[word_dict[word]] = np.random.normal(
                    scale=0.6, size=(glove_dim,)
                )

        assert len(word_dict) == len(weights_matrix)

        return weights_matrix

    def get_text_encoding_layer(self, glove_path, word_dict, train_enc):
        # Get weights matrix given glove vectors and word dict
        weights_matrix = self.get_weights_matrix(glove_path, word_dict)
        weights_matrix = torch.from_numpy(weights_matrix)

        vocab_size, txt_enc_dim = weights_matrix.size()
        # Initialize embedding layer using the glove weights matrix
        txt_enc_layer = nn.Embedding(vocab_size, txt_enc_dim)
        txt_enc_layer.load_state_dict({"weight": weights_matrix})

        # If embeddings are not to be trained
        if train_enc == False:
            txt_enc_layer.weight.requires_grad = False

        return txt_enc_layer, vocab_size, txt_enc_dim

    def load_glove_model(self, glove_path):
        print("Loading Glove Model")
        f = open(glove_path, "r")
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value)
                                     for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        print(len(gloveModel), " words loaded!")

        return gloveModel, len(wordEmbedding)
