r"""
This module is a collection of *factories* for creating objects of datasets,
models, optimizers and other useful components. Refer description of
specific factories for more details.
"""
from functools import partial
import re
from typing import Any, Callable, Dict, Iterable, List

import albumentations as alb
from torch import nn, optim

from config import Config
from model import VLInfoModel
from data.dataloader import (
    CocoCaptionsDataset,
    JsonDataset,
    VOC07ClassificationDataset,
    ImageNetDataset,
    INaturalist2018Dataset,
    ReEvalDataset,
    CocoCaptionsClusteredDataset,
    CocoObjectGender,
    RandomDataset,
    re_eval_dataset
)
from data import transforms as T
from data.tokenizers import GloveTokenizer
from encoder import ImageEncoder, TextEncoder
from optim import Lookahead, lr_scheduler
from loss import JSDInfoMaxLoss
import json
import os


class Factory(object):
    r"""
    Base class for all factories. All factories must inherit this base class
    and follow these guidelines for a consistent behavior:
    * Factory objects cannot be instantiated, doing ``factory = SomeFactory()``
      is illegal. Child classes should not implement ``__init__`` methods.
    * All factories must have an attribute named ``PRODUCTS`` of type
      ``Dict[str, Callable]``, which associates each class with a unique string
      name which can be used to create it.
    * All factories must implement one classmethod, :meth:`from_config` which
      contains logic for creating an object directly by taking name and other
      arguments directly from :class:`~.config.Config`. They can use
      :meth:`create` already implemented in this base class.
    * :meth:`from_config` should not use too many extra arguments than the
      config itself, unless necessary (such as model parameters for optimizer).
    """

    PRODUCTS: Dict[str, Callable] = {}

    def __init__(self):
        raise ValueError(
            f"""Cannot instantiate {self.__class__.__name__} object, use
            `create` classmethod to create a product from this factory.
            """
        )

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        r"""Create an object by its name, args and kwargs."""
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls.PRODUCTS[name](*args, **kwargs)

    @classmethod
    def from_config(cls, config: Config) -> Any:
        r"""Create an object directly from config."""
        raise NotImplementedError


class TokenizerFactory(Factory):
    r"""
    Factory to create text tokenizers. This codebase ony supports one tokenizer
    for now, but having a dedicated factory makes it easy to add more if needed.
    Possible choices: ``{"GloveTokenizer"}``.
    """

    PRODUCTS: Dict[str, Callable] = {"GloveTokenizer": GloveTokenizer}

    @classmethod
    def from_config(cls, config: Config) -> GloveTokenizer:
        r"""
        Create a tokenizer directly from config.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        """

        _C = config

        tokenizer = cls.create(
            "GloveTokenizer",
        )
        return tokenizer


class ImageTransformsFactory(Factory):
    r"""
    Factory to create image transformations for common preprocessing and data
    augmentations. These are a mix of default transformations from
    `albumentations <https://albumentations.readthedocs.io/en/latest/>`_ and
    some extended ones defined in :mod:`.data.transforms`.
    It uses sensible default values, however they can be provided with the name
    in dict syntax. Example: ``random_resized_crop::{'scale': (0.08, 1.0)}``
    .. note::
        This factory does not implement :meth:`from_config` method. It is only
        used by :class:`PretrainingDatasetFactory` and
        :class:`DownstreamDatasetFactory`.
    Possible choices: ``{"center_crop", "horizontal_flip", "random_resized_crop",
    "normalize", "global_resize", "color_jitter", "smallest_resize"}``.
    """

    # fmt: off
    PRODUCTS: Dict[str, Callable] = {
        # Input resize transforms: whenever selected, these are always applied.
        # These transforms require one position argument: image dimension.
        "random_resized_crop": partial(
            T.RandomResizedSquareCrop, scale=(0.2, 1.0), ratio=(0.75, 1.333), p=1.0
        ),
        "center_crop": partial(T.CenterSquareCrop, p=1.0),
        "smallest_resize": partial(alb.SmallestMaxSize, p=1.0),
        "global_resize": partial(T.SquareResize, p=1.0),

        # Keep hue limits small in color jitter because it changes color drastically
        # and captions often mention colors. Apply with higher probability.
        "color_jitter": partial(
            alb.ColorJitter, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        "color_jitter8": partial(
            alb.ColorJitter, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1, p=0.8
        ),
        "random_gray": partial(alb.ToGray, p=0.2),
        "horizontal_flip": partial(T.HorizontalFlip, p=0.5),
        "blur": partial(alb.GaussianBlur, p=0.5),

        # Color normalization: whenever selected, always applied. This accepts images
        # in [0, 255], requires mean and std in [0, 1] and normalizes to `N(0, 1)`.
        "normalize": partial(
            alb.Normalize, mean=T.IMAGENET_COLOR_MEAN, std=T.IMAGENET_COLOR_STD, p=1.0
        ),
    }
    # fmt: on

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        r"""Create an object by its name, args and kwargs."""

        if "::" in name:
            name, __kwargs = name.split("::")
            _kwargs = eval(__kwargs)
        else:
            _kwargs = {}

        _kwargs.update(kwargs)
        return super().create(name, *args, **_kwargs)

    @classmethod
    def from_config(cls, config: Config):
        r"""Augmentations cannot be created from config, only :meth:`create`."""
        raise NotImplementedError


class PretrainingDatasetFactory(Factory):
    r"""
    Factory to create :class:`~torch.utils.data.Dataset` s for pretraining
    models. Datasets provide image-caption pairs from COCO Captions dataset
    (serialized to an LMDB file).
    Possible choices: ``{"captions"}``.
    """

    PRODUCTS: Dict[str, Callable] = {
        "captions": CocoCaptionsDataset,
        "random": RandomDataset,
        "json": JsonDataset,
    }

    @classmethod
    def from_config(cls, config: Config, split: str = "train"):
        r"""
        Create a dataset directly from config. Names in this factory match with
        names in :class:`PretrainingModelFactory` because both use same config
        parameter ``MODEL.NAME`` to create objects.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        split: str, optional (default = "train")
            Which split to load for the dataset. One of ``{"train", "val"}``.
        """

        _C = config
        # Every dataset needs these two args.
        kwargs = {
            "data_root": _C.DATA.ROOT,
            "split": split,
            "mode": _C.DATA.NAME,
            "tokenizer_name": _C.MODEL.TEXTUAL.NETWORK_NAME,
            "use_single_caption": _C.DATA.USE_SINGLE_CAPTION,
            "visual_self_supervised": _C.MODEL.VISUAL.SELF_SUPERVISED,
            "textual_self_supervised": _C.MODEL.TEXTUAL.SELF_SUPERVISED,
            "percentage": _C.DATA.USE_PERCENTAGE,
        }

        # Create a list of image transformations based on transform names.
        image_transform_list: List[Callable] = []

        for name in getattr(_C.DATA, f"IMAGE_TRANSFORM_{split.upper()}"):
            # Pass dimensions if cropping / resizing, else rely on the defaults
            # as per `ImageTransformsFactory`.
            if "resize" in name or "crop" in name:
                image_transform_list.append(
                    ImageTransformsFactory.create(
                        name, _C.DATA.IMAGE_CROP_SIZE)
                )
            else:
                image_transform_list.append(
                    ImageTransformsFactory.create(name))

        kwargs["image_transform"] = alb.Compose(image_transform_list)

        if _C.MODEL.NAME == "json":
            if split == "train":
                json_files = _C.DATA.JSON_FILES_TRAIN
            elif split == "val":
                json_files = _C.DATA.JSON_FILES_VAL
                kwargs["percentage"] = 50.0

            return cls.create(_C.MODEL.NAME, json_files, **kwargs)

        else:
            # Dataset names match with model names (and ofcourse pretext names).
            return cls.create(_C.MODEL.NAME, **kwargs)


class NegativeSamplingDatasetFactory(Factory):
    r"""
    Factory to create :class:`~torch.utils.data.Dataset` s for negative sampling
    models. Datasets provide image-caption-neg_image-neg_caption pairs from COCO Captions dataset
    (serialized to an LMDB file).
    Possible choices: ``{"captions"}``.
    """

    PRODUCTS: Dict[str, Callable] = {
        "clusters": CocoCaptionsClusteredDataset,
    }

    @classmethod
    def from_config(cls, config: Config, split: str = "train"):
        r"""
        Create a dataset directly from config. Names in this factory match with
        names in :class:`PretrainingModelFactory` because both use same config
        parameter ``MODEL.NAME`` to create objects.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        split: str, optional (default = "train")
            Which split to load for the dataset. One of ``{"train", "val"}``.
        """

        _C = config
        # Every dataset needs these two args.
        kwargs = {
            "data_root": _C.DATA.ROOT,
            "split": split,
            "mode": _C.DATA.NAME,
            "tokenizer_name": _C.MODEL.TEXTUAL.NETWORK_NAME,
            "negative_sampling": _C.DATA.NEGATIVE_SAMPLING,
            "total_iters": _C.OPTIM.NUM_ITERATIONS,
            "negative_sampling_start_iter": _C.DATA.NEGATIVE_SAMPLING_START_ITERATION,
            "cluster_path": _C.DATA.CLUSTER_PATH,
            "use_single_caption": _C.DATA.USE_SINGLE_CAPTION,
            "coco_root": _C.DATA.COCO_ROOT,
        }

        # Create a list of image transformations based on transform names.
        image_transform_list: List[Callable] = []

        for name in getattr(_C.DATA, f"IMAGE_TRANSFORM_{split.upper()}"):
            # Pass dimensions if cropping / resizing, else rely on the defaults
            # as per `ImageTransformsFactory`.
            if "resize" in name or "crop" in name:
                image_transform_list.append(
                    ImageTransformsFactory.create(
                        name, _C.DATA.IMAGE_CROP_SIZE)
                )
            else:
                image_transform_list.append(
                    ImageTransformsFactory.create(name))

        kwargs["image_transform"] = alb.Compose(image_transform_list)

        # Dataset names match with model names (and ofcourse pretext names).
        return cls.create(_C.DATA.NEGATIVE_SAMPLING, **kwargs)


class VisualBackboneFactory(Factory):
    r"""
    Factory to create :mod:`~encoder.ImageEncoder`.
    """

    PRODUCTS: Dict[str, Callable] = {
        "captions": ImageEncoder,
        "random": ImageEncoder,
        "json": ImageEncoder,
    }

    @classmethod
    def from_config(cls, config: Config) -> ImageEncoder:
        r"""
        Create a visual backbone directly from config.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        """

        _C = config
        kwargs = {"img_enc_net": _C.MODEL.VISUAL.NETWORK_NAME}

        return cls.create(_C.MODEL.NAME, **kwargs)


class TextualHeadFactory(Factory):
    r"""
    Factory to create :mod:`~encoder.TextEncoder`.
    """

    PRODUCTS: Dict[str, Callable] = {
        "glove": TextEncoder,
        "sbert": TextEncoder,
        "train_sbert": TextEncoder,
    }

    @classmethod
    def from_config(cls, config: Config) -> nn.Module:
        r"""
        Create a textual head directly from config.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        """

        _C = config
        name = _C.MODEL.TEXTUAL.NAME
        kwargs = {
            "word_dict": {},
            "mode": name,
            "transform_embedding": _C.MODEL.TEXTUAL.TRANSFORM,
            "txt_enc_dim": _C.MODEL.TEXTUAL.FEATURE_SIZE,
            "load_glove": _C.MODEL.TEXTUAL.LOAD_GLOVE,
            "glove_path": _C.MODEL.TEXTUAL.GLOVE_PATH,
            "train_enc": _C.MODEL.TEXTUAL.TRAIN_EMBEDDINGS,
            "pretrained": _C.MODEL.TEXTUAL.PRETRAINED,
            "model_name": _C.MODEL.TEXTUAL.NETWORK_NAME,
            "num_hidden_layers": _C.MODEL.TEXTUAL.NUM_HIDDEN_LAYERS,
        }

        return cls.create(name, **kwargs)


class LossFactory(Factory):
    r"""
    Factory to create :mod:`~loss.JSDInfoMaxLoss`.
    """

    PRODUCTS: Dict[str, Callable] = {
        "jsd": JSDInfoMaxLoss,
    }

    @classmethod
    def from_config(cls, config: Config) -> JSDInfoMaxLoss:
        r"""
        Create a loss backbone directly from config.
        Parameters
        ----------
        config: config.Config
            Config object with all the parameters.
        """

        _C = config
        kwargs = {
            "image_dim": _C.MODEL.VISUAL.FEATURE_SIZE,
            "text_dim": _C.MODEL.TEXTUAL.FEATURE_SIZE,
            "type": _C.MODEL.LOSS.TYPE,
            "image_prior": _C.MODEL.LOSS.IMAGE_PRIOR,
            "text_prior": _C.MODEL.LOSS.TEXT_PRIOR,
            "prior_weight": _C.MODEL.LOSS.PRIOR_WEIGHT,
            "visual_self_supervised": _C.MODEL.VISUAL.SELF_SUPERVISED,
            "textual_self_supervised": _C.MODEL.TEXTUAL.SELF_SUPERVISED,
        }

        return cls.create(_C.MODEL.LOSS.NAME, **kwargs)


class PretrainingModelFactory(Factory):
    r"""
    Factory to create :mod:`~models` for different pretraining tasks.
    Possible choices: ``{"captions"}``.
    """

    PRODUCTS: Dict[str, Callable] = {
        "captions": VLInfoModel,
        "random": VLInfoModel,
        "json": VLInfoModel,
    }

    @classmethod
    def from_config(cls, config: Config) -> nn.Module:
        r"""
        Create a model directly from config.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        """

        _C = config
        mode = _C.MODEL.TEXTUAL.NAME

        # Build visual and textual streams based on config.
        visual = VisualBackboneFactory.from_config(_C)
        textual = TextualHeadFactory.from_config(_C)
        loss = LossFactory.from_config(_C)
        is_amp = _C.AMP

        return cls.create(_C.MODEL.NAME, textual, visual, loss, mode, is_amp)


class OptimizerFactory(Factory):
    r"""Factory to create optimizers. Possible choices: ``{"sgd", "adamw"}``."""

    PRODUCTS: Dict[str, Callable] = {"sgd": optim.SGD, "adamw": optim.AdamW}

    @classmethod
    def from_config(
        cls, config: Config, named_parameters: Iterable[Any]
    ) -> optim.Optimizer:
        r"""
        Create an optimizer directly from config.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        named_parameters: Iterable
            Named parameters of model (retrieved by ``model.named_parameters()``)
            for the optimizer. We use named parameters to set different LR and
            turn off weight decay for certain parameters based on their names.
        """

        _C = config

        # Set different learning rate for CNN and rest of the model during
        # pretraining. This doesn't matter for downstream evaluation because
        # there are no modules with "cnn" in their name.
        # Also turn off weight decay for layer norm and bias in textual stream.
        param_groups = []
        for name, param in named_parameters:
            wd = 0.0 if re.match(
                _C.OPTIM.NO_DECAY, name) else _C.OPTIM.WEIGHT_DECAY
            if "image_encoder" in name:
                lr = _C.OPTIM.CNN_LR
            elif "text_encoder" in name:
                lr = _C.OPTIM.TRANS_LR
            else:
                lr = _C.OPTIM.LR
            param_groups.append(
                {"params": [param], "lr": lr, "weight_decay": wd})

        if _C.OPTIM.OPTIMIZER_NAME == "sgd":
            kwargs = {"momentum": _C.OPTIM.SGD_MOMENTUM}
        else:
            kwargs = {}

        optimizer = cls.create(_C.OPTIM.OPTIMIZER_NAME, param_groups, **kwargs)
        if _C.OPTIM.LOOKAHEAD.USE:
            optimizer = Lookahead(
                optimizer, k=_C.OPTIM.LOOKAHEAD.STEPS, alpha=_C.OPTIM.LOOKAHEAD.ALPHA
            )
        return optimizer


class LRSchedulerFactory(Factory):
    r"""
    Factory to create LR schedulers. All schedulers have a built-in LR warmup
    schedule before actual LR scheduling (decay) starts.
    Possible choices: ``{"none", "multistep", "linear", "cosine"}``.
    """

    PRODUCTS: Dict[str, Callable] = {
        "none": lr_scheduler.LinearWarmupNoDecayLR,
        "multistep": lr_scheduler.LinearWarmupMultiStepLR,
        "linear": lr_scheduler.LinearWarmupLinearDecayLR,
        "cosine": lr_scheduler.LinearWarmupCosineAnnealingLR,
    }

    @classmethod
    def from_config(
        cls, config: Config, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LambdaLR:
        r"""
        Create an LR scheduler directly from config.
        Parameters
        ----------
        config: .config.Config
            Config object with all the parameters.
        optimizer: torch.optim.Optimizer
            Optimizer on which LR scheduling would be performed.
        """

        _C = config
        kwargs = {
            "total_steps": _C.OPTIM.NUM_ITERATIONS,
            "warmup_steps": _C.OPTIM.WARMUP_STEPS,
        }
        # Multistep LR requires multiplicative factor and milestones.
        if _C.OPTIM.LR_DECAY_NAME == "multistep":
            kwargs.update(gamma=_C.OPTIM.LR_GAMMA,
                          milestones=_C.OPTIM.LR_STEPS)

        if _C.OPTIM.LR_DECAY_NAME == "cosine":
            kwargs.update(min_mult=_C.OPTIM.MIN_LR_MULT)

        return cls.create(_C.OPTIM.LR_DECAY_NAME, optimizer, **kwargs)


class DownstreamDatasetFactory(Factory):
    r"""
    Factory to create :class:`~torch.utils.data.Dataset` s for evaluating
    VirTex models on downstream tasks.
    Possible choices: ``{"datasets/VOC2007", "datasets/imagenet"}``.
    """

    PRODUCTS: Dict[str, Callable] = {
        "datasets/VOC2007": VOC07ClassificationDataset,
        "datasets/imagenet": ImageNetDataset,
        "datasets/inaturalist": INaturalist2018Dataset,
        "/bigtemp/as3ek/p/vlinfo/datasets/coco": ReEvalDataset,
        "/bigtemp/as3ek/p/vlinfo/datasets/imagenet2012": ImageNetDataset,
        "/bigtemp/as3ek/p/vlinfo/datasets/inaturalist": INaturalist2018Dataset,
        "/export/share/datasets/vision/imagenet": ImageNetDataset,
        "/export/share/datasets/vision/coco/": ReEvalDataset,
        "/bigtemp/as3ek/p/vlinfo/datasets/coco/": CocoObjectGender,
        "/bigtemp/as3ek/p/vlinfo/datasets/VOC2007/": VOC07ClassificationDataset,
        "/export/share/datasets/vision/imagenet/": ImageNetDataset,
        "/bigtemp/as3ek/p/vlinfo/datasets/flickr30k": re_eval_dataset,
    }

    @classmethod
    def from_config(cls, config: Config, split: str = "train"):
        r"""
        Create a dataset directly from config. Names in this factory are paths
        of dataset directories (relative to the project directory), because
        config parameter ``DATA.ROOT`` is used to create objects.
        Parameters
        ----------
        config: virtex.config.Config
            Config object with all the parameters.
        split: str, optional (default = "train")
            Which split to load for the dataset. One of ``{"trainval", "test"}``
            for VOC2007, or one of ``{"train", "val"}`` for ImageNet.
        """

        _C = config
        # Every dataset needs these two args.
        kwargs = {"data_root": _C.DATA.ROOT, "split": split}

        # For VOC2007, `IMAGE_TRANSFORM_TRAIN` is used for "trainval" split and
        # `IMAGE_TRANSFORM_VAL` is used fo "test" split.
        image_transform_names: List[str] = list(
            _C.DATA.IMAGE_TRANSFORM_TRAIN
            if "train" in split
            else _C.DATA.IMAGE_TRANSFORM_VAL
        )
        # Create a list of image transformations based on names.
        image_transform_list: List[Callable] = []

        for name in image_transform_names:
            # Pass dimensions for resize/crop, else rely on the defaults.
            if name.split("::")[0] in {
                "random_resized_crop",
                "center_crop",
                "global_resize",
            }:
                transform = ImageTransformsFactory.create(
                    name, _C.DATA.IMAGE_CROP_SIZE)
            elif name.split("::")[0] in {"smallest_resize"}:
                transform = ImageTransformsFactory.create(
                    name, _C.DATA.IMAGE_CROP_SIZE)
            else:
                transform = ImageTransformsFactory.create(name)

            image_transform_list.append(transform)

        kwargs["image_transform"] = alb.Compose(image_transform_list)

        # Uncomment for retrieval
        # if "datasets/coco" in _C.DATA.ROOT:
        #     kwargs["ann_file"] = os.path.join(
        #         _C.DATA.ROOT, "annotations/captions_val2017.json"
        #     )

        if "flickr" in _C.DATA.ROOT:
            # kwargs["data_root"] = os.path.join(_C.DATA.ROOT, "flickr30k-images")
            kwargs["ann_file"] = os.path.join(
                _C.DATA.ROOT, "data/flickr30k_test.json"
            )

        return cls.create(_C.DATA.ROOT, **kwargs)
