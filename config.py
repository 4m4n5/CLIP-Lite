from typing import Any, List, Optional

from fvcore.common.config import CfgNode as CN
from utils.base import make_directory


class Config(object):
    r"""
    This class provides package-wide configuration management. It is a
    nested dict-like structure with nested keys accessible as attributes. It
    contains sensible default values, which can be modified by (first) a YAML
    file and (second) a list of attributes and values.
    An instantiated object is immutable: modifying any attribute is illegal.
    You must override required parameter values either through ``config_file``
    or ``override_list`` arguments. For adding more parameters at runtime
    (based on existing parameters), modify :meth:`add_derived_params`.
    Parameters
    ----------
    config_file: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default = [])
        A list of sequential attributes and values of parameters to override.
        This happens after overriding from YAML file.
    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::
        OPTIM:
          BATCH_SIZE: 512
          LR: 0.01
    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 1024])
    >>> _C.LR  # default: 0.001
    0.01
    >>> _C.OPTIM.BATCH_SIZE  # default: 256, file: 512
    1024
    """

    def __init__(
        self, config_file: Optional[str] = None, override_list: List[Any] = []
    ):
        _C = CN()

        # Random seed for NumPy and PyTorch, important for reproducibility.
        _C.RANDOM_SEED = 0
        # Train with Automatic Mixed Precision (native PyTorch).
        _C.AMP = True
        # Set CUDNN deterministic flag (torch.backends.cudnn.deterministic).
        # Setting this will ensure exact results on every run at the cost of
        # little slowdown. Good for debugging.
        _C.CUDNN_DETERMINISTIC = False
        # Set CUDNN benchmark flag (torch.backends.cudnn.benchmark). Enables
        # CUDNN to select fastest implementation for operations based on GPU.
        # May change results (in decimals) on different hardware, but faster
        # to train. Turn off while debugging.
        _C.CUDNN_BENCHMARK = True

        # ---------------------------------------------------------------------
        #   Data paths and parameters related to dataloading.
        # ---------------------------------------------------------------------
        _C.DATA = CN()
        # Use "train_sbert" for finetuning with sbert
        _C.DATA.NAME = "train_sbert"
        # Path to the dataset root, which structure as per README. Path is
        # assumed to be relative to project root.
        _C.DATA.ROOT = "/bigtemp/as3ek/p/vlinfo/datasets/serialized2/"
        # Size of the image (square) to crop from original input image.
        _C.DATA.IMAGE_CROP_SIZE = 224
        # Maximum length of input caption (number of tokens).
        # Longer captions will be truncated up to this length.
        _C.DATA.MAX_CAPTION_LENGTH = 30
        # COCO Captions has five captions per image. If ``True``, training will
        # use one random caption per image (data efficiency ablations).
        _C.DATA.USE_SINGLE_CAPTION = False
        # Percentage of dataset to use for training (data efficiency ablations).
        _C.DATA.USE_PERCENTAGE = 100.0
        # List of image transforms (pre-processing and data augmentation) to be
        # applied sequentially (always or randomly) during training and
        # validation. Refer ``vlinfo/facetories.py`` for all possible transforms.
        _C.DATA.IMAGE_TRANSFORM_TRAIN = [
            "random_resized_crop",
            "horizontal_flip",
            "color_jitter",
            "normalize",
        ]
        _C.DATA.IMAGE_TRANSFORM_VAL = [
            "smallest_resize",
            "center_crop",
            "normalize",
        ]
        # If model mode is json, which json files to use
        _C.DATA.JSON_FILES_TRAIN = [
            "/export/share/junnan-li/ALBEF/data/coco_karpathy_train.json",
            "/export/share/junnan-li/ALBEF/data/vg_caption.json",
            "/export/share/junnan-li/ALBEF/data/conceptual_caption_train.json",
            "/export/share/junnan-li/ALBEF/data/conceptual_caption_val.json",
            "/export/share/junnan-li/ALBEF/data/sbu_caption.json"
        ]
        _C.DATA.JSON_FILES_VAL = [
            "/export/share/junnan-li/ALBEF/data/coco_karpathy_val.json",
        ]

        # Type of negative sampling. Options: {'normal', 'clusters'}
        _C.DATA.NEGATIVE_SAMPLING = "normal"
        _C.DATA.NEGATIVE_SAMPLING_START_ITERATION = 250000
        _C.DATA.CLUSTER_PATH = ""
        _C.DATA.COCO_ROOT = "/bigtemp/as3ek/p/vlinfo/datasets/coco/"

        # ---------------------------------------------------------------------
        #   Model architecture: visual backbone and textual head.
        # ---------------------------------------------------------------------
        _C.MODEL = CN()
        _C.MODEL.NAME = "captions"
        _C.MODEL.VISUAL = CN()
        # Options: {'resnet18', 'resnet34', 'resnet50', 'vgg19'}
        _C.MODEL.VISUAL.NETWORK_NAME = "resnet50"
        # Increase the dimensions of the image encodings before projection
        _C.MODEL.VISUAL.FEATURE_SIZE = 2048
        _C.MODEL.VISUAL.FROZEN = False
        # Use a self supervised loss for the visual stream?
        _C.MODEL.VISUAL.SELF_SUPERVISED = False

        _C.MODEL.TEXTUAL = CN()
        # Name of the text encoding method to be used. Options: {'glove'}
        _C.MODEL.TEXTUAL.NAME = "train_sbert"
        # Initialize with pretrained text encoder or not
        _C.MODEL.TEXTUAL.PRETRAINED = False
        # Name of the sentence tranceformer from the huggingface model zoo
        _C.MODEL.TEXTUAL.NETWORK_NAME = "bert-base-uncased"
        # Path to the word dict if using glove vectors
        _C.MODEL.TEXTUAL.WORD_DICT_PATH = (
            "/u/as3ek/github/vlinfo/data/datasets/vocab/word_dict.json"
        )
        # Indicate usage of glove
        _C.MODEL.TEXTUAL.LOAD_GLOVE = False
        # Path to pretrained glove vectors to use
        _C.MODEL.TEXTUAL.GLOVE_PATH = (
            "/u/as3ek/github/vlinfo/data/datasets/glove/glove.42B.300d.txt"
        )
        # Fine tune word embeddings? If using glove otherwise this dont mean shit
        _C.MODEL.TEXTUAL.TRAIN_EMBEDDINGS = False
        # Increase the dimensions of the text vectors before projection?
        _C.MODEL.TEXTUAL.TRANSFORM = False
        # Increase the dimensions of the text vectors before projection to
        _C.MODEL.TEXTUAL.FEATURE_SIZE = 768
        # Use a self supervised loss for the textual stream?
        _C.MODEL.TEXTUAL.SELF_SUPERVISED = False
        _C.MODEL.TEXTUAL.NUM_HIDDEN_LAYERS = 12

        # Name of the loss to be used. Options: {'jsd'}
        _C.MODEL.LOSS = CN()
        _C.MODEL.LOSS.NAME = "jsd"
        # Critic function for cross modal and SSL losses
        # Options: {"dot", "concat", "dotcon", "condot"}
        _C.MODEL.LOSS.TYPE = "dot"
        _C.MODEL.LOSS.IMAGE_PRIOR = True
        _C.MODEL.LOSS.TEXT_PRIOR = True
        _C.MODEL.LOSS.PRIOR_WEIGHT = 0.1

        # ---------------------------------------------------------------------
        #   Optimization hyper-parameters, default values are for pretraining
        #   our best model on bicaptioning task (COCO Captions).
        # ---------------------------------------------------------------------
        _C.OPTIM = CN()

        # Name of optimizer to use. Supported values: {"sgd", "adamw"}.
        # AdamW uses default (beta1, beta2) values from PyTorch.
        _C.OPTIM.OPTIMIZER_NAME = "sgd"
        # Momentum co-efficient for SGD. Ignored for AdamW.
        _C.OPTIM.SGD_MOMENTUM = 0.9
        # Weight decay co-efficient for the optimizer.
        _C.OPTIM.WEIGHT_DECAY = 0.0001
        # Regex pattern of params for which there will be no weight decay.
        _C.OPTIM.NO_DECAY = ".*textual.(embedding|transformer).*(norm.*|bias)"
        # Max gradient norm for clipping to avoid exploding gradients.
        _C.OPTIM.CLIP_GRAD_NORM = 10.0

        # Wrap our optimizer with Lookahead (https://arxiv.org/abs/1907.08610).
        _C.OPTIM.LOOKAHEAD = CN()
        _C.OPTIM.LOOKAHEAD.USE = True
        _C.OPTIM.LOOKAHEAD.ALPHA = 0.5
        _C.OPTIM.LOOKAHEAD.STEPS = 5

        # We set different learning rates for CNN (visual backbone) and rest of
        # the model. CNN LR is typically much higher for training from scratch.
        # Both LRs undergo same warmup-decay schedules.

        # Total batch size (will be distributed evenly across GPUs).
        _C.OPTIM.BATCH_SIZE = 256
        # Max learning rate for CNN (visual backbone).
        _C.OPTIM.CNN_LR = 0.2
        # Max learning rate for rest of the model.
        _C.OPTIM.LR = 0.001
        # Max learning rate for rest of the model.
        _C.OPTIM.TRANS_LR = 0.001
        _C.OPTIM.MIN_LR_MULT = 0.0

        # Number of iterations to train for, batches are randomly sampled.
        _C.OPTIM.NUM_ITERATIONS = 500000

        # Number of steps at the start of training for linear LR warmup.
        _C.OPTIM.WARMUP_STEPS = 10000
        # Learning rate annealing schedule for decay after warmup.
        # Possible choices: {"none", "linear", "cosine", "multistep"}.
        _C.OPTIM.LR_DECAY_NAME = "cosine"
        # Steps to decay LR for "multistep" schedule.
        _C.OPTIM.LR_STEPS = []
        # Factor to multiply with LR for "multistep" schedule.
        _C.OPTIM.LR_GAMMA = 0.1

        _C.RUN_ID = ""

        # Override parameter values from YAML file first, then from override
        # list, then add derived params.
        self._C = _C
        if config_file is not None:
            self._C.merge_from_file(config_file)
        self._C.merge_from_list(override_list)

        self.add_derived_params()

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def add_derived_params(self):
        r"""Add parameters with values derived from existing parameters."""
        self._C.RUN_ID = (
            "/V?"
            + self._C.MODEL.VISUAL.NETWORK_NAME
            + "_T?"
            + self._C.MODEL.TEXTUAL.NAME
            + "_Ty?"
            + self._C.MODEL.LOSS.TYPE
            + "_Vs?"
            + str(self._C.MODEL.VISUAL.SELF_SUPERVISED)
            + "_Ts?"
            + str(self._C.MODEL.TEXTUAL.SELF_SUPERVISED)
            + "_N?"
            + self._C.DATA.NEGATIVE_SAMPLING
            + "_B?"
            + str(self._C.OPTIM.BATCH_SIZE)
            + "_O?"
            + self._C.OPTIM.OPTIMIZER_NAME
            + "_B?"
            + str(self._C.OPTIM.BATCH_SIZE)
            + "_D?"
            + self._C.OPTIM.LR_DECAY_NAME
            + "_Ni?"
            + str(self._C.OPTIM.NUM_ITERATIONS)
            + "_ID?"
            + self._C.RUN_ID
        )

    def dump(self, file_path: str):
        r"""Save config at the specified file path.
        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        return self._C.__str__()

    def __repr__(self):
        return self._C.__repr__()
