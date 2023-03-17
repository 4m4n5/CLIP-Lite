import argparse
import os
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from factories import PretrainingModelFactory, DownstreamDatasetFactory
from utils.checkpointing import CheckpointManager
from utils.common import common_parser, common_setup

import json
import os
import pickle
import torchvision.transforms as transforms
from torchvision.utils import save_image
import utils.we as we


parser = common_parser(
    description="Train SVMs for VOC2007 classification on a pretrained model."
)
group = parser.add_argument_group("Downstream config arguments.")
group.add_argument(
    "--down-config",
    metavar="FILE",
    help="Path to a downstream config file.",
    default="/u/as3ek/github/vlinfo/configs/downstream/bias_eda.yaml",
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
    choices=["random", "imagenet", "torchvision", "vlinfo"],
    default="vlinfo",
    help="""How to initialize weights:
        1. 'random' initializes all weights randomly
        2. 'imagenet' initializes backbone weights from torchvision model zoo
        3. 'vlinfo' load state dict from --checkpoint-path""",
)

parser.add_argument(
    "--loss-type",
    choices=["dot", "concat"],
    default="dot",
    help="""Which MI estimation approach has been used?""",
)

parser.add_argument(
    "--split",
    choices=["train", "val", "test"],
    default="train",
    help="""Which data split to use?""",
)

parser.add_argument(
    "--checkpoint-path",
    help="Path to load checkpoint and run downstream task evaluation.",
    default="/bigtemp/as3ek/p/vlinfo/saves/checkpoints/V?resnet50_T?train_sbert_Ty?dot_N?normal_O?sgd_D?cosine_ID?baseline/checkpoint_500000.pth",
)

parser.add_argument(
    "--gender-data-path",
    help="Path to load gender metdata for COCO dataset.",
    default="/bigtemp/as3ek/p/vlinfo/datasets/coco_gender/",
)

parser.add_argument(
    "--image-id-caption-map-path",
    help="Path to image to caption map",
    default="/bigtemp/as3ek/p/vlinfo/datasets/clusters/clusters_3/img_id_caption_map_train.pkl",
)

parser.add_argument(
    "--num-results",
    help="Number of results to return",
    default=10,
)


def save_pickle(data, path):
    with open(path, "wb") as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def count_parameters(model):
    mem_params = sum(
        [param.nelement() * param.element_size()
         for param in model.parameters()]
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_params, mem_params


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def softplus(x):
    x = -1.0 * np.array(x)
    t = 1 + np.exp(x)
    return np.log(t)


def debias_image(img_vec, E, definitional):
    gender_direction = we.doPCA(definitional, E).components_[0]
    img_vec = we.drop(img_vec, gender_direction)

    # img_vec /= np.linalg.norm(img_vec, axis=1)[:, np.newaxis]

    return img_vec


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

    # Create a (pretraining) config object and backup in serialization directory.
    _C = Config(_A.config, _A.config_override)

    train_dataset = DownstreamDatasetFactory.from_config(
        _DOWNC, split=_A.split)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_DOWNC.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )

    # Initialize from a checkpoint, but only keep the visual module.
    arch = PretrainingModelFactory.from_config(_C)

    num_params, mem_params = count_parameters(arch)
    print("Number of parameters in the model: " + str(num_params))
    print("Total memory usage of the model: " + str(mem_params))

    # Load weights according to the init method, do nothing for `random`, and
    # `imagenet` is already taken care of.
    if _A.weight_init == "vlinfo":
        _ = CheckpointManager(model=arch).load(_A.checkpoint_path)

    # Get the encoders
    img_encoder = arch.image_encoder.to(device).eval()
    text_encoder = arch.text_encoder.to(device).eval()

    # Get projectors if critic function is dot-product based
    if _A.loss_type == "dot":
        image_projector = arch.loss.global_d.img_block.to(device).eval()
        text_projector = arch.loss.global_d.text_block.to(device).eval()

    E = Embedding(
        text_encoder=text_encoder,
        tokenizer=train_dataset.tokenizer,
        text_projector=text_projector,
        loss_type="dot",
        device=device,
    )

    definitional = load_json(
        "/bigtemp/as3ek/p/vlinfo/datasets/coco_gender/meta/definitional_pairs.json"
    )

    # Dictionaries which store data in the following form:
    # {
    #     image_id: {
    #         "image_features": image_features,
    #         "text_features": text_features,
    #         "image": image
    #     }
    # }
    men_data = {}
    women_data = {}

    # Load if already exists
    if os.path.exists(os.path.join(_A.gender_data_path, "men_data_" + _A.split + ".pkl")):
        men_data = load_pickle(os.path.join(_A.gender_data_path, "men_data_" + _A.split + ".pkl"))
    if os.path.exists(os.path.join(_A.gender_data_path, "women_data_" + _A.split + ".pkl")):
        women_data = load_pickle(os.path.join(_A.gender_data_path, "women_data_" + _A.split + ".pkl"))

    else:
        with torch.no_grad():
            for batch in tqdm(train_dataloader):
                image_features = img_encoder(batch["image"].to(device))

                if _A.loss_type == "dot":
                    image_features = image_projector(image_features)

                for id in range(_DOWNC.OPTIM.BATCH_SIZE):
                    if batch["gender"][id][0].item() == 1:
                        men_data[batch["image_id"][id].item()] = {
                            "image": batch["image"][id].numpy(),
                            "image_features": image_features[id].cpu().numpy(),
                            "image_features_debiased": debias_image(
                                image_features[id].cpu().numpy(),
                                E,
                                definitional,
                            )
                        }
                    else:
                        women_data[batch["image_id"][id].item()] = {
                            "image": batch["image"][id].numpy(),
                            "image_features": image_features[id].cpu().numpy(),
                            "image_features_debiased": debias_image(
                                image_features[id].cpu().numpy(),
                                E,
                                definitional,
                            )
                        }

            save_pickle(
                men_data,
                os.path.join(_A.gender_data_path, "men_data_" + _A.split + ".pkl"),
            )
            save_pickle(
                women_data,
                os.path.join(_A.gender_data_path, "women_data_" + _A.split + ".pkl"),
            )

    while True:
        # Get a user input for a prompt sentence
        prompt_caption = input("Enter query text [type q to quit]: ")
        if prompt_caption == "q":
            break

        # Tokenize the caption from the prompt
        prompt_features = E.encode_text(prompt_caption)
        prompt_features = (
            torch.tensor(prompt_features, dtype=torch.float).unsqueeze(
                0).to(device)
        )
        # Init cosine similarity function

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        with torch.no_grad():
            # Dictionary to map image_id to its
            # cosine similarity with the caption
            men_sims = {}
            men_sims_debiased = {}
            for image_id, data in men_data.items():
                # Get image features from the data dictionary
                image_features = (
                    torch.tensor(data["image_features"], dtype=torch.float)
                    .unsqueeze(0)
                    .to(device)
                )

                image_features_debiased = (
                    torch.tensor(
                        data["image_features_debiased"], dtype=torch.float)
                    .unsqueeze(0)
                    .to(device)
                )

                image_features, image_features_debiased, prompt_features = map(
                    lambda t: F.normalize(t, p=2, dim=-1),
                    (image_features, image_features_debiased, prompt_features),
                )

                men_sims[image_id] = cos(
                    image_features, prompt_features).item()
                men_sims_debiased[image_id] = cos(
                    image_features_debiased, prompt_features
                ).item()

            women_sims = {}
            women_sims_debiased = {}
            for image_id, data in women_data.items():
                image_features = (
                    torch.tensor(data["image_features"], dtype=torch.float)
                    .unsqueeze(0)
                    .to(device)
                )

                image_features_debiased = (
                    torch.tensor(
                        data["image_features_debiased"], dtype=torch.float)
                    .unsqueeze(0)
                    .to(device)
                )

                image_features, image_features_debiased, prompt_features = map(
                    lambda t: F.normalize(t, p=2, dim=-1),
                    (image_features, image_features_debiased, prompt_features),
                )

                women_sims[image_id] = cos(
                    image_features, prompt_features).item()
                women_sims_debiased[image_id] = cos(
                    image_features_debiased, prompt_features
                ).item()

        # Sort dictionaries descending by similarity. most similar -> least similar
        men_sims_sorted = dict(
            sorted(men_sims.items(), key=lambda item: -item[1]))
        women_sims_sorted = dict(
            sorted(women_sims.items(), key=lambda item: -item[1]))
        men_sims_debiased_sorted = dict(
            sorted(men_sims_debiased.items(), key=lambda item: -item[1])
        )
        women_sims_debiased_sorted = dict(
            sorted(women_sims_debiased.items(), key=lambda item: -item[1])
        )

        if _A.split == "val":
            _A.image_id_caption_map_path = _A.image_id_caption_map_path.replace(
                "train", "val"
            )
        id2caption = load_pickle(_A.image_id_caption_map_path)
        n_inv = transforms.Normalize(
            [-0.485 / 0.229, -0.546 / 0.224, -0.406 / 0.225],
            [1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

        num_results = _A.num_results
        # Print the results on the terminal
        print("Mean Alignment | Men | Biased: " + str(np.mean(list(men_sims_sorted.values())[:num_results])))
        print("Mean Alignment | Men | Debiased: " + str(np.mean(list(women_sims_sorted.values())[:num_results])))
        print("Mean Alignment | Women | Biased: " + str(np.mean(list(men_sims_debiased_sorted.values())[:num_results])))
        print("Mean Alignment | Women | Debiased: " + str(np.mean(list(women_sims_debiased_sorted.values())[:num_results])))

        biased_men_sims = []
        biased_women_sims = []

        debiased_men_sims = []
        debiased_women_sims = []

        for mode in ["normal", "debiased"]:
            if mode == "debiased":
                men_sims_sorted = men_sims_debiased_sorted
                women_sims_sorted = women_sims_debiased_sorted

            print("++++++++++ Mode: " + mode + " ++++++++++")

            for i in range(int(num_results)):
                men_id = list(men_sims_sorted.keys())[i]
                women_id = list(women_sims_sorted.keys())[i]

                men_sim = list(men_sims_sorted.values())[i]
                women_sim = list(women_sims_sorted.values())[i]

                biased_men_sims.append(men_sim)
                biased_women_sims.append(women_sim)
                if mode == "debiased":
                    debiased_men_sims.append(men_sim)
                    debiased_women_sims.append(women_sim)

                print("============== Rank " + str(i + 1) + " ==============")
                print("Male Set: " + str(men_sim))
                print(str(id2caption[men_id]) + "\n")

                print("Female Set: " + str(women_sim))
                print(str(id2caption[women_id]) + "\n")

                men_image = n_inv(
                    torch.tensor(men_data[men_id]["image"], dtype=torch.float)
                )
                women_image = n_inv(
                    torch.tensor(women_data[women_id]
                                 ["image"], dtype=torch.float)
                )

                men_path = os.path.join(
                    _A.gender_data_path, "outputs", mode, "men_" +
                    str(i) + ".jpeg"
                )
                women_path = os.path.join(
                    _A.gender_data_path, "outputs", mode, "women_" +
                    str(i) + ".jpeg"
                )
                save_image(men_image.unsqueeze(0), men_path)
                save_image(women_image.unsqueeze(0), women_path)

        print("Mens similarity biased: " + str(biased_men_sims))
        print("Mens similarity debiased: " + str(debiased_men_sims))
        print("Womens similarity biased: " + str(biased_women_sims))
        print("Womens similarity debiased: " + str(debiased_women_sims))


if __name__ == "__main__":
    _A = parser.parse_args()
    _A.num_gpus_per_machine = 1

    # No distributed training here, just a single process.
    main(_A)
