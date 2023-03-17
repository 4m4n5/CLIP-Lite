import copy
import functools
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from data.tokenizers import GloveTokenizer
from encoder import ImageEncoder, TextEncoder
from loss import JSDInfoMaxLoss
from torch.cuda import amp


class VLInfoModel(nn.Module):
    def __init__(
        self,
        text_encoder: TextEncoder,
        image_encoder: ImageEncoder,
        loss: JSDInfoMaxLoss,
        mode: str = "sbert",
        is_amp: bool = True,
    ):

        super(VLInfoModel, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.loss = loss
        self.mode = mode
        self.is_amp = is_amp

    def forward(self, batch):
        with amp.autocast(enabled=self.is_amp):
            # Encode the image
            image = batch["image"]
            image_features = self.image_encoder(image)

            # Placeholders that will be used later
            neg_text_features = None
            neg_image_features = None
            aug_image_features = None
            aug_text_features = None

            if self.mode == "glove":
                caption = batch["caption_tokens"]
                text_features = self.text_encoder(caption)

            elif self.mode == "sbert":
                caption = batch["caption_encodings"]
                text_features = self.text_encoder(caption)

            elif self.mode == "train_sbert":
                # Encode the text
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                text_features = self.text_encoder(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                )

                # If the dataloader has sent a negative sample
                if "neg_input_ids" in batch:
                    # Encode the negative image
                    neg_image = batch["neg_image"]
                    neg_image_features = self.image_encoder(neg_image)

                    # Encode the negative text
                    neg_input_ids = batch["neg_input_ids"]
                    neg_attention_mask = batch["neg_attention_mask"]
                    neg_text_features = self.text_encoder(
                        {
                            "input_ids": neg_input_ids,
                            "attention_mask": neg_attention_mask,
                        }
                    )

                # If the dataloader has sent an augmented image
                if "aug_image" in batch:
                    # Encode the augmented image
                    aug_image = batch["aug_image"]
                    aug_image_features = self.image_encoder(aug_image)

                # If the dataloader has sent an augmented text
                if "aug_input_ids" in batch:
                    # Encode the augmented text
                    aug_input_ids = batch["aug_input_ids"]
                    aug_attention_mask = batch["aug_attention_mask"]
                    aug_text_features = self.text_encoder(
                        {
                            "input_ids": aug_input_ids,
                            "attention_mask": aug_attention_mask,
                        }
                    )

            loss_dict = self.loss(
                image_features=image_features,
                text_features=text_features,
                neg_image_features=neg_image_features,
                neg_text_features=neg_text_features,
                aug_image_features=aug_image_features,
                aug_text_features=aug_text_features,
            )

            output_dict = {
                "loss": loss_dict["total_loss"],
                "loss_components": {
                    "total_loss": loss_dict["total_loss"].clone().detach(),
                    "cross_modal_loss": loss_dict["cross_modal_loss"].clone().detach(),
                    "visual_loss": loss_dict["visual_loss"].clone().detach(),
                    "textual_loss": loss_dict["textual_loss"].clone().detach(),
                },
            }

            return output_dict
