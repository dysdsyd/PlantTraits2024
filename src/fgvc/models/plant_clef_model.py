import os
import cv2
import torch
import joblib
import timm
import torch.nn as nn
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.regression import R2Score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.models import efficientnet
from torchmetrics.regression import R2Score
from torchmetrics.classification import MultilabelF1Score, MulticlassF1Score
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecisionRecallCurve,
    MulticlassAveragePrecision,
)

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from typing import Any, Dict, Tuple


import timm
import torch.nn as nn


# class PlantDINO(nn.Module):
#     def __init__(self, num_targets=6):
#         super(PlantDINO, self).__init__()
#         self.train_tokens = False
#         self.trainable_backbone_layers = 4
#         self.body = torch.hub.load(
#             "facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=True
#         )
#         for i, layer in enumerate([self.body.patch_embed, self.body.norm]):
#             for p in layer.parameters():
#                 p.requires_grad = False

#         if not self.train_tokens:
#             self.body.cls_token.requires_grad = False
#             self.body.pos_embed.requires_grad = False
#             self.body.register_tokens.requires_grad = False
#             self.body.mask_token.requires_grad = False

#         if self.trainable_backbone_layers is not None:
#             for i in range(0, len(self.body.blocks) - self.trainable_backbone_layers):
#                 for p in self.body.blocks[i].parameters():
#                     p.requires_grad = False
#         self.relu1 = nn.ReLU()
#         self.fc1 = nn.Linear(self.body.num_features, self.body.num_features // 2)
#         self.relu2 = nn.ReLU()
#         self.fc2 = nn.Linear(self.body.num_features // 2, num_targets)

#     def forward(self, x):
#         x = self.body(x)
#         x = self.relu1(x)
#         x = self.fc1(x)
#         x = self.relu2(x)
#         x = self.fc2(x)
#         return x


class TCBCELoss(nn.Module):
    def __init__(self, num_classes=7806, method="bce"):
        super().__init__()
        self.num_classes = num_classes
        self.method = method

    def forward(self, input, target):
        EPS = 1e-5
        if self.method == "bce":
            loss = -target * torch.log(input + EPS) - (1 - target) * torch.log(
                1 - input + EPS
            )
        elif self.method == "weak_negatives":
            loss = -target * torch.log(input + EPS) - (
                (1 - target) / (self.num_classes - 1)
            ) * torch.log(1 - input + EPS)

        elif self.method == "label_smoothing":
            loss = -target * torch.log(input + EPS) - (1 - target) * torch.log(
                1 - input + EPS
            )
        else:
            raise NotImplementedError
        return loss.mean()


def expected_positive_regularizer(preds, expected_num_pos, norm="2"):
    # Assumes predictions in [0,1].
    if norm == "1":
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == "2":
        reg = (preds.sum(1).mean(0) - expected_num_pos) ** 2
    else:
        raise NotImplementedError
    return reg


class TimmModel(nn.Module):
    def __init__(
        self,
        backbone="edgenext_small",
        num_classes=7806,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, num_classes=num_classes, pretrained=True
        )
        # self.sigm = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        return self.backbone(inputs)


class PlantCLEFModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 7806,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.criterion = TCBCELoss(num_classes=num_classes, method="bce")
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.train_ac = MulticlassAccuracy(
            num_classes=num_classes, average="macro", topk=1
        )
        self.val_ac = MulticlassAccuracy(
            num_classes=num_classes, average="macro", topk=1
        )
        self.test_ac = MulticlassAccuracy(
            num_classes=num_classes, average="macro", topk=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds = self.model(batch["image"])
        loss = self.criterion(preds, batch["encoded_label"])
        self.train_ac(
                preds.detach(),
                torch.argmax(batch["encoded_label"], dim=1)
                )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/accuracy",
            self.train_ac,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds = self.model(batch["image"])
        loss = self.criterion(preds, batch["encoded_label"])
        self.val_ac(
                preds.detach(),
                torch.argmax(batch["encoded_label"], dim=1)
                )
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/accuracy",
            self.val_ac,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds = self.model(batch["image"])
        loss = self.criterion(preds, batch["encoded_label"])
        self.test_ac(
                preds.detach(),
                torch.argmax(batch["encoded_label"], dim=1)
                )
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/accuracy",
            self.test_ac,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
