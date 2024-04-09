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
        self.sigm = nn.Sigmoid()

    def forward(self, inputs):
        return self.sigm(self.backbone(inputs))


class PlantCLEFModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterian: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterian
        # self.metrics = R2Score(num_outputs=6, multioutput="uniform_average")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds = self.model(batch["image"])
        loss = self.criterion(preds, batch["encoded_label"])
        # metric = self.metrics(y, preds)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # self.log(
        #     "train/R2_Metric",
        #     metric,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds = self.model(batch["image"])
        loss = self.criterion(preds, batch["encoded_label"])
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
           
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds = self.model(batch["image"])
        loss = self.criterion(preds, batch["encoded_label"])
        self.log(
            "test/loss",
            loss,
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
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
