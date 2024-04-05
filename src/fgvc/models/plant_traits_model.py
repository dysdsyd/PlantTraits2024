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

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from typing import Any, Dict, Tuple


import timm
import torch.nn as nn


class PlantCNN(nn.Module):
    def __init__(self, num_targets=6):
        super(PlantCNN, self).__init__()
        self.train_tokens = False
        self.trainable_backbone_layers = 4
        self.body = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=True
        )
        for i, layer in enumerate([self.body.patch_embed, self.body.norm]):
            for p in layer.parameters():
                p.requires_grad = False

        if not self.train_tokens:
            self.body.cls_token.requires_grad = False
            self.body.pos_embed.requires_grad = False
            self.body.register_tokens.requires_grad = False
            self.body.mask_token.requires_grad = False

        if self.trainable_backbone_layers is not None:
            for i in range(0, len(self.body.blocks) - self.trainable_backbone_layers):
                for p in self.body.blocks[i].parameters():
                    p.requires_grad = False
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(
            self.body.num_features, self.body.num_features // 2
        )
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(self.body.num_features // 2, num_targets)

    def forward(self, x):
        x = self.body(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


class R2Loss(nn.Module):
    def __init__(self, use_mask=False):
        super(R2Loss, self).__init__()
        self.use_mask = use_mask

    def forward(self, y_true, y_pred):
        if self.use_mask:
            mask = y_true != -1
            y_true = torch.where(mask, y_true, torch.zeros_like(y_true))
            y_pred = torch.where(mask, y_pred, torch.zeros_like(y_pred))

        SS_res = torch.sum((y_true - y_pred) ** 2, dim=0)  # (B, C) -> (C,)
        SS_tot = torch.sum(
            (y_true - torch.mean(y_true, dim=0)) ** 2, dim=0
        )  # (B, C) -> (C,)
        r2_loss = SS_res / (SS_tot + 1e-6)  # (C,)
        return torch.mean(r2_loss)  # ()


class R2Metric(nn.Module):
    def __init__(self, num_classes=6):
        super(R2Metric, self).__init__()
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.SS_res = torch.zeros(self.num_classes)
        self.SS_tot = torch.zeros(self.num_classes)
        self.num_samples = torch.tensor(0, dtype=torch.float32)

    def forward(self, y_true, y_pred):
        # y_true = y_true.to(self.device)
        # y_pred = y_pred.to(self.device)

        SS_res = torch.sum((y_true - y_pred) ** 2, dim=0)
        SS_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
        self.SS_res += SS_res
        self.SS_tot += SS_tot
        self.num_samples += y_true.size(0)

    def compute(self):
        r2 = 1 - self.SS_res / (self.SS_tot + 1e-6)
        return torch.mean(r2)


class PlantTraitModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = R2Loss(use_mask=False)
        self.metrics = R2Score(num_outputs=6, multioutput="uniform_average")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch["image"], batch["label"]
        preds = self.model(x)
        loss = self.criterion(y, preds)
        metric = self.metrics(y, preds)
        self.log(
            "train/R2_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/R2_Metric",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch["image"], batch["label"]
        preds = self.model(x)
        loss = self.criterion(y, preds)
        metric = self.metrics(y, preds)
        self.log(
            "val/R2_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/R2_Metric",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        pass
        # x= batch["image"]
        # preds = self.model(x)
        # loss = self.criterion(y, preds)
        # metric = self.metrics(y, preds)
        # self.log(
        #     "test/R2_loss",
        #     loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     sync_dist=True,
        # )
        # self.log(
        #     "test/R2_Metric",
        #     metric,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     sync_dist=True,
        # )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/R2_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
