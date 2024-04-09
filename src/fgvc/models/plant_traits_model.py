import os
import cv2
import torch
import pickle
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
from torchmetrics import Metric

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from typing import Any, Dict, Tuple


import timm
import torch.nn as nn

trait_columns = [
    "X4_mean",
    "X11_mean",
    "X18_mean",
    "X50_mean",
    "X26_mean",
    "X3112_mean",
]
aux_columns = list(map(lambda x: x.replace("mean", "sd"), trait_columns))


class LabelEncoder(nn.Module):
    def __init__(self):
        """
        Initialize the encoder with a specific mean and variance.
        """
        super().__init__()
        self.mean = torch.nn.Parameter(
            torch.tensor(
                [-0.3060, 1.1513, -0.0671, 0.1698, 0.3407, 2.7966], dtype=torch.float32
            )
        )
        self.std = torch.nn.Parameter(
            torch.tensor(
                [0.1226, 0.2133, 0.6449, 0.1594, 0.9975, 0.6355], dtype=torch.float32
            )
        )

    def transform(self, X):
        """
        Transform the labels by first taking their log scale and then
        standardizing.

        Parameters:
        - X: Input tensor of size n x c.

        Returns:
        - Transformed tensor of size n x c.
        """
        with torch.no_grad():
            log_X = torch.log10(X)
            standardized_X = (log_X - self.mean) / self.std
        return standardized_X

    def inverse_transform(self, X):
        """
        Revert the labels back to their original scale.

        Parameters:
        - X: Transformed tensor of size n x c.

        Returns:
        - Original labels tensor of size n x c.
        """
        with torch.no_grad():
            original_X = 10 ** (X * self.std + self.mean)
        return original_X


class PlantDINO(nn.Module):
    def __init__(self, num_targets=6, trainable_backbone_layers=4):
        super(PlantDINO, self).__init__()
        self.le = LabelEncoder()
        self.train_tokens = False
        self.trainable_backbone_layers = trainable_backbone_layers
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

        self.tabular = nn.Sequential(
            nn.Linear(163, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.reg = nn.Sequential(
            nn.Linear(128 + self.body.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_targets),
        )

    def forward(self, x, dense_input):
        x = self.body(x)
        x_ = self.tabular(dense_input)
        x = torch.cat([x, x_], dim=1)
        x = self.reg(x)
        return x


class TimmModel(nn.Module):
    def __init__(
        self,
        backbone="swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        num_classes=6,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, num_classes=num_classes, pretrained=True
        )
        self.le = LabelEncoder()

    def forward(self, inputs, extra_input=None):
        return self.backbone(inputs)


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


class TCR2Score(Metric):
    def __init__(self, num_classes=6):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        # We'll store sums and counts for each class to compute individual R2 scores
        self.add_state(
            "sum_squared_errors", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "sum_squared_totals", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target shape: [batch_size, num_classes]
        if preds.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected preds to have shape [batch_size, num_classes], got {preds.shape}"
            )
        if target.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected target to have shape [batch_size, num_classes], got {target.shape}"
            )

        errors = target - preds
        squared_errors = torch.square(errors)
        self.sum_squared_errors += torch.sum(squared_errors, dim=0)

        mean_target = torch.mean(target, dim=0)
        totals = target - mean_target
        squared_totals = torch.square(totals)
        self.sum_squared_totals += torch.sum(squared_totals, dim=0)

        self.count += target.size(0)

    def compute(self):
        mean_squared_errors = self.sum_squared_errors / self.count
        mean_squared_totals = self.sum_squared_totals / self.count
        # R2 score for each class
        r2_scores = 1 - (mean_squared_errors / (mean_squared_totals + 1e-6))
        # Mean R2 score across all classes
        mean_r2_score = torch.mean(r2_scores)
        out = {
            f"r2_{trait_columns[i]}": r2_scores[i].item()
            for i in range(self.num_classes)
        }
        out["r2"] = mean_r2_score.item()
        return out


class PlantTraitModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = R2Loss()  # criterion
        self.metrics = TCR2Score(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # image, metadata, raw label
        x, x_, y = batch["image"], batch["metadata"], batch["label"]
        # encode label
        y_enc = self.model.le.transform(y)
        # predicts encoded label
        pred_enc = self.model(x, x_)
        # raw predicted label
        pred = self.model.le.inverse_transform(pred_enc.clone().detach())

        # encoded/normalized labels for loss calculation
        loss = self.criterion(y_enc, pred_enc)
        # raw labels for metric calculation
        metrics = self.metrics(y, pred)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in metrics.items():
            self.log(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # image, metadata, raw label
        x, x_, y = batch["image"], batch["metadata"], batch["label"]
        # encode label
        y_enc = self.model.le.transform(y)
        # predicts encoded label
        pred_enc = self.model(x, x_)
        # raw predicted label
        pred = self.model.le.inverse_transform(pred_enc.clone().detach())

        # encoded/normalized labels for loss calculation
        loss = self.criterion(y_enc, pred_enc)
        # raw labels for metric calculation
        metrics = self.metrics(y, pred)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for k, v in metrics.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        pass

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
