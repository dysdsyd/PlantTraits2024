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
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecisionRecallCurve,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelAveragePrecision,
    MultilabelPrecision,
    MultilabelRecall,
)

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from typing import Any, Dict, Tuple
import random

import timm
from timm.layers import AttentionPoolLatent
import torch.nn as nn


class ConvClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(ConvClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_features, out_channels=512, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization after conv1
        self.dropout1 = nn.Dropout(0.25)  # Dropout layer after BN

        self.conv2 = nn.Conv1d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(256)  # Batch normalization after conv2
        self.dropout2 = nn.Dropout(0.25)  # Dropout layer after BN

        # Adaptive pooling: specify output size, e.g., (1) to pool to a single value
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
        self.dropout_fc = nn.Dropout(0.5)  # Dropout before the fully connected layer

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolutional layers with Kaiming normalization
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")

        # Initialize linear layer weights with Xavier normalization
        nn.init.xavier_normal_(self.fc.weight)
        # Set biases to zero (if applicable)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Permute for Conv1d (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        # Apply convolutional layers with ReLU activation, batch normalization, and dropout
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))

        # Apply adaptive pooling to handle variable lengths
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer

        # Apply dropout before the final fully connected layer for classification
        x = self.dropout_fc(x)
        x = self.fc(x)
        return x


class LearnableWeightedBCELoss(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightedBCELoss, self).__init__()
        # Initialize positive weights as learnable parameters
        self.pos_weights = nn.Parameter(
            torch.ones(num_classes) * 0.5
        )  # Initial weight for positive examples

    def forward(self, inputs, targets):
        # Ensure positive weights are positive and normalized between 0 and 1
        pos_weights = torch.sigmoid(self.pos_weights)

        # Calculate binary cross-entropy manually for numerical stability and flexibility
        EPS = 1e-8
        bce_loss = -targets * torch.log(inputs + EPS) - (1 - targets) * torch.log(
            1 - inputs + EPS
        )

        # Apply positive dynamic weights specifically to the positive parts of the loss
        weighted_loss = pos_weights * bce_loss * targets + bce_loss * (1 - targets)

        return weighted_loss.mean()


class TCFocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(TCFocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        EPS = 1e-8
        # Calculate the BCE loss for each element in the batch and class
        bce_loss = -target * torch.log(input + EPS) - (1 - target) * torch.log(
            1 - input + EPS
        )

        # Calculate the probabilities of being correct
        pt = input * target + (1 - input) * (1 - target)

        # Calculate the focal factor
        focal_factor = (1 - pt) ** self.gamma

        # Adjust the loss using the focal factor and alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_bce_loss = alpha_t * focal_factor * bce_loss

        return focal_bce_loss.mean()


def expected_positive_regularizer(preds, expected_num_pos, norm="2"):
    # Assumes predictions in [0,1].
    if norm == "1":
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == "2":
        reg = (preds.sum(1).mean(0) - expected_num_pos) ** 2
    else:
        raise NotImplementedError
    return reg


class AdvancedMultiLabelClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(AdvancedMultiLabelClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_features, out_channels=512, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(256)

        # Adaptive pooling to handle varying sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Attention mechanism
        self.attention_fc = nn.Linear(256, 256)
        self.attention_out = nn.Linear(256, 1, bias=False)
        # Dense interaction layer
        self.interaction = nn.Linear(256, 256)

        # Dropout and Output layer
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Correct dimension ordering

        # Apply convolutional layers and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Adaptive pooling
        x = self.adaptive_pool(x).squeeze(-1)  # Remove the last dimension after pooling

        # Attention mechanism
        attention_weights = F.softmax(
            self.attention_out(F.tanh(self.attention_fc(x))), dim=1
        )
        x = torch.sum(x * attention_weights, dim=1)  # Apply attention weights

        # Dense interaction and final output
        x = F.relu(self.interaction(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class PlantDINO(nn.Module):
    def __init__(
        self,
        num_targets=7806,
        train_blocks=4,
        ckpt_path=None,
    ):
        super(PlantDINO, self).__init__()
        self.train_tokens = False
        self.train_blocks = train_blocks
        self.body = timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            pretrained=False,
            num_classes=7806,  # initialize since this model is from PlantClef
            checkpoint_path=ckpt_path,
        )
        # self.body.reset_classifier(num_targets, "avg")
        # self.body.global_pool == "map"
        # self.body.attn_pool = AttentionPoolLatent(
        #     self.body.embed_dim,
        #     num_heads=self.body.num_heads,
        #     mlp_ratio=self.body.mlp_ratio,
        #     norm_layer=self.body.norm_layer,
        # )

        for i, layer in enumerate([self.body.patch_embed, self.body.norm]):
            for p in layer.parameters():
                p.requires_grad = False

        if not self.train_tokens:
            self.body.cls_token.requires_grad = False
            self.body.pos_embed.requires_grad = False
            self.body.reg_token.requires_grad = False
            # self.body.mask_token.requires_grad = False

        if self.train_blocks is not None:
            for i in range(0, len(self.body.blocks) - self.train_blocks):
                for p in self.body.blocks[i].parameters():
                    p.requires_grad = False

        self.clf = ConvClassifier(
            input_features=self.body.num_features, num_classes=num_targets
        )
        # delete body.head
        del self.body.head

    def forward(self, x):
        x = self.body(x)
        return x


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


class CostSensitiveBCELoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        pos_weights = self.num_classes / 10
        weights = (1 - targets) + (targets) * pos_weights

        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, weight=weights, reduction="mean"
        )
        return BCE_loss


def reshape_tensor_labels(tensor, labels, N):
    # Original dimensions
    seq_len, feature_dim = tensor.size(1), tensor.size(2)
    _, num_classes = labels.size()

    # Reshape the tensor and labels
    reshaped_tensor = tensor.view(-1, N * seq_len, feature_dim)
    reshaped_labels = labels.view(-1, N, num_classes).sum(dim=1)
    reshaped_labels = (reshaped_labels > 0).float()
    # Convert to binary classification target

    return reshaped_tensor, reshaped_labels


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
        # self.criterion = TCFocalBCELoss(gamma=2.0, alpha=0.25)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = TCFocalBCELoss()

        self.train_f1 = MultilabelF1Score(num_classes, average="macro")
        self.train_ac = MultilabelAccuracy(num_classes, average="macro")
        self.train_prec = MultilabelPrecision(num_classes, average="macro")
        self.train_recall = MultilabelRecall(num_classes, average="macro")

        self.val_f1 = MultilabelF1Score(num_classes, average="macro")
        self.val_ac = MultilabelAccuracy(num_classes, average="macro")
        self.val_prec = MultilabelPrecision(num_classes, average="macro")
        self.val_recall = MultilabelRecall(num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        image, labels = batch["image"], batch["encoded_label"]
        feats = self.model.body.forward_features(image)
        N = min(random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]), feats.size(0))
        feats, labels = reshape_tensor_labels(feats, labels, N)
        logits = self.model.clf(feats)
        preds = torch.sigmoid(logits)
        loss = self.criterion(logits, labels)

        self.train_f1(preds.detach(), labels)
        self.train_ac(preds.detach(), labels)
        self.train_prec(preds.detach(), labels)
        self.train_recall(preds.detach(), labels)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/F1",
            self.train_f1,
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
        self.log(
            "train/precision",
            self.train_prec,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/recall",
            self.train_recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        image, labels = batch["image"], batch["encoded_label"]
        feats = self.model.body.forward_features(image)
        # N = random.randint(1, 16)
        # feats, labels = reshape_and_pad(feats, labels, N)
        logits = self.model.clf(feats)
        preds = torch.sigmoid(logits)
        loss = self.criterion(logits, labels)

        self.val_f1(preds.detach(), labels)
        self.val_ac(preds.detach(), labels)
        self.val_prec(preds.detach(), labels)
        self.val_recall(preds.detach(), labels)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/F1",
            self.val_f1,
            on_step=False,
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
        self.log(
            "val/precision",
            self.val_prec,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # return loss

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


#########################################################################################################################################


class VectorPlantCLEFModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 7806,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.clf = ConvClassifier(
            input_features=768, num_classes=num_classes
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.criterion = TCBCELoss(num_classes=num_classes, method="bce")
        # self.criterion = TCFocalBCELoss(gamma=2.0, alpha=0.25)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = CostSensitiveBCELoss(num_classes)

        self.train_f1 = MultilabelF1Score(num_classes, average="macro")
        self.train_ac = MultilabelAccuracy(num_classes, average="macro")
        self.train_prec = MultilabelPrecision(num_classes, average="macro")
        self.train_recall = MultilabelRecall(num_classes, average="macro")

        self.val_f1 = MultilabelF1Score(num_classes, average="macro")
        self.val_ac = MultilabelAccuracy(num_classes, average="macro")
        self.val_prec = MultilabelPrecision(num_classes, average="macro")
        self.val_recall = MultilabelRecall(num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # load features and labels
        feats, labels = batch["vector"], batch["encoded_label"]
        N = min(random.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24]), feats.size(0))

        feats, labels = reshape_tensor_labels(feats, labels, N)

        logits = self.clf(feats)
        preds = torch.sigmoid(logits)
        loss = self.criterion(logits, labels)

        self.train_f1(preds.detach(), labels)
        self.train_ac(preds.detach(), labels)
        self.train_prec(preds.detach(), labels)
        self.train_recall(preds.detach(), labels)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/F1",
            self.train_f1,
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
        self.log(
            "train/precision",
            self.train_prec,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/recall",
            self.train_recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        feats, labels = batch["vector"], batch["encoded_label"]

        logits = self.clf(feats)
        preds = torch.sigmoid(logits)
        loss = self.criterion(logits, labels)

        self.val_f1(preds.detach(), labels)
        self.val_ac(preds.detach(), labels)
        self.val_prec(preds.detach(), labels)
        self.val_recall(preds.detach(), labels)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/F1",
            self.val_f1,
            on_step=False,
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
        self.log(
            "val/precision",
            self.val_prec,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # return loss

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
