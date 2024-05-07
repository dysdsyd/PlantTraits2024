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
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import Metric
from timm.layers import AttentionPoolLatent
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from typing import Any, Dict, Tuple
from torchmetrics.regression import R2Score
from fgvc.models.augmentations import Mixup_transmix


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


class RepeatedOneCycleLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_lr=1e-4,
        total_steps=150,
        epochs_per_cycle=30,
        lr_decay=0.7,
        last_epoch=-1,
    ):
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.epochs_per_cycle = epochs_per_cycle
        self.lr_decay = lr_decay
        self.one_cycle_scheduler = None
        super().__init__(optimizer, last_epoch, verbose=False)
        self._reset_scheduler(self.last_epoch)

    def _reset_scheduler(self, last_epoch):
        self.one_cycle_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            total_steps=self.epochs_per_cycle,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e3,
            last_epoch=last_epoch,
        )

    def get_lr(self):
        if self.last_epoch != self._step_count:
            self._step_count += 1
        return self.one_cycle_scheduler.get_lr()

    def step(self, epoch=None):
        # Increment the internal epoch count only after a cycle completes
        if self._step_count % self.epochs_per_cycle == 0:
            self.max_lr *= self.lr_decay
            self._reset_scheduler(-1)
        self.one_cycle_scheduler.step()

    def state_dict(self):
        state = super().state_dict()
        state["base_max_lr"] = self.base_max_lr
        state["max_lr"] = self.max_lr
        state["one_cycle_state_dict"] = self.one_cycle_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_max_lr = state_dict["base_max_lr"]
        self.max_lr = state_dict["max_lr"]
        self.one_cycle_scheduler.load_state_dict(state_dict["one_cycle_state_dict"])


class LabelEncoder(nn.Module):
    def __init__(self):
        """
        Initialize the encoder with a specific mean and variance.
        """
        super().__init__()
        self.mean = torch.nn.Parameter(
            torch.tensor(
                [-0.3060, 1.1513, -0.0671, 0.1698, 0.3407, 2.7966], dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.std = torch.nn.Parameter(
            torch.tensor(
                [0.1226, 0.2133, 0.6449, 0.1594, 0.9975, 0.6355], dtype=torch.float32
            ),
            requires_grad=False,
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
            log_X = torch.log10(X + 1e-6)
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


class MinMaxLabelEncoder(nn.Module):
    def __init__(self):
        """
        Initialize the encoder with specific minimum and maximum values.
        """
        super().__init__()
        self.min = nn.Parameter(
            torch.tensor(
                [
                    -0.5424844389045688,
                    0.6484638238914191,
                    -1.1240372205951517,
                    -0.10589238793491958,
                    -1.5553251870196876,
                    1.534919701938955,
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.max = nn.Parameter(
            torch.tensor(
                [
                    -0.08311075653326473,
                    1.541786160427708,
                    1.3064602333081552,
                    0.49182749638768486,
                    2.495978047145072,
                    3.995379650131168,
                ],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def transform(self, X):
        """
        Transform the labels by first taking their log scale and then applying
        Min-Max scaling.

        Parameters:
        - X: Input tensor of size n x c.

        Returns:
        - Scaled tensor of size n x c.
        """
        with torch.no_grad():
            log_X = torch.log10(X + 1e-6)
            scaled_X = (log_X - self.min) / (self.max - self.min)
        return scaled_X

    def inverse_transform(self, X):
        """
        Revert the scaled labels back to their original scale.

        Parameters:
        - X: Scaled tensor of size n x c.

        Returns:
        - Original labels tensor of size n x c.
        """
        with torch.no_grad():
            original_X = 10 ** (X * (self.max - self.min) + self.min)
        return original_X


class StructuredSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=1):
        super(StructuredSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            query = nn.Linear(input_dim, output_dim)
            key = nn.Linear(input_dim, output_dim)
            value = nn.Linear(input_dim, output_dim)

            # # Initialize the weights and biases
            # nn.init.kaiming_normal_(query.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.constant_(query.bias, 0)
            # nn.init.kaiming_normal_(key.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.constant_(key.bias, 0)
            # nn.init.kaiming_normal_(value.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.constant_(value.bias, 0)

            self.blocks.append(
                nn.ModuleDict({"query": query, "key": key, "value": value})
            )

        self.output = nn.Linear(output_dim * num_blocks, output_dim)
        nn.init.kaiming_normal_(self.output.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.output.bias, 0)

    def forward(self, x):
        block_outputs = []
        for block in self.blocks:
            Q = block["query"](x)
            K = block["key"](x)
            V = block["value"](x)

            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
                self.output_dim**0.5
            )
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_weights, V)
            block_outputs.append(attention_output)

        combined_output = torch.cat(block_outputs, dim=-1)
        projected_output = self.output(combined_output)
        return projected_output


class PlantDINO(nn.Module):
    def __init__(
        self,
        num_targets=6,
        train_blocks=4,
        ckpt_path=None,
    ):
        super(PlantDINO, self).__init__()
        self.le = LabelEncoder()
        self.train_tokens = True
        self.train_blocks = train_blocks
        self.body = timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=7806,  # initialize since this model is from PlantClef
            checkpoint_path=ckpt_path,
        )
        self.body.reset_classifier(num_targets, "avg")
        self.body.global_pool == "map"
        self.body.attn_pool = AttentionPoolLatent(
            self.body.embed_dim,
            num_heads=self.body.num_heads,
            mlp_ratio=self.body.mlp_ratio,
            norm_layer=self.body.norm_layer,
        )

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

        self.tabular = StructuredSelfAttention(163, 128, num_blocks=4)

        self.reg = nn.Sequential(
            nn.Linear(128 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_targets),
        )

        self.clf = nn.Sequential(
            nn.Linear(128 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 17396),
        )

    def forward(self, x):
        x = self.body(x)
        return x

    def forward_alt(self, x, x_):
        x = self.body.forward_features(x)

        x = self.body.forward_head(x, pre_logits=True)
        # pooled image features B * 768

        x_ = self.tabular(x_)
        # tabular features

        # cat and regression
        x = torch.cat([x, x_], dim=1)
        reg = self.reg(x)
        clf = self.clf(x)
        return reg, clf


class TraitBlender(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_blocks=1, n_models=2, dropout_rate=0.1
    ):
        super(TraitBlender, self).__init__()
        self.n_models = n_models
        # Adjust the input dimension based on the number of models (traits sets)
        self.self_attention = StructuredSelfAttention(
            input_dim * n_models, output_dim, num_blocks
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.final_layer = nn.Linear(output_dim, input_dim)

    def forward(self, traits_list):
        # Ensure we have the correct number of trait sets
        if len(traits_list) != self.n_models:
            raise ValueError(f"Expected {self.n_models}, but got {len(traits_list)}")

        # Concatenate the input traits along the feature dimension
        combined_input = torch.cat(traits_list, dim=1)

        # Process combined input through the Structured Self-Attention
        attention_output = self.self_attention(combined_input)

        # Apply dropout and pass through the final linear layer
        dropped_output = self.dropout(attention_output)
        blended_traits = self.final_layer(dropped_output)
        return blended_traits


class R2Loss(nn.Module):
    def __init__(self, num_classes=6):
        super(R2Loss, self).__init__()
        # Initialize learnable weights for each class, one weight per class
        # self.class_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32))
        # Increase weight for X_26_mean

    def forward(self, y_pred, y_true):
        # Calculate residual sum of squares per class
        SS_res = torch.sum((y_true - y_pred) ** 2, dim=0)  # (B, C) -> (C,)
        # Calculate total sum of squares per class
        SS_tot = torch.sum(
            (y_true - torch.mean(y_true, dim=0)) ** 2, dim=0
        )  # (B, C) -> (C,)
        # Calculate R2 loss per class, avoiding division by zero
        r2_loss = SS_res / (SS_tot + 1e-6)  # (C,)
        # Weight the R2 loss by the learnable class weights
        # weighted_r2_loss = self.class_weights * r2_loss
        # Return the mean of the weighted R2 loss
        return torch.mean(r2_loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else 1.0
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


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
            f"TC_r2_{trait_columns[i]}": r2_scores[i].item()
            for i in range(self.num_classes)
        }
        out["TC_r2"] = mean_r2_score.item()
        return out


class PlantTraitModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 6,
        cutmix_aug: bool = False,
        blend_traits: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.scheduler = RepeatedOneCycleLR(self.optimizer)
        self.criterion = R2Loss()
        self.similarity = nn.CosineSimilarity(dim=1)
        self.clf_loss = FocalLoss()
        # self.loss_weights = nn.Parameter(
        #     torch.tensor([1.0, 0.5, 0.01], dtype=torch.float32)
        # )
        self.cutmix_aug = cutmix_aug
        if self.cutmix_aug:
            self.trans_mix = Mixup_transmix(
                mixup_alpha=0.8,
                cutmix_alpha=1.0,
                cutmix_minmax=None,
                prob=0.6,
                switch_prob=0.3,
                mode="batch",
                correct_lam=True,
                label_smoothing=0.1,
                num_classes=17396,
            )
        self.specie_traits = nn.Parameter(
            torch.load("/home/ubuntu/FGVC11/data/PlantTrait/specie_traits.pt"),
            requires_grad=False,
        )

        self.train_metrics = R2Score(
            num_outputs=num_classes, multioutput="uniform_average"
        )
        self.val_metrics = R2Score(
            num_outputs=num_classes, multioutput="uniform_average"
        )
        self.tc_train_metrics = TCR2Score(num_classes=num_classes)
        self.tc_val_metrics = TCR2Score(num_classes=num_classes)

        self.train_clf_metrics = MulticlassAccuracy(num_classes=17396, topk=10)
        self.val_clf_metrics = MulticlassAccuracy(num_classes=17396, top_k=10)

        self.train_clf_R2 = R2Score(
            num_outputs=num_classes, multioutput="uniform_average"
        )
        self.val_clf_R2 = R2Score(
            num_outputs=num_classes, multioutput="uniform_average"
        )

        ##### Blend Traits #####
        self.blend_traits = blend_traits
        if self.blend_traits:
            self.reg_weight = nn.Parameter(
                torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    dtype=torch.float32,
                    requires_grad=True,
                )
            )
            self.clf_weight = nn.Parameter(
                torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    dtype=torch.float32,
                    requires_grad=True,
                )
            )
            # # Initialize the TraitBlender module
            # self.trait_blender = TraitBlender(
            #     input_dim=6, output_dim=6, num_blocks=2, dropout_rate=0.2, n_models=2
            # )
            self.blend_loss = R2Loss()
            # self.blend_sim = nn.CosineSimilarity(dim=1)
            self.blend_train_r2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )
            self.blend_val_r2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # image, metadata, raw label
        x, x_tab, y_reg, y_clf = (
            batch["image"],
            batch["metadata"],
            batch["label"],
            batch["specie"],
        )
        if self.cutmix_aug:
            # apply trans mixup augmentation
            x, x_tab, y_reg, y_clf, lam = self.trans_mix(x, x_tab, y_reg, y_clf)

        # assert that y is never nan
        assert not torch.isnan(y_reg).any()
        # encode label
        y_enc = self.model.le.transform(y_reg)
        assert not torch.isnan(y_enc).any()
        # predicts encoded label
        pred_enc, specie_logits = self.model.forward_alt(x, x_tab)
        assert not torch.isnan(pred_enc).any()
        # raw predicted label
        pred = self.model.le.inverse_transform(pred_enc.clone().detach())
        assert not torch.isnan(pred).any()

        # encoded/normalized labels for loss calculation
        reg_loss = self.criterion(pred_enc, y_enc)
        similarity_inv = (1 - self.similarity(pred_enc, y_enc)).mean()
        clf_loss = self.clf_loss(specie_logits, y_clf)
        loss = 1 * reg_loss + 0.4 * similarity_inv + 0.01 * clf_loss

        pred_specie = torch.argmax(specie_logits, dim=1)
        pred_specie_traits = self.specie_traits[pred_specie]

        self.log(
            "train/reg_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/similarity_inv",
            similarity_inv,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/clf_loss",
            clf_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_metrics(pred, y_reg)
        self.log(
            "train/r2",
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # self.train_clf_metrics(specie_logits, y_clf)
        # self.log(
        #     "train/clf_acc",
        #     self.train_clf_metrics,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        self.train_clf_R2(pred_specie_traits, y_reg)
        self.log(
            "train/clf_r2",
            self.train_clf_R2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.tc_train_metrics.update(pred, y_reg)

        #### Blend traits #####
        if self.blend_traits:
            blended_traits = (
                self.reg_weight * pred + self.clf_weight * pred_specie_traits
            ) / (self.reg_weight + self.clf_weight)
            
            blend_loss = self.blend_loss(blended_traits, y_reg)
            loss += blend_loss
            # # Blend the predicted traits with the predicted species traits
            # pred_clf_enc = self.model.le.transform(pred_specie_traits)
            # blended_traits_enc = self.trait_blender([pred_enc, pred_clf_enc])
            # assert not torch.isnan(blended_traits_enc).any()
            # # Calculate the R2 loss between the blended traits and the true traits
            # blend_loss = self.blend_loss(blended_traits_enc, y_enc)
            # blend_sim = (1 - self.blend_sim(blended_traits_enc, y_enc)).mean()
            # loss += 0.1 * blend_loss + 0.05 * blend_sim
            self.log(
                "train/blend_loss",
                blend_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            # self.log(
            #     "train/blend_sim_inv",
            #     blend_sim,
            #     on_step=False,
            #     on_epoch=True,
            #     prog_bar=True,
            #     sync_dist=True,
            # )
            # blended_traits = self.model.le.inverse_transform(blended_traits_enc)
            self.blend_train_r2(blended_traits, y_reg)
            self.log(
                "train/blend_r2",
                self.blend_train_r2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        for k, v in self.tc_train_metrics.compute().items():
            self.log(
                f"train/{k}",
                v,
                sync_dist=True,
            )
        self.tc_train_metrics.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # image, metadata, raw label
        x, x_, y, y_clf = (
            batch["image"],
            batch["metadata"],
            batch["label"],
            batch["specie"],
        )
        # encode label
        # y_enc = self.model.le.transform(y)
        # predicts encoded label
        pred_enc, specie_logits = self.model.forward_alt(x, x_)
        # raw predicted label
        pred = self.model.le.inverse_transform(pred_enc.clone().detach())

        pred_specie = torch.argmax(specie_logits, dim=1)
        pred_specie_traits = self.specie_traits[pred_specie]

        self.val_metrics(pred, y)
        self.log(
            "val/r2",
            self.val_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_clf_metrics(specie_logits, y_clf)
        self.log(
            "val/clf_acc",
            self.val_clf_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_clf_R2(pred_specie_traits, y)
        self.log(
            "val/clf_r2",
            self.val_clf_R2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.tc_val_metrics.update(pred, y)

        #### Blend traits #####
        if self.blend_traits:
            blended_traits = (
                self.reg_weight * pred + self.clf_weight * pred_specie_traits
            ) / (self.reg_weight + self.clf_weight)
            # pred_clf_enc = self.model.le.transform(pred_specie_traits)
            # blended_traits_enc = self.trait_blender([pred_enc, pred_clf_enc])
            # assert not torch.isnan(blended_traits_enc).any()
            # blended_traits = self.model.le.inverse_transform(blended_traits_enc)
            self.blend_val_r2(blended_traits, y)
            self.log(
                "val/blend_r2",
                self.blend_val_r2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self):
        for k, v in self.tc_val_metrics.compute().items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        self.tc_val_metrics.reset()

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
