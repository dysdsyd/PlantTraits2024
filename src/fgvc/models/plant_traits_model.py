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
from functools import partial
import warnings
from typing import Any, Callable, Dict, List, Tuple, Optional, Union


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
                [-0.3060, 1.1513, -0.0671, 0.1698, 0.3407, 2.7966],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.std = torch.nn.Parameter(
            torch.tensor(
                [0.1226, 0.2133, 0.6449, 0.1594, 0.9975, 0.6355],
                dtype=torch.float32,
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
        train_tokens=False,
        ckpt_path=None,
        reg_head=True,
        clf_head=True,
        body="vitb",
    ):
        super(PlantDINO, self).__init__()
        self.le = LabelEncoder()
        self.train_blocks = train_blocks
        if body == "vitb":
            self.body = timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m",
                pretrained=True,
                num_classes=7806,
                checkpoint_path=ckpt_path,
            )
        elif body == "vitl":
            self.body = timm.create_model(
                "vit_large_patch14_reg4_dinov2.lvd142m",
                pretrained=True,
                num_classes=7806,
                checkpoint_path=None,
            )
        elif body == "vitg":
            self.body = timm.create_model(
                "vit_giant_patch14_reg4_dinov2.lvd142m",
                pretrained=True,
                num_classes=7806,
                checkpoint_path=None,
            )
        else:
            raise ValueError("Invalid body type")
        self.body.reset_classifier(num_targets, "avg")
        self.body.global_pool == "map"
        self.body.attn_pool = AttentionPoolLatent(
            self.body.embed_dim,
            num_heads=self.body.num_heads,
            mlp_ratio=self.body.mlp_ratio,
            norm_layer=self.body.norm_layer,
        )
        self.num_targets = num_targets

        for i, layer in enumerate([self.body.patch_embed, self.body.norm]):
            for p in layer.parameters():
                p.requires_grad = False

        if not train_tokens:
            self.body.cls_token.requires_grad = False
            self.body.pos_embed.requires_grad = False
            self.body.reg_token.requires_grad = False

        if self.train_blocks is not None:
            for i in range(0, len(self.body.blocks) - self.train_blocks):
                for p in self.body.blocks[i].parameters():
                    p.requires_grad = False

        self.tabular = StructuredSelfAttention(163, 128, num_blocks=4)
        self.reg_head = reg_head
        self.clf_head = clf_head
        del self.body.head

    def setup_heads(self):
        ### Regression head ###
        if self.reg_head:
            self.reg = nn.Sequential(
                nn.Linear(128 + self.body.num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, self.num_targets),
            )

        ### Classification head ###
        if self.clf_head:
            self.clf = nn.Sequential(
                nn.Linear(128 + self.body.num_features, 256),
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

        reg, clf = None, None
        # cat and regression
        x = torch.cat([x, x_], dim=1)
        if self.reg_head:
            reg = self.reg(x)
        if self.clf_head:
            clf = self.clf(x)
        return reg, clf


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


def config_dino_yolo_optimizers(
    model,
    optimizers: dict,
    schedulers: dict,
    lr_mult: float,
) -> tuple[dict]:
    total_params = 0
    opt = []
    sched = []
    ###########################################################
    # Optimizer and Scheduler for the reg ,clf head
    ###########################################################
    head_parameters = []
    if model.reg_traits:
        head_parameters += [p for p in model.model.reg.parameters() if p.requires_grad]
    if model.clf_traits:
        head_parameters += [p for p in model.model.clf.parameters() if p.requires_grad]

    head_parameters += [
        p for p in model.model.body.attn_pool.parameters() if p.requires_grad
    ]

    head_parameters += [p for p in model.model.tabular.parameters() if p.requires_grad]
    if head_parameters:
        if "head" in optimizers and "head" in schedulers:
            head_optimizer = optimizers["head"](head_parameters)
            head_scheduler = schedulers["head"](head_optimizer)
            opt.append(head_optimizer)
            sched.append(head_scheduler)
            total_params += sum([p.numel() for p in head_parameters])
            print("Added optimizer and scheduler for the head and tabular data.")
        else:
            raise ValueError(
                "Optimizer or scheduler configuration missing for head and tabular data."
            )
    else:
        warnings.warn("head and tab are non-trainable.")

    ###########################################################
    # Optimizer and Scheduler for blending weights
    ###########################################################
    bld_parameters = []
    if model.bld_traits:
        if model.reg_traits and model.reg_weight.requires_grad:
            bld_parameters += [model.reg_weight]
        if model.clf_traits and model.clf_weight.requires_grad:
            bld_parameters += [model.clf_weight]
        if model.soft_clf_traits and model.soft_clf_weight.requires_grad:
            bld_parameters += [model.soft_clf_weight]

        if bld_parameters:
            if "bld" in optimizers and "bld" in schedulers:
                bld_optimizer = optimizers["bld"](bld_parameters)
                bld_scheduler = schedulers["bld"](bld_optimizer)
                opt.append(bld_optimizer)
                sched.append(bld_scheduler)
                print("Added optimizer and scheduler for the blending weights.")
                total_params += sum([p.numel() for p in bld_parameters])
            else:
                raise ValueError(
                    "Optimizer or scheduler configuration missing for blending weights."
                )
        else:
            warnings.warn("blending weights are non-trainable.")

    ###########################################################
    # Optimizer and Scheduler for blocks
    ###########################################################
    model_layers = model.model.body.blocks
    num_layers = len(model_layers)

    # Extract the warmup start learning rate directly from the scheduler configuration
    max_lr = schedulers.blocks.keywords["max_lr"]

    for i, layer in enumerate(model_layers):
        layer_id = i + 1  # Layer IDs start at 1 for decay calculation

        # Calculate the scaled learning rate multiplier for the current layer
        max_lr_scaled = max_lr * (lr_mult ** (num_layers + 1 - layer_id))

        layer_parameters = [p for p in layer.parameters() if p.requires_grad]
        if layer_parameters:
            if "blocks" in optimizers and "blocks" in schedulers:
                # Create optimizer and scheduler with scaled learning rates
                current_optimizer = optimizers["blocks"](layer_parameters)
                current_scheduler = schedulers["blocks"](
                    current_optimizer, max_lr=max_lr_scaled
                )
                opt.append(current_optimizer)
                sched.append(current_scheduler)
                total_params += sum([p.numel() for p in layer_parameters])
                print(
                    f"Added optimizer and scheduler for block {i}, max LR: {max_lr:.8f}, scaled LR: {max_lr_scaled:.8f}"
                )
            else:
                raise ValueError(
                    f"Optimizer or scheduler configuration missing for blocks but block {i} is trainable."
                )
        else:
            warnings.warn(f"block {i} is non-trainable.")

    ###########################################################
    # Optimizers and Schedulers for tokens
    ###########################################################
    token_param_dict = {
        "cls_token": model.model.body.cls_token,
        "pos_embed": model.model.body.pos_embed,
        "reg_token": model.model.body.reg_token,
    }
    # Extract the warmup start learning rate directly from the scheduler configuration
    max_lr = schedulers.tokens.keywords["max_lr"]
    layer_id = 6  # Layer ID for tokens is 0
    max_lr_scaled = max_lr * (lr_mult ** (num_layers + 1 - layer_id))

    token_params = []
    for name, param in token_param_dict.items():
        if param.requires_grad:
            token_params.append(param)
        else:
            warnings.warn(f"{name} is non-trainable.")

    if len(token_params) > 0:
        if "tokens" in optimizers and "tokens" in schedulers:
            token_optimizer = optimizers["tokens"](
                token_params
            )  # Pass parameter wrapped in a list
            token_scheduler = schedulers["tokens"](
                token_optimizer, max_lr=max_lr_scaled
            )
            opt.append(token_optimizer)
            sched.append(token_scheduler)
            print(f"Added optimizer and scheduler for tokens")
            total_params += sum([p.numel() for p in token_params])
        else:
            raise ValueError(
                f"Optimizer and scheduler missing for tokens, but tokens are trainable"
            )

    # Creating dictionary entries for optimizers and schedulers
    opt_scheduler_dicts = [
        {"optimizer": o, "lr_scheduler": s} for o, s in zip(opt, sched)
    ]
    print(f"Total trainable parameters: {total_params}")
    return tuple(opt_scheduler_dicts)


class PlantTraitModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_fn: Callable = None,
        num_classes: int = 6,
        reg_traits: bool = True,
        clf_traits: bool = True,
        bld_traits: bool = False,
        soft_clf_traits: bool = False,
        cutmix_aug: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.model.reg_head = reg_traits
        self.model.clf_head = clf_traits or soft_clf_traits
        self.model.setup_heads()

        self.optimizer_fn = optimizer_fn
        self.automatic_optimization = False

        self.specie_traits = nn.Parameter(
            torch.load("/home/ubuntu/FGVC11/data/PlantTrait/specie_traits.pt"),
            requires_grad=False,
        )
        self.dummy_weights = nn.Parameter(
            torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        ##### Regression head #####
        self.reg_traits = reg_traits
        if self.reg_traits:
            self.reg_loss = R2Loss()
            self.reg_similarity = nn.CosineSimilarity(dim=1)
            self.train_reg_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )
            self.val_reg_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )

        ##### Classification head #####
        self.clf_traits = clf_traits
        if self.clf_traits:
            self.clf_loss = FocalLoss()
            self.train_clf_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )
            self.val_clf_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )

        ##### Regression using soft classifier head #####
        self.soft_clf_traits = soft_clf_traits
        if self.soft_clf_traits:
            self.soft_clf_train_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )
            self.soft_clf_val_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )

        # Classification accuracy
        if self.clf_traits or self.soft_clf_traits:
            self.train_clf_acc = MulticlassAccuracy(num_classes=17396, topk=10)
            self.val_clf_acc = MulticlassAccuracy(num_classes=17396, top_k=10)

        ##### Blend traits head #####
        self.bld_traits = bld_traits
        if self.bld_traits:
            if self.reg_traits:
                self.reg_weight = nn.Parameter(
                    torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        dtype=torch.float32,
                    ),
                    requires_grad=True,
                )
            if self.clf_traits:
                self.clf_weight = nn.Parameter(
                    torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        dtype=torch.float32,
                    ),
                    requires_grad=True,
                )
            if self.soft_clf_traits:
                self.soft_clf_weight = nn.Parameter(
                    torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        dtype=torch.float32,
                    ),
                    requires_grad=True,
                )
            self.blend_loss = R2Loss()
            self.blend_train_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )
            self.blend_val_R2 = R2Score(
                num_outputs=num_classes, multioutput="uniform_average"
            )

        ##### Cutmix Augmentation #####
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        assert (
            self.automatic_optimization == False
        ), "Manual optimization is not enabled"

        # zero grad for all optimizers
        opt = self.optimizers()
        for o in opt:
            o.zero_grad()

        # get batch
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

        total_loss = 0

        ##### Regression head #####
        if self.reg_traits:
            assert not torch.isnan(pred_enc).any()
            # raw predicted label
            pred = self.model.le.inverse_transform(pred_enc.clone().detach())
            assert not torch.isnan(pred).any()
            # regression loss
            reg_loss = self.reg_loss(pred_enc, y_enc)
            similarity_inv = (1 - self.reg_similarity(pred_enc, y_enc)).mean()
            total_loss += 1 * reg_loss + 0.4 * similarity_inv
            # log regression metrics
            self.log(
                "train/reg_loss",
                reg_loss,
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
            self.train_reg_R2(pred, y_reg)
            self.log(
                "train/reg_r2",
                self.train_reg_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # log clf accuracy
        if self.clf_traits or self.soft_clf_traits:
            self.train_clf_acc(specie_logits, y_clf)
            self.log(
                "train/clf_acc",
                self.train_clf_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        ##### Classification head #####
        if self.clf_traits:
            pred_specie = torch.argmax(specie_logits, dim=1)
            pred_specie_traits = self.specie_traits[pred_specie]
            clf_loss = self.clf_loss(specie_logits, y_clf)
            total_loss += 0.01 * clf_loss
            # log metrics
            self.log(
                "train/clf_loss",
                clf_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.train_clf_R2(pred_specie_traits, y_reg)
            self.log(
                "train/clf_r2",
                self.train_clf_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        ##### Soft classification as regression head #####
        if self.soft_clf_traits:
            specie_probs = F.softmax(specie_logits, dim=1)
            pred_specie_traits_soft = torch.matmul(specie_probs, self.specie_traits)
            assert not torch.isnan(pred_specie_traits_soft).any()
            self.soft_clf_train_R2(pred_specie_traits_soft, y_reg)
            self.log(
                "train/soft_clf_r2",
                self.soft_clf_train_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        #### Blend traits #####
        if self.bld_traits:
            assert (
                sum([self.reg_traits, self.clf_traits, self.soft_clf_traits]) > 1
            ), "At least two heads should be active to blend traits"
            bld_traits = torch.zeros_like(y_reg)
            denominator = torch.zeros_like(self.dummy_weights)
            if self.reg_traits:
                bld_traits += self.reg_weight * pred
                denominator += self.reg_weight
            if self.clf_traits:
                bld_traits += self.clf_weight * pred_specie_traits
                denominator += self.clf_weight
            if self.soft_clf_traits:
                bld_traits += self.soft_clf_weight * pred_specie_traits_soft
                denominator += self.soft_clf_weight
            bld_traits = bld_traits / denominator
            blend_loss = self.blend_loss(bld_traits, y_reg)
            total_loss += 0.1 * blend_loss
            self.log(
                "train/blend_loss",
                blend_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.blend_train_R2(bld_traits, y_reg)
            self.log(
                "train/blend_r2",
                self.blend_train_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.manual_backward(total_loss)
        for optimizer in self.optimizers():
            optimizer.step()
        return total_loss

    def on_train_epoch_end(self) -> None:
        for i, scheduler in enumerate(self.lr_schedulers()):
            scheduler.step()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, x_tab, y_reg, y_clf = (
            batch["image"],
            batch["metadata"],
            batch["label"],
            batch["specie"],
        )
        # assert that y is never nan
        assert not torch.isnan(y_reg).any()
        # encode label
        y_enc = self.model.le.transform(y_reg)
        assert not torch.isnan(y_enc).any()
        # predicts encoded label
        pred_enc, specie_logits = self.model.forward_alt(x, x_tab)

        ##### Regression head #####
        if self.reg_traits:
            assert not torch.isnan(pred_enc).any()
            # raw predicted label
            pred = self.model.le.inverse_transform(pred_enc.clone().detach())
            assert not torch.isnan(pred).any()
            # regression loss
            self.val_reg_R2(pred, y_reg)
            self.log(
                "val/reg_r2",
                self.val_reg_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # log clf accuracy
        if self.clf_traits or self.soft_clf_traits:
            self.val_clf_acc(specie_logits, y_clf)
            self.log(
                "val/clf_acc",
                self.val_clf_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        ##### Classification head #####
        if self.clf_traits:
            pred_specie = torch.argmax(specie_logits, dim=1)
            pred_specie_traits = self.specie_traits[pred_specie]
            self.val_clf_R2(pred_specie_traits, y_reg)
            self.log(
                "val/clf_r2",
                self.val_clf_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        ##### Soft classification as regression head #####
        if self.soft_clf_traits:
            specie_probs = F.softmax(specie_logits, dim=1)
            pred_specie_traits_soft = torch.matmul(specie_probs, self.specie_traits)
            assert not torch.isnan(pred_specie_traits_soft).any()
            self.soft_clf_val_R2(pred_specie_traits_soft, y_reg)
            self.log(
                "val/soft_clf_r2",
                self.soft_clf_val_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        #### Blend traits #####
        if self.bld_traits:
            assert (
                sum([self.reg_traits, self.clf_traits, self.soft_clf_traits]) > 1
            ), "At least two heads should be active to blend traits"
            bld_traits = torch.zeros_like(y_reg)
            denominator = torch.zeros_like(self.dummy_weights)
            if self.reg_traits:
                bld_traits += self.reg_weight * pred
                denominator += self.reg_weight
            if self.clf_traits:
                bld_traits += self.clf_weight * pred_specie_traits
                denominator += self.clf_weight
            if self.soft_clf_traits:
                bld_traits += self.soft_clf_weight * pred_specie_traits_soft
                denominator += self.soft_clf_weight
            bld_traits = bld_traits / denominator
            self.blend_val_R2(bld_traits, y_reg)
            self.log(
                "val/blend_r2",
                self.blend_val_R2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        d = self.optimizer_fn(self)
        return d
