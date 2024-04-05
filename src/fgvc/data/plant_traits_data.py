import os
import torch
import cv2
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class PlantTraitsDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        return_image=True,
        return_labels=True,
        return_metadata=False,
    ):
        self.df = df
        self.transform = transform
        self.return_image = return_image
        self.return_labels = return_labels
        self.return_metadata = return_metadata
        self.class_names = [
            "X4_mean",
            "X11_mean",
            "X18_mean",
            "X26_mean",
            "X50_mean",
            "X3112_mean",
        ]
        self.aux_class_names = list(
            map(lambda x: x.replace("mean", "sd"), self.class_names)
        )
        if "id" in self.df.columns:
            self.df = self.df.drop(columns=["id"])
        self.paths = self.df["path"].values
        if self.return_labels:
            self.labels = self.df[self.class_names].values
            self.aux_labels = self.df[self.aux_class_names].values
        if self.return_metadata:
            self.metadata = self.df.drop(
                columns=["path"] + self.class_names + self.aux_class_names
            ).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}
        if self.return_image:
            image = cv2.imread(self.paths[idx])[:,:,::-1]
            if self.transform:
                image = self.transform(image=image)["image"]
            data["image"] = to_tensor(image)
        if self.return_metadata:
            data["metadata"] = torch.tensor(self.metadata[idx], dtype=torch.float32)
        if self.return_labels:
            data["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
            data["aux_label"] = torch.tensor(self.aux_labels[idx], dtype=torch.float32)

        return data


class PlantTraitsDataModule(LightningDataModule):
    def __init__(
        self,
        df_train,
        df_test,
        transform=None,
        val_transform=None,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        collate_fn=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.df_train  = pd.read_csv(df_train)
        self.df_val = pd.read_csv(df_train)
        self.df_test = pd.read_csv(df_test)
        self.transform = transform
        self.val_transform = val_transform
        self.collate_fn = collate_fn

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 6

    def prepare_data(self):
        """Download data if needed.

        Terraclear datasets manage downloading opaquely so this is unused.

        DO NOT use this function to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train:
            self.data_train = PlantTraitsDataset(
                self.df_train,
                transform=self.transform,
            )
        if not self.data_val:
            self.data_val = PlantTraitsDataset(
                self.df_val,
                transform=self.val_transform,
            )
        if not self.data_test:
            self.data_test = PlantTraitsDataset(
                self.df_test,
                transform=self.val_transform,
                return_labels=False,
            )

    def train_dataloader(self):
        if self.data_train is not None:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                collate_fn=self.collate_fn,
                persistent_workers=True,
            )

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=self.collate_fn,
                persistent_workers=True,
            )

    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=self.collate_fn,
                persistent_workers=True,
            )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
