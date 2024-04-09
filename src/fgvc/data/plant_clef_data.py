import os
import torch
import cv2
import pickle
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from tqdm.notebook import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.preprocessing import OneHotEncoder


class PlantCLEFDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        label_encoder=None,
        return_image=True,
        return_labels=True,
        return_metadata=False,
    ):
        self.df = df
        self.transform = transform
        self.le = label_encoder
        self.return_image = return_image
        self.return_labels = return_labels
        self.return_metadata = return_metadata
        self.metadata_cols = [
            "organ",
            "altitude",
            "latitude",
            "longitude",
            "species",
            "genus",
            "family",
        ]
        self.paths = self.df["path"].values
        if self.return_labels:
            self.labels = self.df["species_id"].values
            self.encoded_labels = self.le.transform(self.labels.reshape(-1, 1))
        if self.return_metadata:
            self.metadata = self.df[self.metadata_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}
        if self.return_image:
            image = cv2.imread(self.paths[idx])[:, :, ::-1]
            if self.transform:
                image = self.transform(image=image)["image"]
            data["image"] = to_tensor(image)
        if self.return_metadata:
            data["metadata"] = torch.tensor(self.metadata[idx], dtype=torch.float32)
        if self.return_labels:
            data["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
            data["encoded_label"] = torch.tensor(
                self.encoded_labels[idx].toarray()[0], dtype=torch.int
            )
        return data


class PlantCLEFDataModule(LightningDataModule):
    def __init__(
        self,
        df_train,
        label_encoder="/home/ubuntu/FGVC11/data/PlantClef/le.pkl",
        transform=None,
        test_transform=None,
        batch_size=64,
        num_workers=8,
        pin_memory=False,
        collate_fn=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # load training data and splitting into train, val and test
        self.df = pd.read_csv(df_train, delimiter=";", escapechar="/")
        self.df_train = self.df[self.df["learn_tag"] == "train"]
        self.df_val = self.df[self.df["learn_tag"] == "val"]
        self.df_test = self.df[self.df["learn_tag"] == "test"]
        # loading transforms
        self.transform = transform
        self.test_transform = test_transform
        self.collate_fn = collate_fn

        # load one hot encoder for species id
        assert label_encoder is not None, "Label encoder path is required"
        try:
            with open(label_encoder, "rb") as f:
                self.le = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Label Encoder file not found")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.le.categories_[0])

    def prepare_data(self):
        """Download data if needed.

        Terraclear datasets manage downloading opaquely so this is unused.

        DO NOT use this function to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train:
            self.data_train = PlantCLEFDataset(
                self.df_train,
                transform=self.transform,
                label_encoder=self.le,
            )
        if not self.data_val:
            self.data_val = PlantCLEFDataset(
                self.df_val,
                transform=self.test_transform,
                label_encoder=self.le,
            )
        if not self.data_test:
            self.data_test = PlantCLEFDataset(
                self.df_test,
                transform=self.test_transform,
                label_encoder=self.le,
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
