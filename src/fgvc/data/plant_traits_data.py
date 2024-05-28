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
from sklearn.preprocessing import StandardScaler

trait_columns = [
    "X4_mean",
    "X11_mean",
    "X18_mean",
    "X50_mean",
    "X26_mean",
    "X3112_mean",
]
aux_columns = list(map(lambda x: x.replace("mean", "sd"), trait_columns))


class PlantTraitsDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        return_image=True,
        return_labels=True,
        return_metadata=True,
        response_variation=False,
    ):
        self.df = df
        self.transform = transform
        self.return_image = return_image
        self.return_labels = return_labels
        self.return_metadata = return_metadata
        self.response_variation = response_variation
        self.class_names = trait_columns
        self.aux_class_names = aux_columns
        self.paths = self.df["path"].values

        if self.return_labels:
            self.labels = self.df[self.class_names].values
            self.aux_labels = self.df[self.aux_class_names].values
            self.species = self.df["species"].values

        if self.return_metadata:
            self.metadata = self.df[self.df.columns[1:164]].values
            # 163 columns
            assert (
                self.metadata.shape[1] == 163
            ), "Should be 163 metadata columns, got {self.metadata.shape[1]}"

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
            data["specie"] = torch.tensor(self.species[idx])
            data["aux_label"] = torch.tensor(self.aux_labels[idx], dtype=torch.float32)
            # apply response variation augmentation with mean and SD
            if self.response_variation:
                data["original_label"] = data["label"].clone()
                data["label"] = torch.clamp(torch.normal(data["label"], data["aux_label"]), min=0)
                # asser that the label is never NaN
                assert not torch.isnan(data["label"]).any()

        return data


class PlantTraitsDataModule(LightningDataModule):
    def __init__(
        self,
        df_path: str,
        transform=None,
        test_transform=None,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        collate_fn=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.df_path = df_path
        self.transform = transform
        self.test_transform = test_transform
        self.collate_fn = collate_fn

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(trait_columns)

    def prepare_data(self):
        """
        Download data if needed.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        # split data and create datasets
        self.df = pd.read_csv(self.df_path)
        if not self.data_train:
            self.data_train = PlantTraitsDataset(
                df=self.df[self.df["split"] == "train"],
                transform=self.transform,
                response_variation=False,
            )
        if not self.data_val:
            self.data_val = PlantTraitsDataset(
                df=self.df[self.df["split"] == "val"],
                transform=self.test_transform,
            )
        if not self.data_test:
            self.data_test = PlantTraitsDataset(
                df=self.df[self.df["split"] == "test"],
                transform=self.test_transform,
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
                drop_last=True,
            )

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size*2,
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
