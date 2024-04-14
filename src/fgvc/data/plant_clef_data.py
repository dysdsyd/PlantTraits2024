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
import random


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
                self.encoded_labels[idx].toarray()[0], dtype=torch.float
            )
        return data


class PlantSPECIESDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        label_encoder=None,
        n_repeat=1,
    ):
        self.df = df
        self.transform = transform
        self.le = label_encoder
        self.paths = self.df["path"].values
        self.labels = self.df["species_id"].values
        self.encoded_labels = self.le.transform(self.labels.reshape(-1, 1))
        self.unique_species = self.df["species_id"].unique()
        self.spc_to_idx = {spc: np.where(self.labels == spc)[0] for spc in self.unique_species}
        self.n_repeat = n_repeat

    def __len__(self):
        return len(self.unique_species) * self.n_repeat

    def __getitem__(self, idx):
        data = {}
        idx = idx % len(self.unique_species)
        obj_idx = random.choice(self.spc_to_idx[self.unique_species[idx]])
        image = cv2.imread(self.paths[obj_idx])[:, :, ::-1]
        if self.transform:
            image = self.transform(image=image)["image"]
        data["image"] = to_tensor(image)
        data["label"] = torch.tensor(self.labels[obj_idx], dtype=torch.float32)
        data["encoded_label"] = torch.tensor(
            self.encoded_labels[obj_idx].toarray()[0], dtype=torch.float
        )
        return data


class PlantMossaicDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        label_encoder=None,
    ):
        self.df = df
        self.transform = transform
        self.le = label_encoder
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
        self.species = self.df["species_id"].values
        # print(self.species)
        self.enc_species = self.le.transform(self.species.reshape(-1, 1))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}
        data["species"] = []
        data["enc_species"] = []
        # Create a blank canvas
        canvas = np.zeros((3000, 3000, 3), dtype=np.uint8)

        # Randomly select 4-6 indices
        num_images = random.randint(4, 10)
        selected_idxs = random.sample(range(len(self.paths)), num_images)

        # Fit the selected images randomly on the canvas without overlap
        for idx in selected_idxs:
            path = self.paths[idx]
            image = cv2.imread(path)[:, :, ::-1]
            h, w, _ = image.shape
            x = random.randint(0, 3000 - w)
            y = random.randint(0, 3000 - h)

            # Check if the selected image overlaps with any existing image on the canvas
            if np.any(canvas[y : y + h, x : x + w]):
                continue

            canvas[y : y + h, x : x + w] = image
            data["species"].append(self.species[idx])
            data["enc_species"].append(self.enc_species[idx].toarray()[0])

            # Apply the transform if provided
            if self.transform:
                canvas = self.transform(image=canvas)["image"]

            data["image"] = to_tensor(canvas)

        data["enc_species"] = np.array(data["enc_species"])
        # take OR across all the encoded labels
        data["enc_species"] = np.max(data["enc_species"], axis=0)

        return data

        # if self.return_labels:
        #     data["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        #     data["encoded_label"] = torch.tensor(
        #         self.encoded_labels[idx].toarray()[0], dtype=torch.int
        #     )

        # return data


class PlantCLEFDataModule(LightningDataModule):
    def __init__(
        self,
        df_path,
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
        self.df_path = df_path
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
        # load training data and splitting into train, val and test
        self.df = pd.read_csv(self.df_path, delimiter=";", escapechar="/")
        self.df_train = self.df[self.df["learn_tag"] == "train"].reset_index(drop=True)  # [:100]
        self.df_val = self.df[self.df["learn_tag"] == "val"].reset_index(drop=True)  # [:100]
        self.df_test = self.df[self.df["learn_tag"] == "test"].reset_index(drop=True)  # [:100]

        # load and split datasets only if not loaded already
        if not self.data_train:
            # self.data_train = PlantCLEFDataset(
            #     self.df_train,
            #     transform=self.transform,
            #     label_encoder=self.le,
            # )
            self.data_train = PlantSPECIESDataset(
                self.df_train,
                transform=self.transform,
                label_encoder=self.le,
                n_repeat=10,
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
