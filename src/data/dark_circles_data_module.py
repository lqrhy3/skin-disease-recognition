import glob
import os.path
from typing import Optional, Tuple

import torch
from monai.data import ImageDataset
from monai.transforms import (
    Compose,
    Lambda,
    NormalizeIntensity,
    Resize,
    ScaleIntensityRange,
    Transpose
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]


class DarkCirclesDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int or Tuple[int, int],
        train_split: float,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transform = Compose([
            Transpose((2, 1, 0)),
            Resize(image_size),
            ScaleIntensityRange(0, 255, 0., 1.),
            NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ])
        self.seg_transform = Compose([
            Transpose((2, 1, 0)),
            Lambda(lambda x: x[0:1]),
            Resize(image_size, mode='nearest-exact'),
            ScaleIntensityRange(0, 255, 0., 1.),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val:
            image_files = glob.glob(os.path.join(self.hparams.data_dir, 'data', '*'))
            seg_files = glob.glob(os.path.join(self.hparams.data_dir, 'labels', '*'))

            dataset = ImageDataset(
                image_files=image_files,
                seg_files=seg_files,
                transform=self.transform,
                seg_transform=self.seg_transform
            )
            trainset_size = int(len(dataset) * self.hparams.train_split)
            valset_size = len(dataset) - trainset_size

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[trainset_size, valset_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
