import glob
import os.path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from monai.data import ImageDataset, DataLoader
from monai.transforms import (
    Compose,
    RandFlip,
    RandAffine,
    SpatialPad,
    RandSpatialCrop,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    NormalizeIntensity,
    ScaleIntensityRange,
    Transpose,
    OneOf,
    RandZoom
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split

from src.utils.data import IMAGENET_MEAN, IMAGENET_STD


def get_fit_transforms(image_size: Tuple[int, int] or int) -> Compose:

    transform = Compose([
        Transpose((2, 1, 0)),
        ScaleIntensityRange(0, 255, 0., 1.),

        RandFlip(prob=0.5, spatial_axis=1),
        RandAffine(
            prob=1.0,
            translate_range=(100, 100),
            padding_mode='zeros',
        ),
        OneOf(
            transforms=[
                RandAffine(prob=1., scale_range=(0.2, 0.2), padding_mode='zeros'),
                RandZoom(prob=1., min_zoom=0.8, max_zoom=1.2, padding_mode='constant')
            ],
            weights=(0.2, 0.8)
        ),
        RandAffine(
            prob=0.25,
            rotate_range=np.pi / 18,
            shear_range=(0.05, 0.05),
            padding_mode='zeros'
        ),

        SpatialPad(spatial_size=image_size, mode='constant'),
        RandSpatialCrop(image_size, random_size=False),

        RandGaussianNoise(prob=0.1, std=0.05),
        RandGaussianSmooth(prob=0.25, sigma_x=(0.5, 2.5), sigma_y=(0.5, 2.5)),
        RandAdjustContrast(prob=0.9, gamma=(0.8, 1.25)),
        NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
    ])

    return transform


def get_predict_transforms(image_size: Tuple[int, int] or int) -> Compose:
    transform = Compose([
        Transpose((2, 1, 0)),
        ScaleIntensityRange(0, 255, 0., 1.),
        NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        SpatialPad(spatial_size=image_size),
    ])

    return transform


class DarkCirclesClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int or Tuple[int, int],
        train_split: float,
        batch_size: int,
        num_workers: int = 0,
        predict_data_dir: Optional[str] = None,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.transforms: Optional[Compose] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        self.predict_transforms: Optional[Compose] = None
        self.data_predict: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'predict':
            self.predict_transforms = get_predict_transforms(self.hparams.image_size)

            image_files = glob.glob(os.path.join(self.hparams.predict_data_dir, 'data', '*'))

            self.data_predict = ImageDataset(
                image_files=image_files,
                transform=self.predict_transforms
            )

            return

        if not self.data_train and not self.data_val:
            if not self.transforms:
                self.transforms = get_fit_transforms(self.hparams.image_size)

            dataset = ImageFolder(
                root=self.hparams.data_dir,
                transform=self.transforms
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

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_predict,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == '__main__':
    dm = DarkCirclesDataModule(
        '/home/lqrhy3/PycharmProjects/skin-deseases-detection-project/data/processed/dark_circles',
        image_size=(512, 512),
        train_split=0.7,
        batch_size=2
    )
    dm.setup()
    # for i in range(10):
    for i in [0] * 5:
        image, label = dm.data_train[i]
        image = image.numpy()
        label = label.numpy()
        print(image.shape)
        image = image * np.array(IMAGENET_STD)[:, None, None] + np.array(IMAGENET_MEAN)[:, None, None]
        image = np.clip(image, 0, 1).transpose((1, 2, 0))
        label = label.transpose((1, 2, 0))

        print(np.unique(label))

        plt.subplot(1, 2, 1)
        plt.imshow(image, vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.imshow(label)
        plt.title(i)
        plt.show()
