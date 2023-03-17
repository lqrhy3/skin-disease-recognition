import glob
import os.path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import ImageDataset
from monai.transforms import (
    Compose,
    RandFlip,
    RandAffine,
    SpatialPad,
    RandSpatialCrop,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandAdjustContrast,
    Lambda,
    NormalizeIntensity,
    ScaleIntensityRange,
    Transpose,
    OneOf
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils.data import IMAGENET_MEAN, IMAGENET_STD


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
            ScaleIntensityRange(0, 255, 0., 1.),
            NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),

            RandFlip(prob=0.5, spatial_axis=1),
            RandAffine(
                prob=1.0,
                translate_range=(100, 100),
                scale_range=(0.25, 0.25),
                padding_mode='zeros'
            ),
            RandAffine(
                prob=0.5,
                rotate_range=np.pi / 12,
                shear_range=(0.1, 0.1),
                padding_mode='zeros'
            ),

            SpatialPad(spatial_size=image_size),
            RandSpatialCrop(image_size, random_size=False),

            RandGaussianNoise(prob=0.1, std=0.05),
            OneOf(
                transforms=[
                    RandGaussianSharpen(prob=0.75),
                    RandGaussianSmooth(prob=0.75, sigma_x=(0.5, 2.5), sigma_y=(0.5, 2.5)),
                ],
                weights=(0.3, 0.7)
            ),
            RandAdjustContrast(prob=0.9, gamma=(0.8, 1.25)),
        ])

        self.seg_transform = Compose([
            Transpose((2, 1, 0)),
            Lambda(lambda x: x[0:1]),
            ScaleIntensityRange(0, 255, 0., 1.),

            RandFlip(prob=0.5, spatial_axis=1),
            RandAffine(
                prob=1.0,
                translate_range=(100, 100),
                scale_range=(0.25, 0.25),
                padding_mode='zeros',
                mode='nearest'
            ),
            RandAffine(
                prob=0.5,
                rotate_range=np.pi / 12,
                shear_range=(0.1, 0.1),
                padding_mode='zeros',
                mode='nearest'
            ),
            SpatialPad(spatial_size=image_size),
            RandSpatialCrop(image_size, random_size=False),
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


if __name__ == '__main__':
    dm = DarkCirclesDataModule(
        '/home/lqrhy3/PycharmProjects/skin-deseases-detection-project/data/processed/dark_circles',
        image_size=(512, 512),
        train_split=0.7,
        batch_size=2
    )
    dm.setup()
    for i in range(10):
    # for i in [0] * 5:
        image, label = dm.data_train[i]
        image = image.numpy()
        label = label.numpy()
        print(image.shape)
        image = image * np.array(IMAGENET_STD)[:, None, None] + np.array(IMAGENET_MEAN)[:, None, None]
        image = np.clip(image, 0, 1).transpose((1, 2, 0))
        label = label.transpose((1, 2, 0))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(label)
        plt.title(i)
        plt.show()
