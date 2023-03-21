import sys
from typing import Any, Optional

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from monai.visualize import blend_images

from src.utils.data import unnormalize_imagenet


class LogPredictionImagesCallback(pl.Callback):
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def on_train_batch_end(
        self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
    ) -> None:
        # `outputs` come from `LightningModule.train_step`
        if trainer.limit_train_batches != 1.0:
            if isinstance(trainer.limit_train_batches, int):
                dataloader_size = trainer.limit_train_batches
            elif isinstance(trainer.limit_train_batches, float):
                dataloader_size = int(trainer.limit_train_batches * len(trainer.train_dataloader))
            else:
                raise RuntimeError
        else:
            dataloader_size = len(trainer.train_dataloader)
        pre_last_batch_idx = dataloader_size - 2 if dataloader_size > 1 else 0
        if batch_idx == pre_last_batch_idx:
            self._on_trainval_batch_end('train', trainer, outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # `outputs` come from `LightningModule.validation_step`
        if trainer.limit_val_batches != 1.0:
            if isinstance(trainer.limit_val_batches, int):
                dataloader_size = trainer.limit_val_batches
            elif isinstance(trainer.limit_val_batches, float):
                dataloader_size = int(trainer.limit_val_batches * len(trainer.val_dataloaders[dataloader_idx]))
            else:
                raise RuntimeError
        else:
            dataloader_size = len(trainer.val_dataloaders[dataloader_idx])
        pre_last_batch_idx = dataloader_size - 2 if dataloader_size > 1 else 0
        if batch_idx == pre_last_batch_idx:
            self._on_trainval_batch_end('val', trainer, outputs, batch, batch_idx)

    def _on_trainval_batch_end(self, stage, trainer, outputs, batch, batch_idx):
        if trainer.sanity_checking:
            return
        wandb_logger = trainer.logger
        assert isinstance(wandb_logger, WandbLogger)

        masks_pred = outputs['preds'][:self.num_samples].detach().cpu()  # [N, 1, H, W]
        images = batch[0][:self.num_samples].cpu()  # [N, 3, H, W]
        masks_gt = batch[1][:self.num_samples].cpu()  # [N, 1, H, W]

        images = unnormalize_imagenet(images)

        raw_masks_vis = []
        blend_masks_vis = []
        for i in range(len(images)):
            image = images[i]
            mask_pred = masks_pred[i]
            mask_gt = masks_gt[i]

            raw_mask_vis = torch.cat([mask_pred, mask_gt], dim=-1).permute(1, 2, 0)
            raw_masks_vis.append(
                wandb.Image(raw_mask_vis, caption='prediction vs ground_truth')
            )
            blend_pred = blend_images(image, mask_pred, rescale_arrays=False, cmap='hsv')
            blend_gt = blend_images(image, mask_gt, rescale_arrays=False, cmap='winter')
            blend_mask_vis = torch.cat([blend_pred, blend_gt], dim=-1).permute(1, 2, 0)  # [H, 2*W, 3]
            blend_masks_vis.append(
                wandb.Image(blend_mask_vis, caption=f'prediction vs ground_truth {trainer.global_step}')
            )

        wandb_logger.experiment.log(
            {
                f'prediction_images/{stage}/masks': raw_masks_vis,
                f'prediction_images/{stage}/blend_masks': blend_masks_vis
            },
            # step=trainer.global_step
            commit=False
        )
