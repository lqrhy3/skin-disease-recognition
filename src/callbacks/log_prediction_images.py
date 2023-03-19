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
        if batch_idx != 0 or trainer.sanity_checking:
            return

        wandb_logger = trainer.logger
        assert isinstance(wandb_logger, WandbLogger)

        masks_pred = outputs['preds'][:self.num_samples].detach()  # [N, 1, H, W]
        images = batch[0][:self.num_samples]  # [N, 3, H, W]
        masks_gt = batch[1][:self.num_samples]  # [N, 1, H, W]

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
                f'prediction_images/val/masks': raw_masks_vis,
                f'prediction_images/val/blend_masks': blend_masks_vis
            },
            step=trainer.global_step
        )
