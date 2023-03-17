import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.utils.data import unnormalize_imagenet


class LogPredictionImagesCallback(pl.Callback):
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # `outputs` come from `LightningModule.validation_step`
        if batch_idx != 0 or trainer.sanity_checking:
            return

        wandb_logger = trainer.logger
        assert isinstance(wandb_logger, WandbLogger)

        preds = outputs['preds'][:self.num_samples].detach()  # [N, 1, H, W]
        samples = batch[0][:self.num_samples]  # [N, 3, H, W]
        gts = batch[1][:self.num_samples]  # [N, 1, H, W]

        preds = (preds * 255).expand_as(samples)  # [N, 3, H, W]
        samples = torch.clamp(unnormalize_imagenet(samples) * 255, 0, 255)  # [N, 3, H, W]
        gts = (gts * 255).expand_as(samples)  # [N, 3, H, W]

        visualizations = torch.cat([preds, samples, gts], dim=-1)  # [N, 3, H, 3*W]
        visualizations = visualizations.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # [N, H, 3*W, 3]
        visualizations = [vis for vis in visualizations]

        # ochen' jal', but step passing does not work
        wandb_logger.log_image(key='prediction_images', images=visualizations)
