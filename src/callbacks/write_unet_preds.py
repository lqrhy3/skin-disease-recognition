import os
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import torch
from monai.transforms import CenterSpatialCrop
from pytorch_lightning.callbacks import BasePredictionWriter
from monai.data import decollate_batch


class SegmentationWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        # inferting transforms for segmentation mask
        # zdes' tolko kostyli i pizda
        assert prediction.shape[0] == 1
        pseudo_inverted_transform = CenterSpatialCrop((prediction.meta['height'][0], prediction.meta['width'][0]))
        prediction = pseudo_inverted_transform(prediction[0])

        prediction = (prediction * 255).permute(1, 2, 0)
        # prediction = decollate_batch(prediction, detach=True)
        path_to_save = self.output_dir / Path(prediction.meta['filename_or_obj']).name
        pred_sample = prediction.detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite(path_to_save.as_posix(), pred_sample)
