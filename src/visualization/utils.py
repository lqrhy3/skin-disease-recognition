from typing import List

import cv2
import numpy as np


def draw_bounding_boxes(image: np.ndarray, bboxes: List[int], color=(34, 187, 69)):
    for bbox in bboxes:
        xl, yt, xr, yb = bbox
        image = cv2.rectangle(image, (xl, yt), (xr, yb), color=color, thickness=3)

    return image
