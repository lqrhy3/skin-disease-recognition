from copy import deepcopy
from typing import Tuple

import dlib
import sys
import cv2
import numpy as np


LEFT_EYE_IDXS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDXS = [42, 43, 44, 45, 46, 47]


def draw_face_landmarks(image: np.ndarray, landmarks, color: Tuple = (0, 255, 0), radius: int = 3) -> np.ndarray:
    image = deepcopy(image)
    for i, p in enumerate(landmarks.parts()):
        cv2.circle(image, (p.x, p.y), radius, color, -1)
        cv2.putText(image, str(i), (p.x + 5, p.y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return image


class LandmarksDlibPredictor:
    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def __call__(self, image: np.ndarray):
        """
        :poram image: RGB
        """
        faces = self.detector(image, 0)
        if len(faces) < 1:
            return None
        face = faces[0]
        bbox = dlib.rectangle(int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        landmarks = self.predictor(image, bbox)
        return landmarks


def create_eyes_mask(image: np.ndarray, face_landmarks) -> np.ndarray:
    mask = np.zeros((image.shape[0], image.shape[1]))

    landmarks_list = face_landmarks.parts()

    left_eye_pts = [[landmarks_list[idx].x, landmarks_list[idx].y] for idx in LEFT_EYE_IDXS]
    right_eye_pts = [[landmarks_list[idx].x, landmarks_list[idx].y] for idx in RIGHT_EYE_IDXS]
    left_eye_pts = np.array(left_eye_pts)
    right_eye_pts = np.array(right_eye_pts)

    cv2.fillPoly(mask, [left_eye_pts, right_eye_pts], color=(255, 255, 255))
    return mask


if __name__ == '__main__':
    path_to_predictor = '/Users/vladimir/dev/data/shape_predictor_68_face_landmarks.dat'
    landmarks_predictor = LandmarksDlibPredictor(path_to_predictor)
    path_to_image = '/Users/vladimir/dev/data/dark_circles_train/data/IMG_0626.PNG'
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = landmarks_predictor(image)
    result_image = draw_face_landmarks(image, landmarks)
    mask = create_eyes_mask(image, landmarks)
    masked_image = np.logical_not(mask)[:, :, None] * image
    pass
