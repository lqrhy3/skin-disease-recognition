import math
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2

from src.models.face_landmarks import (
    LandmarksDlibPredictor,
    draw_face_landmarks,
    create_eyes_mask,
    LEFT_EYE_IDXS,
    RIGHT_EYE_IDXS,
)


def find_equidistant_point(x1, y1, x2, y2):
    ax = x2 - x1
    ay = y2 - y1

    nx = -ay / ax
    ny = 1

    nx = nx / math.hypot(nx, ny)
    ny = ny / math.hypot(nx, ny)

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    k = math.sqrt(3) * math.dist((x1, y1), (x2, y2)) / 2

    x3 = xc + k * nx
    y3 = yc + k * ny

    return x3, y3


def find_circle(x1, y1, x2, y2, x3, y3):
    # https://stackoverflow.com/a/50974391/20380842
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2
    cd = (temp - x3 * x3 - y3 * y3) / 2
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

    if abs(det) < 1.0e-6:
        return (None, float('inf'))

    # Center of circle
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det

    radius = math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    return cx, cy, radius


class NoMLClassifier:
    def __init__(self, landmarks_predictor_path: str):
        self.landmarks_predictor = LandmarksDlibPredictor(landmarks_predictor_path)

    def __call__(self, image: np.ndarray):
        face_landmarks = self.landmarks_predictor(image)
        if face_landmarks is None:
            return None, None

        image = self.match_histogram(image)
        eye_height = self.calculate_mean_eye_height(face_landmarks)
        rule_mask, rure_mask, rarule_mask, rarure_mask = self.calculate_region_masks(image, face_landmarks, eye_height)
        rule_intensity = self.calculate_intensity_by_mask(image, rule_mask)
        rure_intensity = self.calculate_intensity_by_mask(image, rure_mask)
        rarule_intensity = self.calculate_intensity_by_mask(image, rarule_mask)
        rarure_intensity = self.calculate_intensity_by_mask(image, rarure_mask)
        mask = np.logical_or(np.logical_or(rule_mask, rure_mask), np.logical_or(rarule_mask, rarure_mask))

        left_measure = rarule_intensity - rule_intensity
        right_measure = rarure_intensity - rure_intensity
        mean_measure = 0.5 * (left_measure + right_measure)
        # predicted_label = mean_measure > self.thr

        # print(f'{left_measure=},\n '
        #       f'{right_measure=},\n '
        #       f'{rule_intensity=},\n '
        #       f'{rarule_intensity=},\n '
        #       f'{rure_intensity=}, \n'
        #       f'{rarure_intensity=}\n'
        #       f'========================')

        return mean_measure, mask

    @staticmethod
    def _match_histogram(source, tmpl_counts):
        """
          Return modified source array so that the cumulative density function of
          its values matches the cumulative density function of the template.
        """

        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / tmpl_counts.sum()

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        return interp_a_values[src_lookup].reshape(source.shape)

    def match_histogram(self, image):
        channels = []
        for channel in (0, 1, 2):
            channels.append(np.load(f'data/processed/mean_histogram_channel_{channel}.npy'))

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = self._match_histogram(image[..., channel], channels[channel])
            matched[..., channel] = matched_channel
        return matched

    def calculate_mean_eye_height(self, landmarks) -> float:
        landmark_list = landmarks.parts()

        left_top_point = ((landmark_list[37].x + landmark_list[38].x) / 2, (landmark_list[37].y + landmark_list[38].y) / 2)
        left_bot_point = ((landmark_list[41].x + landmark_list[40].x) / 2, (landmark_list[41].y + landmark_list[40].y) / 2)
        right_top_point = ((landmark_list[43].x + landmark_list[44].x) / 2, (landmark_list[43].y + landmark_list[44].y) / 2)
        right_bot_point = ((landmark_list[47].x + landmark_list[46].x) / 2, (landmark_list[47].y + landmark_list[46].y) / 2)

        left_height = math.dist(left_top_point, left_bot_point)
        right_height = math.dist(right_top_point, right_bot_point)

        mean_height = (left_height + right_height) / 2
        return mean_height

    def calculate_intensity_by_mask(self, image: np.ndarray, mask: np.ndarray) -> float:
        roi = image * mask[:, :, None]

        roi_nonzero = roi[roi.nonzero()]
        low_quantile = np.quantile(roi_nonzero, 0.25)
        high_quantile = np.quantile(roi_nonzero, 0.75)

        roi_dropped_quantile = roi_nonzero[(roi_nonzero >= low_quantile) & (roi_nonzero <= high_quantile)]

        return roi_dropped_quantile.mean()

    def calculate_region_masks(self, image: np.ndarray, face_landmarks, eye_height: float):
        left_eye_circle_mask, right_eye_circle_mask, left_eye_circle, right_eye_circle = \
            self.create_circles_mask(image, face_landmarks)
        left_pillar_mask, right_pillar_mask = self.create_eyes_pillar_masks(image, face_landmarks, eye_height)

        rule_mask = np.logical_and(left_eye_circle_mask, np.logical_not(left_pillar_mask))
        rure_mask = np.logical_and(right_eye_circle_mask, np.logical_not(right_pillar_mask))

        rarule_mask, rarure_mask = self.create_rarue_mask(
            image, left_eye_circle_mask, left_eye_circle, right_eye_circle_mask, right_eye_circle, face_landmarks, 2
        )

        return rule_mask, rure_mask, rarule_mask, rarure_mask

    def create_rarue_mask(
            self,
            image: np.ndarray,
            left_eye_circle_mask: np.ndarray,
            left_rue_circle: Tuple,
            right_eye_circle_mask: np.ndarray,
            right_rue_circle: Tuple,
            face_landmarks,
            radius_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        blank_mask = np.zeros((image.shape[0], image.shape[1]))
        rarule_mask = deepcopy(blank_mask)
        rarure_mask = deepcopy(blank_mask)

        cv2.circle(rarule_mask, (int(left_rue_circle[0]), int(left_rue_circle[1])), int(radius_factor * left_rue_circle[2]), (255, ), -1)
        cv2.circle(rarure_mask, (int(right_rue_circle[0]), int(right_rue_circle[1])),  int(radius_factor * right_rue_circle[2]), (255, ), -1)

        rarule_mask = np.logical_and(rarule_mask, np.logical_not(left_eye_circle_mask))
        rarure_mask = np.logical_and(rarure_mask, np.logical_not(right_eye_circle_mask))

        left_triangle_pts = [[face_landmarks.parts()[idx].x, face_landmarks.parts()[idx].y] for idx in [1, 38, 48]]
        left_triangle_pts = np.array(left_triangle_pts)
        left_triangle_mask = cv2.fillPoly(deepcopy(blank_mask), [left_triangle_pts], color=(255, ))

        right_triangle_pts = [[face_landmarks.parts()[idx].x, face_landmarks.parts()[idx].y] for idx in [15, 43, 54]]
        right_triangle_pts = np.array(right_triangle_pts)
        right_triangle_mask = cv2.fillPoly(deepcopy(blank_mask), [right_triangle_pts], color=(255, ))

        rarule_mask = np.logical_and(rarule_mask, left_triangle_mask)
        rarure_mask = np.logical_and(rarure_mask, right_triangle_mask)
        return rarule_mask, rarure_mask

    def create_eyes_pillar_masks(self, image: np.ndarray, face_landmarks, eye_height: float) -> Tuple[np.ndarray, np.ndarray]:
        left_pillar_mask = np.zeros((image.shape[0], image.shape[1]))
        right_pillar_mask = np.zeros((image.shape[0], image.shape[1]))

        left_pillar_pts = [[face_landmarks.parts()[idx].x, face_landmarks.parts()[idx].y] for idx in [36, 41, 40, 39]]
        left_pillar_pts.extend([[face_landmarks.parts()[36].x, 0], [face_landmarks.parts()[39].x, 0]])
        left_pillar_pts = np.array(left_pillar_pts)
        cv2.fillPoly(left_pillar_mask, [left_pillar_pts], color=(255, ))

        right_pillar_pts = [[face_landmarks.parts()[idx].x, face_landmarks.parts()[idx].y] for idx in [42, 47, 46, 45]]
        right_pillar_pts.extend([[face_landmarks.parts()[42].x, 0], [face_landmarks.parts()[45].x, 0]])
        right_pillar_pts = np.array(right_pillar_pts)
        cv2.fillPoly(right_pillar_mask, [right_pillar_pts], color=(255, ))

        kernel = np.ones((2, 2))
        iterations = max(int(0.2 * eye_height), 1)
        left_pillar_mask = cv2.dilate(left_pillar_mask, kernel, iterations=iterations)
        right_pillar_mask = cv2.dilate(right_pillar_mask, kernel, iterations=iterations)

        return left_pillar_mask, right_pillar_mask

    def create_circles_mask(self, image: np.ndarray, face_landmarks):
        plb = find_equidistant_point(
            x1=face_landmarks.parts()[36].x,
            y1=face_landmarks.parts()[36].y,
            x2=face_landmarks.parts()[39].x,
            y2=face_landmarks.parts()[39].y,
        )
        prb = find_equidistant_point(
            x1=face_landmarks.parts()[42].x,
            y1=face_landmarks.parts()[42].y,
            x2=face_landmarks.parts()[45].x,
            y2=face_landmarks.parts()[45].y,
        )

        left_eye_circle = find_circle(
            x1=face_landmarks.parts()[36].x,
            y1=face_landmarks.parts()[36].y,
            x2=face_landmarks.parts()[39].x,
            y2=face_landmarks.parts()[39].y,
            x3=plb[0],
            y3=plb[1],
        )

        right_eye_circle = find_circle(
            x1=face_landmarks.parts()[42].x,
            y1=face_landmarks.parts()[42].y,
            x2=face_landmarks.parts()[45].x,
            y2=face_landmarks.parts()[45].y,
            x3=prb[0],
            y3=prb[1],
        )

        eye_circles_mask = np.zeros((image.shape[0], image.shape[1]))
        left_eye_circle_mask = deepcopy(eye_circles_mask)
        right_eye_circle_mask = deepcopy(eye_circles_mask)

        cv2.circle(left_eye_circle_mask, (int(left_eye_circle[0]), int(left_eye_circle[1])), int(left_eye_circle[2]), (255,), -1)
        cv2.circle(right_eye_circle_mask, (int(right_eye_circle[0]), int(right_eye_circle[1])), int(right_eye_circle[2]), (255,), -1)

        return left_eye_circle_mask, right_eye_circle_mask, left_eye_circle, right_eye_circle


def test_imgs(path_to_image, path_to_predictor, thr):
    # visualizer
    clf = NoMLClassifier(path_to_predictor, thr)
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result, mask = clf(image)
    masked_image = np.logical_not(mask)[:, :, None] * image
    result_image = np.concatenate([masked_image, image], axis=1)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('1', result_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return masked_image


if __name__ == '__main__':
    path_to_predictor = '/Users/vladimir/dev/data/shape_predictor_68_face_landmarks.dat'
    path_to_folder = Path('/Users/vladimir/dev/data/dark_circles_train/data')
    for path_to_image in path_to_folder.glob('*'):
        print(path_to_image.name)
        try:
            test_imgs(path_to_image.as_posix(), path_to_predictor, 0)
        except:
            print('!!!!!!!!!')

    pass
