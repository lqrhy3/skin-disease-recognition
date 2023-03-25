import os.path

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, accuracy_score
from tqdm import tqdm

from src.models.noml import NoMLClassifier


NEG_LABEL_NAME = '0'
POS_LABEL_NAME = '1'
LABEL_NAMES = [NEG_LABEL_NAME, POS_LABEL_NAME]
LABEL2NAME = {0: NEG_LABEL_NAME, 1: POS_LABEL_NAME}


def f1_score(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall)

@click.command()
@click.option('--data_dir', type=str, help='Path to a classification data.')
@click.option('--path_to_landmarks_predictor', type=str, help='Path to dlib face_landmarks_predictor.')
def main(data_dir: str, path_to_landmarks_predictor: str):
    model = NoMLClassifier(
        landmarks_predictor_path=path_to_landmarks_predictor,
    )

    scores_pred = []
    labels = []
    skipped_image_names = []
    for class_name in LABEL_NAMES:
        class_data_dir = os.path.join(data_dir, class_name)
        for image_name in tqdm(os.listdir(class_data_dir), desc=f'Class {class_name}'):
            path_to_image = os.path.join(class_data_dir, image_name)
            image = cv2.imread(path_to_image)
            score_pred, _ = model(image)
            if score_pred is None:
                skipped_image_names.append((class_name, image_name))
                continue

            scores_pred.append(score_pred)
            labels.append(class_name)

    precision, recall, thresholds = precision_recall_curve(labels, scores_pred, pos_label='1')
    f1 = np.array([f1_score(p, r) for p, r in zip(precision, recall)])

    arg_max = np.argmax(f1)
    best_f1 = f1[arg_max]
    best_thr = thresholds[arg_max]
    labels_pred = np.vectorize(LABEL2NAME.get)(np.array(scores_pred) > best_thr)
    acc = accuracy_score(labels, labels_pred)

    PrecisionRecallDisplay(precision, recall, estimator_name='NoMLClassifier', pos_label='1').plot()
    plt.title('Precision-Recall curve')
    plt.show()

    print(f'Best f1-score: {best_f1}; threshold: {best_thr}.')
    print(f'Best accuracy: {acc}')
    print(f'Skipped images: {skipped_image_names}')


if __name__ == '__main__':
    main('/home/lqrhy3/PycharmProjects/skin-deseases-detection-project/data/processed/manual_dark_circles_classification')
    # /home/lqrhy3/PycharmProjects/skin-deseases-detection-project/data/processed/manual_dark_circles_classification