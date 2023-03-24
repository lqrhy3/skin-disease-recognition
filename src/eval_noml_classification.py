import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

from src.models.noml import NoMLClassifier


CLASS_NAMES = ['0', '1']


def f1_score(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall)


def main(data_dir: str):
    model = NoMLClassifier(
        landmarks_predictor_path='/home/lqrhy3/PycharmProjects/skin-deseases-detection-project/data/shape_predictor_68_face_landmarks.dat',
        thr=None
    )

    scores_pred = []
    labels = []
    skipped_image_names = []
    for class_name in CLASS_NAMES:
        class_data_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_data_dir):
            path_to_image = os.path.join(class_data_dir, image_name)
            image = cv2.imread(path_to_image)
            score_pred = model(image)
            if score_pred is None:
                skipped_image_names.append((class_name, image_name))
                continue

            scores_pred.append(score_pred)
            labels.append(class_name)

    precision, recall, thresholds = precision_recall_curve(labels, scores_pred, pos_label='1')
    f1 = np.array([f1_score(p, r) for p, r in zip(precision, recall)])
    arg_max = np.argmax(f1)

    PrecisionRecallDisplay(precision, recall, estimator_name='NoMLClassifier', pos_label='1').plot()
    plt.title('Precision-Recall curve')
    plt.show()

    print(f'Best f1-score: {f1[arg_max]}; threshold: {thresholds[arg_max]}.')
    print(f'Skipped images: {skipped_image_names}')


if __name__ == '__main__':
    main('/data/processed/manual_dark_circles_classification_bac')