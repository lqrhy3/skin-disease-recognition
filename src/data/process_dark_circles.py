import shutil
from pathlib import Path
import os
from typing import Tuple

from tqdm import tqdm
import click
import cv2
import numpy as np

DATASET_NAME = 'dark_circles'
DATA_NAME = 'data'
LABELS_NAME = 'labels'


@click.command()
@click.option('--raw_data_dir', type=str, help='Path to a folder with raw data '
                                                   '(should contain dataset/ and dark_circles/ folders)')
@click.option('--processed_data_dir', type=str, help='Path to a folder where processed data will be saved. '
                                                 'If folder doesnt exist, it will be created')
def main(raw_data_dir: str, processed_data_dir: str):
    raw_data_dir = Path(raw_data_dir).expanduser()
    processed_data_dir = Path(processed_data_dir).expanduser() / DATASET_NAME
    clean_processed_dir_if_needed(processed_data_dir)

    # creating folders where processed data will be saved
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    (processed_data_dir / DATA_NAME).mkdir(exist_ok=True)
    (processed_data_dir / LABELS_NAME).mkdir(exist_ok=True)

    num_strange_files = 0
    dropped_label_names = list()

    for filename in tqdm(os.listdir(raw_data_dir / 'dark_circles'), desc='Processing data samples'):
        # processing labels
        filename = Path(filename)
        path_to_label = raw_data_dir / 'dark_circles' / filename
        if not path_to_label.is_file():
            num_strange_files += 1
            continue

        label = cv2.imread(path_to_label.as_posix())
        processed_label, drop_label = process_label(label)
        if drop_label:
            dropped_label_names.append(filename.stem)
            continue

        path_to_save_label = processed_data_dir / LABELS_NAME / filename.with_suffix('.PNG')
        cv2.imwrite(path_to_save_label.as_posix(), processed_label)

        # processing images
        path_to_raw_image = raw_data_dir / 'dataset' / filename
        path_to_save_image = processed_data_dir / DATA_NAME / filename.with_suffix('.PNG')
        image = cv2.imread(path_to_raw_image.as_posix())
        cv2.imwrite(path_to_save_image.as_posix(), image)

    print(f'Number of dropped samples: {len(dropped_label_names)}.')
    print(f'Dropped sample names: {dropped_label_names}.')


def clean_processed_dir_if_needed(processed_data_dir: os.PathLike):
    if processed_data_dir.exists():
        print(f'Remove existing dataset version? ({processed_data_dir})')
        print('y/[n]:', end=' ')
        ans = input()
        if ans.lower() in ['y', 'yes']:
            shutil.rmtree(processed_data_dir)


def process_label(label: np.ndarray) -> Tuple[np.ndarray, bool]:
    label = binarize_label(label)
    is_empty = (label == 0).all()
    label = remove_noise_from_label(label)
    is_empty_after_cleaning = (label == 0).all()
    drop_label = is_empty != is_empty_after_cleaning
    return label, drop_label


def binarize_label(label: np.ndarray) -> np.ndarray:
    unique_values = np.unique(label)
    median = np.median(unique_values)
    label = (label > median) * 255.
    label = label.astype(np.uint8)
    return label


def remove_noise_from_label(label: np.ndarray) -> np.ndarray:
    # morphological opening to remove noise
    kernel = np.ones((3, 3), dtype=label.dtype)
    label = cv2.morphologyEx(label, cv2.MORPH_OPEN, kernel)

    # morphological closing to remove small holes
    label = cv2.morphologyEx(label, cv2.MORPH_CLOSE, kernel)

    return label


if __name__ == '__main__':
    main()
