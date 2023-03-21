import os
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import click
import numpy as np
from tqdm import tqdm

from src.data.process_dark_circles import clean_processed_dir_if_needed, process_label
DATASET_NAME = 'labelme_dark_circles'
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

    dropped_label_names = list()

    for filename in tqdm(os.listdir(raw_data_dir), desc='Processing data samples'):

        # processing labels
        filename = Path(filename)
        if 'npy' not in str(filename):
            continue

        path_to_label = raw_data_dir / filename
        label = np.load(path_to_label.as_posix())

        path_to_save_label = processed_data_dir / LABELS_NAME / filename.with_suffix('.PNG')
        cv2.imwrite(path_to_save_label.as_posix(), label)

        # processing images
        filename_without_npy = Path(str(filename).replace('.npy', ''))
        path_to_raw_image = raw_data_dir / '..' / 'dataset' / filename_without_npy
        path_to_save_image = processed_data_dir / DATA_NAME / filename_without_npy.with_suffix('.PNG')
        image = cv2.imread(path_to_raw_image.as_posix())

        try:
            cv2.imwrite(path_to_save_image.as_posix(), image)
        except cv2.error:
            print("Image not found: ", path_to_save_image)

    print(f'Number of dropped samples: {len(dropped_label_names)}.')
    print(f'Dropped sample names: {dropped_label_names}.')


if __name__ == '__main__':
    main()