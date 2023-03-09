from pathlib import Path
import os

from tqdm import tqdm
import click
import cv2
import numpy as np

DATASET_NAME = 'dark_circles'
DATA_NAME = 'data'
LABELS_NAME = 'labels'


@click.command()
@click.option('--path_to_raw_data', type=str, help='Path to a folder with raw data '
                                                   '(should contain dataset/ and dark_circles/ folders)')
@click.option('--processed_dir', type=str, help='Path to a folder where processed data will be saved. '
                                                 'If folder doesnt exist, it will be created')
def main(path_to_raw_data: str, processed_dir: str):
    path_to_raw_data = Path(path_to_raw_data).expanduser()
    processed_dir = Path(processed_dir).expanduser() / DATASET_NAME

    # creating folders where processed data will be saved
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / DATA_NAME).mkdir(exist_ok=True)
    (processed_dir / LABELS_NAME).mkdir(exist_ok=True)

    for filename in tqdm(os.listdir(path_to_raw_data / 'dark_circles')):
        # processing labels
        filename = Path(filename)
        path_to_label = path_to_raw_data / 'dark_circles' / filename
        if not path_to_label.is_file():
            continue

        label = cv2.imread(path_to_label.as_posix())
        processed_label = process_label(label)

        path_to_save_label = processed_dir / LABELS_NAME / filename.with_suffix('.PNG')
        cv2.imwrite(path_to_save_label.as_posix(), processed_label)

        # processing images
        path_to_raw_image = path_to_raw_data / 'dataset' / filename
        path_to_save_image = processed_dir / DATA_NAME / filename.with_suffix('.PNG')
        image = cv2.imread(path_to_raw_image.as_posix())
        cv2.imwrite(path_to_save_image.as_posix(), image)


def process_label(label: np.ndarray) -> np.ndarray:
    try:
        unique_values = np.unique(label)
    except:
        pass
    median = np.median(unique_values)
    label = (label > median) * 255.
    label = label.astype(np.uint8)
    return label


if __name__ == '__main__':
    main()
