import os
import shutil
from pathlib import Path


import click

DATASET_NAME = 'dark_circles'
TRAIN_DATASET_NAME = 'dark_circles_train'
TEST_DATASET_NAME = 'dark_circles_test'


@click.command()
@click.option('--processed_data_dir', type=str, help='Path to a folder which contain folder with dataset')
@click.option('--train_split', type=float, help='Train split percent (float)')
def main(processed_data_dir: str, train_split: float):
    processed_data_dir = Path(processed_data_dir).expanduser()
    path_to_save_train = processed_data_dir / TRAIN_DATASET_NAME
    path_to_save_test = processed_data_dir / TEST_DATASET_NAME

    # creating all required dirs
    os.makedirs(path_to_save_train, exist_ok=True)
    os.makedirs(path_to_save_train / 'data', exist_ok=True)
    os.makedirs(path_to_save_train / 'labels', exist_ok=True)

    os.makedirs(path_to_save_test, exist_ok=True)
    os.makedirs(path_to_save_test / 'data', exist_ok=True)
    os.makedirs(path_to_save_test / 'labels', exist_ok=True)

    path_to_data = processed_data_dir / DATASET_NAME / 'data'
    path_to_labels = processed_data_dir / DATASET_NAME / 'labels'
    data_list = os.listdir(path_to_data)

    num_train_samples = int(len(data_list) * train_split)
    train_samples = data_list[:num_train_samples]
    test_samples = data_list[num_train_samples:]

    # copying samples from source dataset dir to splitted train/test
    for train_sample in train_samples:
        path_to_data_sample = path_to_data / train_sample
        path_to_label_sample = path_to_labels / train_sample
        shutil.copy2(path_to_data_sample, path_to_save_train / 'data' / train_sample)
        shutil.copy2(path_to_label_sample, path_to_save_train / 'labels' / train_sample)

    for test_sample in test_samples:
        path_to_data_sample = path_to_data / test_sample
        path_to_label_sample = path_to_labels / test_sample
        shutil.copy2(path_to_data_sample, path_to_save_test / 'data' / test_sample)
        shutil.copy2(path_to_label_sample, path_to_save_test / 'labels' / test_sample)


if __name__ == '__main__':
    main()
