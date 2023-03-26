import os
import shutil
from pathlib import Path


import click

DATASET_NAME = 'dark_circles_classification'
TRAIN_DATASET_NAME = 'dark_circles_classification_train'
TEST_DATASET_NAME = 'dark_circles_classification_test'


@click.command()
@click.option('--processed_data_dir', type=str, help='Path to a folder which contain folder with dataset')
@click.option('--train_split', type=float, help='Train split percent (float)')
def main(processed_data_dir: str, train_split: float):

    processed_data_dir = Path(processed_data_dir).expanduser()
    path_to_save_train = processed_data_dir / TRAIN_DATASET_NAME
    path_to_save_test = processed_data_dir / TEST_DATASET_NAME

    # creating all required dirs
    os.makedirs(path_to_save_train, exist_ok=True)
    os.makedirs(path_to_save_test, exist_ok=True)

    for _class in ['0', '1']:
        os.makedirs(path_to_save_train / _class, exist_ok=True)
        os.makedirs(path_to_save_test / _class, exist_ok=True)

        path_to_files = processed_data_dir / DATASET_NAME / _class

        data_list = os.listdir(path_to_files)

        num_train_samples = int(len(data_list) * train_split)

        train_samples = data_list[:num_train_samples]
        test_samples = data_list[num_train_samples:]

        # copying samples from source dataset dir to splitted train/test
        for train_sample in train_samples:
            path_to_data_sample = path_to_files / train_sample
            shutil.copy2(path_to_data_sample, path_to_save_train / _class / train_sample)

        for test_sample in test_samples:
            path_to_data_sample = path_to_files / test_sample
            shutil.copy2(path_to_data_sample, path_to_save_test / _class / test_sample)


if __name__ == '__main__':
    main()
