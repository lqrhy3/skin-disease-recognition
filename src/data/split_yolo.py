import os.path

import click
from random import shuffle

TRAIN_RATIO = 0.8


@click.command()
@click.option('--yolo_data_dir', type=str, help='Path to a folder with yolo data.')
def main(yolo_data_dir: str):
    yolo_data_dir = os.path.expanduser(yolo_data_dir)

    path_to_images = os.path.join(yolo_data_dir, 'images')
    list_images = create_full_path(path_to_images, os.listdir(path_to_images))
    shuffle(list_images)

    train_images_num = int(len(list_images) * TRAIN_RATIO)

    train_images_list = list_images[:train_images_num]
    val_images_list = list_images[train_images_num:]

    create_txt_file_list(train_images_list, 'train', yolo_data_dir)
    create_txt_file_list(val_images_list, 'val', yolo_data_dir)


def create_full_path(path_to_data: str, list_files: list):
    return list(map(lambda x: os.path.join(path_to_data, x), list_files))


def create_txt_file_list(list_files: list, split: str, txt_path: str):
    txt_path = os.path.join(txt_path, f'{split}.txt')

    with open(txt_path, 'w') as txt_file:
        for file_path in list_files:
            txt_file.write(f"{file_path}\n")


if __name__ == '__main__':
    main()
