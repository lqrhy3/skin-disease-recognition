import os.path
import shutil
import subprocess
import sys

import click

BAKED_DATA_NAME = 'dark_circles'


@click.command()
@click.option('--raw_data_dir', type=str, help='Path to a folder with raw Acne04 data.')
@click.option('--baked_data_dir', type=str, help='Path to a directory where to save prepared data'
                                                 ' (`acne04_voc` folder will be created there).')
def main(raw_data_dir: str, baked_data_dir: str):
    raw_data_dir = os.path.expanduser(raw_data_dir)
    baked_data_dir = os.path.expanduser(baked_data_dir)

    baked_data_dir = os.path.join(baked_data_dir, BAKED_DATA_NAME)
    unzip_archives(raw_data_dir)
    create_baked_data_folders(baked_data_dir)
    move_images(raw_data_dir, baked_data_dir)
    move_annotations(raw_data_dir, baked_data_dir)


def unzip_archives(raw_data_root: str):
    assert 'win' not in sys.platform, 'ochen\' zhal\'.'

    if not os.path.exists(os.path.join(raw_data_root, 'dark_circles')):
        path_to_zip = os.path.join(raw_data_root, 'dark_circles.zip')
        assert os.path.exists(path_to_zip)

        cmd = ['tar', '-xf', path_to_zip, '-C', raw_data_root]
        subprocess.run(cmd)

    if not os.path.exists(os.path.join(raw_data_root, 'dataset')):
        path_to_zip = os.path.join(raw_data_root, 'dataset.zip')
        assert os.path.exists(path_to_zip)

        cmd = ['tar', '-xf', path_to_zip, '-C', raw_data_root]
        subprocess.run(cmd)


def create_baked_data_folders(baked_data_dir: str):
    os.makedirs(baked_data_dir, exist_ok=True)
    os.makedirs(os.path.join(baked_data_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(baked_data_dir, 'labels'), exist_ok=True)


def move_images(raw_data_dir: str, baked_data_dir: str):
    path_to_images_dir = os.path.join(raw_data_dir, 'dataset')
    path_to_baked_images_dir = os.path.join(baked_data_dir, 'data')
    shutil.copytree(path_to_images_dir, path_to_baked_images_dir, dirs_exist_ok=True)


def move_annotations(raw_data_dir: str, baked_data_dir: str):
    path_to_anns_dir = os.path.join(raw_data_dir, 'dark_circles', 'dark_circles')
    path_to_baked_anns_dir = os.path.join(baked_data_dir, 'labels')
    shutil.copytree(path_to_anns_dir, path_to_baked_anns_dir, dirs_exist_ok=True)


if __name__ == '__main__':
    main()
