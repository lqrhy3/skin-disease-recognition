import json
import os.path
import shutil
import subprocess
import sys

import click
import xmltodict


@click.command()
@click.option('--raw_data_dir', type=str, help='Path to a folder with raw Acne04 data.')
@click.option('--baked_data_dir', type=str, help='Path to save prepared data.')
def main(raw_data_dir: str, baked_data_dir: str):
    unzip_archives(raw_data_dir)
    create_baked_data_folders(baked_data_dir)
    move_images(raw_data_dir, baked_data_dir)
    move_and_convert_annotations(raw_data_dir, baked_data_dir)


def unzip_archives(raw_data_root: str):
    assert 'win' not in sys.platform, 'ochen\' zhal\'.'

    if not os.path.exists(os.path.join(raw_data_root, 'Classification')):
        path_to_zip = os.path.join(raw_data_root, 'Classification.tar')
        assert os.path.exists(path_to_zip)

        cmd = ['tar', '-xf', path_to_zip, '-C', raw_data_root]
        subprocess.run(cmd)

    if not os.path.exists(os.path.join(raw_data_root, 'Detection')):
        path_to_zip = os.path.join(raw_data_root, 'Detection.tar')
        assert os.path.exists(path_to_zip)

        cmd = ['tar', '-xf', path_to_zip, '-C', raw_data_root]
        subprocess.run(cmd)


def create_baked_data_folders(baked_data_dir: str):
    os.makedirs(baked_data_dir, exist_ok=True)
    os.makedirs(os.path.join(baked_data_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(baked_data_dir, 'annotations'), exist_ok=True)


def move_images(raw_data_dir: str, baked_data_dir: str):
    path_to_images_dir = os.path.join(raw_data_dir, 'Classification', 'JPEGImages')
    path_to_baked_images_dir = os.path.join(baked_data_dir, 'images')
    shutil.copytree(path_to_images_dir, path_to_baked_images_dir, dirs_exist_ok=True)


def move_and_convert_annotations(raw_data_dir: str, baked_data_dir: str):
    path_to_ann_dir = os.path.join(raw_data_dir, 'Detection', 'VOC2007', 'Annotations')
    path_to_baked_ann_dir = os.path.join(baked_data_dir, 'annotations')

    for ann_file_name in os.listdir(path_to_ann_dir):
        path_to_ann_file = os.path.join(path_to_ann_dir, ann_file_name)
        with open(path_to_ann_file, 'r') as f:
            raw_xml = f.read()

        ann_dict = xmltodict.parse(raw_xml)
        ann_dict = tweak_annotation_dict(ann_dict)

        path_to_baked_ann_file = os.path.join(
            path_to_baked_ann_dir, os.path.splitext(ann_file_name)[0] + '.json'
        )
        with open(path_to_baked_ann_file, 'w') as f:
            json.dump(ann_dict, f, indent=2)


def tweak_annotation_dict(ann_dict):
    tweaked_dict = dict()
    tweaked_dict['filename'] = ann_dict['annotation']['filename']

    bboxes_ann = ann_dict['annotation']['object']
    if isinstance(bboxes_ann, dict):  # 1 bounding box
        bboxes_ann = [bboxes_ann]

    bboxes = []
    for bbox_ann in bboxes_ann:
        bboxes.append(
            list(map(int, bbox_ann['bndbox'].values()))
        )

    tweaked_dict['bboxes'] = bboxes
    # del ann_dict['annotation']['folder']
    # if isinstance(ann_dict['annotation']['object'], dict):  # 1 bounding box
    #     ann_dict['annotation']['object'] = [ann_dict['annotation']['object']]
    #
    # ann_dict['annotation']['size'] = {k: int(v) for k, v in ann_dict['annotation']['size'].items()}
    # for obj in ann_dict['annotation']['object']:
    #     obj['bndbox'] = {k: int(v) for k, v in obj['bndbox'].items()}
    return tweaked_dict


if __name__ == '__main__':
    main()
