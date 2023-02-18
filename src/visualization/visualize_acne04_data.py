import json
import os

import click
import cv2
import numpy as np

from src.visualization.utils import draw_bounding_boxes


@click.command()
@click.option('--data_dir', type=str, help='Path to folder with prepared acne04 data.')
@click.option('--resize_images', type=bool, default=True)
def main(data_dir: str, resize_images: bool):
    annotations_dir = os.path.join(data_dir, 'annotations')
    ann_file_names = os.listdir(annotations_dir)
    ann_i = 0
    images_dir = os.path.join(data_dir, 'images')

    enable_draw_bboxes = True
    print_instructions()
    cv2.startWindowThread()
    while True:
        path_to_ann_file = os.path.join(annotations_dir, ann_file_names[ann_i])
        image_name, bboxes = read_annotation(path_to_ann_file)
        path_to_image = os.path.join(images_dir, image_name)

        image = cv2.imread(path_to_image)
        if enable_draw_bboxes:
            image = draw_bounding_boxes(image, bboxes)
        if resize_images:
            image = resize_min_side(image, 512)
        cv2.imshow('image', image)

        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord('a'):
            ann_i = max(0, ann_i - 1)
        elif pressed_key == ord('d'):
            ann_i = min(len(ann_file_names), ann_i + 1)
        elif pressed_key == ord('e'):
            enable_draw_bboxes = not enable_draw_bboxes
        elif pressed_key == ord('q'):
            break


def print_instructions():
    print('Press `a` and `d` to switch to the previous and next image respectively.')
    print('Press `e` to enable/disable annotations.')
    print('Press `q` to exit.')


def read_annotation(path_to_ann_file: str):
    with open(path_to_ann_file, 'r') as f:
        ann_dict = json.load(f)

    return ann_dict['filename'], ann_dict['bboxes']


def resize_min_side(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    min_side = min(h, w)
    scale = size / min_side

    image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


if __name__ == '__main__':
    main()
