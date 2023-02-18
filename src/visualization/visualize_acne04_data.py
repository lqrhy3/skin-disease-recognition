import os

import click
import cv2
import numpy as np

from src.utils import read_json
from src.visualization.utils import draw_bounding_boxes


@click.command()
@click.option('--data_dir', type=str, help='Path to folder with prepared acne04 data.')
@click.option('--resize_images', type=bool, default=True)
def main(data_dir: str, resize_images: bool):
    annotations = read_json(os.path.join(data_dir, 'annotations.json'))
    images_dir = os.path.join(data_dir, 'images')
    i = 0

    enable_draw_bboxes = True
    print_instructions()
    cv2.startWindowThread()
    while True:
        image_name = annotations[i]['filename']
        bboxes = annotations[i]['bboxes']
        path_to_image = os.path.join(images_dir, image_name)

        image = cv2.imread(path_to_image)
        if enable_draw_bboxes:
            image = draw_bounding_boxes(image, bboxes)
        if resize_images:
            image = resize_min_side(image, 512)
        cv2.imshow(image_name, image)

        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord('a'):
            i = max(0, i - 1)
        elif pressed_key == ord('d'):
            i = min(len(annotations), i + 1)
        elif pressed_key == ord('e'):
            enable_draw_bboxes = not enable_draw_bboxes
        elif pressed_key == ord('q'):
            break

        cv2.destroyWindow(image_name)


def print_instructions():
    print('Press `a` and `d` to switch to the previous and next image respectively.')
    print('Press `e` to enable/disable annotations.')
    print('Press `q` to exit.')


def read_annotation(path_to_ann_file: str):
    ann_dict = read_json(path_to_ann_file)
    return ann_dict['filename'], ann_dict['bboxes']


def resize_min_side(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    min_side = min(h, w)
    scale = size / min_side

    image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


if __name__ == '__main__':
    main()
