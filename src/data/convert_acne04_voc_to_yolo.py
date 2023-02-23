import os.path
import yaml

import click
import shutil
import xmltodict

BAKED_DATA_NAME = 'acne04_yolo'
CLASS_ID_MAP = {'fore': 0}
YOLO_IMAGES_DIR = 'images'


@click.command()
@click.option('--voc_data_dir', type=str, help='Path to a folder with voc data.')
@click.option('--baked_data_dir', type=str, help='Path to a directory where to save prepared data'
                                                 ' (`acne04_yolo` folder will be created there).')
def main(voc_data_dir: str, baked_data_dir: str):
    voc_data_dir = os.path.expanduser(voc_data_dir)
    baked_data_dir = os.path.expanduser(baked_data_dir)

    baked_data_dir = os.path.join(baked_data_dir, BAKED_DATA_NAME)
    move_and_convert_annotation(voc_data_dir, baked_data_dir)
    move_images(voc_data_dir, baked_data_dir)
    create_dataset_yaml(baked_data_dir)


def create_dataset_yaml(baked_data):

    data = {
        'nc': len(CLASS_ID_MAP),
        'names': list(CLASS_ID_MAP.keys()),
        'val': os.path.join(baked_data, 'val.txt'),
        'train': os.path.join(baked_data, 'train.txt'),
        'path': baked_data
    }

    with open(os.path.join(baked_data, 'dataset.yaml'), 'w') as f:
        yaml.dump(data, f)


def move_and_convert_annotation(data_dir: str, yolo_dir: str):
    os.makedirs(os.path.join(yolo_dir, 'labels'), exist_ok=True)

    path_to_annotations = os.path.join(data_dir, 'labels')

    for annotation_file_name in os.listdir(path_to_annotations):
        path_to_annotation_file = os.path.join(path_to_annotations, annotation_file_name)

        with open(path_to_annotation_file, 'r') as f:
            raw_xml = f.read()

        annotation_dict = xmltodict.parse(raw_xml)

        yolo_annotation_str = get_yolo_str_annotation(annotation_dict=annotation_dict, class_id_map=CLASS_ID_MAP)

        txt_label_filename = annotation_file_name.replace('xml', 'txt')
        txt_filename_path = os.path.join(yolo_dir, 'labels', txt_label_filename)

        create_txt(txt_filename_path, yolo_annotation_str)


def move_images(data_dir: str, yolo_dir: str):
    path_to_images_dir = os.path.join(data_dir, 'data')
    path_to_yolo_images_dir = os.path.join(yolo_dir, YOLO_IMAGES_DIR)
    shutil.copytree(path_to_images_dir, path_to_yolo_images_dir, dirs_exist_ok=True)


def get_yolo_str_annotation(annotation_dict: dict, class_id_map: dict):

    image_width, image_height = int(annotation_dict['annotation']['size']['width']),\
                                int(annotation_dict['annotation']['size']['height'])

    annotations = annotation_dict['annotation']['object']
    if isinstance(annotations, dict):
        annotations = [annotations]

    yolo_annotation_str = ''

    for annotation in annotations:

        bboxes = annotation['bndbox']
        x_min, x_max, y_min, y_max = int(bboxes['xmin']), int(bboxes['xmax']), int(bboxes['ymin']), int(bboxes['ymax'])

        scaled_x_min, scaled_y_min = x_min / image_width, y_min / image_height
        scaled_width, scaled_height = (x_max - x_min) / image_width, (y_max - y_min) / image_height

        scaled_x_center, scaled_y_center = scaled_x_min + scaled_width / 2, scaled_y_min + scaled_height / 2

        class_id = class_id_map[annotation['name']]
        yolo_annotation_str += f'{class_id} {scaled_x_center} {scaled_y_center} {scaled_width} {scaled_height}\n'

    return yolo_annotation_str


def create_txt(filename: str, content: str):
    with open(filename, 'w') as txt_file:
        txt_file.write(content)


if __name__ == '__main__':
    main()
