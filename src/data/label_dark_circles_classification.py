import json
import os.path
import shutil

import click
import cv2


BAKED_DATA_NAME = 'manual_dark_circles_classification'
LOG_FILE_NAME = 'log.json'


@click.command()
@click.option('--raw_data_dir', type=str, help='Path to a folder with dark_circles images.')
@click.option('--baked_data_dir', type=str, help='Path to a directory where to save prepared data'
                                                 ' (`manual_dark_circles_classification` folder will be created there).')
def main(raw_data_dir: str, baked_data_dir: str):
    raw_data_dir = os.path.expanduser(raw_data_dir)

    baked_data_dir = os.path.expanduser(baked_data_dir)
    baked_data_dir = os.path.join(baked_data_dir, BAKED_DATA_NAME)
    clean_processed_dir_if_needed(baked_data_dir)

    num_processed = 0
    path_to_log_file = os.path.join(baked_data_dir, LOG_FILE_NAME)
    if os.path.exists(path_to_log_file):
        print('Continue labelling from the last image?\n[y]/n:', end=' ')
        ans = input()
        if ans.lower() in ['y', 'yes']:
            with open(path_to_log_file, 'r') as f:
                log = json.load(f)
                num_processed = int(log['num_processed'])


    os.makedirs(baked_data_dir, exist_ok=True)

    negative_data_dir = os.path.join(baked_data_dir, '0')
    positive_data_dir = os.path.join(baked_data_dir, '1')
    os.makedirs(negative_data_dir, exist_ok=True)
    os.makedirs(positive_data_dir, exist_ok=True)

    positive_names = []
    negative_names = []
    skipped_names = []

    image_names = os.listdir(raw_data_dir)
    i = num_processed
    while i < len(image_names):
        image_name = image_names[i]
        path_to_image = os.path.join(raw_data_dir, image_name)
        image = cv2.imread(path_to_image)
        image_to_show = resize_biggest_size(image, 512)
        cv2.imshow('image', image_to_show)

        path_to_save = None
        pressed = cv2.waitKey(0)
        if pressed == ord('n'):
            path_to_save = os.path.join(negative_data_dir, image_name)
            i += 1
            print(f'Image `{image_name}` labeled as negative sample.')
            negative_names.append(image_name)
        elif pressed == ord('p'):
            path_to_save = os.path.join(positive_data_dir, image_name)
            i += 1
            print(f'Image `{image_name}` labeled as positive sample.')
            positive_names.append(image_name)
        elif pressed == ord('s'):
            i += 1
            print(f'Image `{image_name}` skipped.')
            skipped_names.append(image_name)

        elif pressed == ord('q'):
            break
        else:
            pass

        if path_to_save is not None:
            cv2.imwrite(path_to_save, image)

    log = {'num_processed': 0, 'positive_names': [], 'negative_names': [], 'skipped_names': []}
    if os.path.exists(path_to_log_file):
        with open(path_to_log_file, 'r') as f:
            log = json.load(f)

    log['num_processed'] = i
    log['positive_names'].extend(positive_names)
    log['negative_names'].extend(negative_names)
    log['skipped_names'].extend(skipped_names)

    with open(path_to_log_file, 'w') as f:
        json.dump(log, f, indent=4)


def clean_processed_dir_if_needed(processed_data_dir: str):
    if os.path.exists(processed_data_dir):
        print(f'Remove existing dataset version? ({processed_data_dir})')
        print('y/[n]:', end=' ')
        ans = input()
        if ans.lower() in ['y', 'yes']:
            shutil.rmtree(processed_data_dir)

def resize_biggest_size(image, size):
    h, w = image.shape[:2]
    biggest_size = max(h, w)
    scale = size / biggest_size
    image = cv2.resize(image, (int(scale * w), int(scale * h)))
    return image



if __name__ == '__main__':
    main()
