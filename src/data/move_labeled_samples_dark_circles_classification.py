import json
from pathlib import Path
import shutil
import os
from typing import Any, List

import click


@click.command()
@click.option('--raw_data_dir', type=str, help='Path to a folder with dark_circles images.')
@click.option('--dst_dir', type=str, help='Path to destination folder where will be created'
                                           'folders "0" and "1"')
@click.option('--json_path', type=str, help='Paht to json file with labels.'
                                             'Results of `label_dark_circles_classification.py` script')
def main(raw_data_dir: str, dst_dir: str, json_path: str):
    raw_data_dir = Path(raw_data_dir).expanduser()
    dst_dir = Path(dst_dir).expanduser()
    json_path = Path(json_path).expanduser()

    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
    else:
        clean_processed_dir_if_needed(dst_dir / '0')
        clean_processed_dir_if_needed(dst_dir / '1')

    if not (dst_dir / '0').exists():
        (dst_dir / '0').mkdir()

    if not (dst_dir / '1').exists():
        (dst_dir / '1').mkdir()

    labels = read_json(json_path)
    move_files(labels['negative_names'], raw_data_dir, dst_dir / '0')
    move_files(labels['positive_names'], raw_data_dir, dst_dir / '1')


def move_files(filenames: List, src_dir: Path, dst_dir: Path):
    for filename in filenames:
        shutil.copy2(src_dir / filename, dst_dir / filename)


def read_json(filename: Path) -> Any:
    with open(filename, 'r') as file:
        result = json.load(file)

    return result


def clean_processed_dir_if_needed(processed_data_dir: Path):
    if processed_data_dir.exists():
        print(f'Remove existing dataset version? ({processed_data_dir})')
        print('y/[n]:', end=' ')
        ans = input()
        if ans.lower() in ['y', 'yes']:
            shutil.rmtree(processed_data_dir)


if __name__ == '__main__':
    main()
