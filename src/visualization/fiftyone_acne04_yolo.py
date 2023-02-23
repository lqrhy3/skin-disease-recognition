import os

import click
import fiftyone as fo


@click.command()
@click.option('--data_dir', type=str, help='Path to a folder with data in VOC format.')
def main(data_dir: str):
    dataset_name = os.path.split(data_dir)[-1]

    # The splits to load
    splits = ["train", "val"]
    dataset = fo.Dataset(dataset_name)
    for split in splits:
        dataset.add_dir(
            dataset_dir=data_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            tags=split,
        )

    session = fo.launch_app(dataset, remote=True)
    session.wait()


if __name__ == '__main__':
    main()
