import os.path

import click
import fiftyone as fo


@click.command()
@click.option('--data_dir', type=str, help='Path to a folder with data in VOC format.')
def main(data_dir: str):
    data_dir = os.path.expanduser(data_dir)

    dataset_name = os.path.split(data_dir)[-1]
    dataset = fo.Dataset.from_dir(
        dataset_dir=data_dir,
        dataset_type=fo.types.VOCDetectionDataset,
        name=dataset_name
    )

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    main()
