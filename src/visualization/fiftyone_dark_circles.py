import os

import click
import fiftyone as fo


@click.command()
@click.option('--data_dir', type=str, help='Path to a folder with dark_circle dataset '
                                           '(should contain data/ and labels/ folders)')
def main(data_dir: str):
    dataset_name = os.path.split(data_dir)[-1]

    dataset = fo.Dataset.from_dir(
        name=dataset_name,
        dataset_dir=data_dir,
        dataset_type=fo.types.ImageSegmentationDirectory,
    )

    session = fo.launch_app(dataset, remote=True)
    session.wait()


if __name__ == '__main__':
    main()
