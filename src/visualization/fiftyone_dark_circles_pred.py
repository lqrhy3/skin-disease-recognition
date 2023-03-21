import os
from pathlib import Path

import click
import fiftyone as fo


def load_dataset(data_dir: Path) -> fo.Dataset:
    dataset_name = data_dir.name

    dataset: fo.Dataset = fo.Dataset.from_dir(
        name=dataset_name,
        dataset_dir=data_dir,
        dataset_type=fo.types.ImageSegmentationDirectory,
    )

    return dataset


def add_predicted_masks(dataset: fo.Dataset, preds_dir: Path) -> fo.Dataset:
    for sample in dataset:
        path_to_predicted_mask = preds_dir / sample.filename

        if path_to_predicted_mask.exists():
            sample['prediction'] = fo.Segmentation(mask_path=path_to_predicted_mask.as_posix())
            sample.save()

    return dataset


@click.command()
@click.option('--data_dir', type=str, help='Path to a folder with dark_circle dataset '
                                           '(should contain data/ and labels/ folders)')
@click.option('--preds_dir', type=str, help='Path to a folder with predicted masks '
                                            '(masks should be named same as source images)')
def main(data_dir: str, preds_dir: str):
    data_dir = Path(data_dir).expanduser()
    preds_dir = Path(preds_dir).expanduser()

    dataset = load_dataset(data_dir)
    dataset = add_predicted_masks(dataset, preds_dir)

    session = fo.launch_app(dataset, remote=True)
    session.wait()


if __name__ == '__main__':
    main()
