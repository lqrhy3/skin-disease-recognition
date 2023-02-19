import click
import fiftyone as fo


@click.command()
@click.option('--data_path', type=str, help='Path to folder with images.')
@click.option('--labels_path', type=str, help='Path to folder with xml files.')
@click.option('--dataset_name', type=str, help='Name for displaying dataset', default='acne04_detection')
def main(data_path: str, labels_path: str, dataset_name: str):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.VOCDetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        name=dataset_name
    )

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    main()
