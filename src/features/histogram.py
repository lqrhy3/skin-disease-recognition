import glob

import cv2
import click
import numpy as np


@click.command
@click.option("--data_dir", type=str, help="Path to a classification data")
def calculate_histogram(data_dir: str):
    all_images = glob.glob(f'{data_dir}/*/*')
    histograms_b = []
    histograms_g = []
    histograms_r = []
    sizes = []
    for image_path in all_images:
        image = cv2.imread(image_path)
        sizes.append(image[:, :, 0].size)

        for channel, array in zip((0, 1, 2), (histograms_b, histograms_g, histograms_r)):
            histogram, edges = np.histogram(image[:, :, channel], bins=256, range=(0, 256))
            array.append(histogram)

    for channel, array in zip((0, 1, 2), (histograms_b, histograms_g, histograms_r)):
        mean_histogram = np.mean(array, axis=0)
        np.save(f'data/processed/mean_histogram_channel_{channel}.npy', mean_histogram)

    print(np.mean(sizes, axis=0))


if __name__ == '__main__':
    calculate_histogram()
