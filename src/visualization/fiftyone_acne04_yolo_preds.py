import os

import cv2
import torch
import click
import numpy as np
import fiftyone as fo
from tqdm import tqdm

from src.models.yolov7.utils.datasets import letterbox
from src.models.yolov7.models.experimental import attempt_load
from src.models.yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh


def load_dataset(data_dir):
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

    return dataset


def load_image(file_path, img_size, stride):
    raw_img = cv2.imread(file_path)  # BGR
    assert raw_img is not None, 'Image Not Found ' + file_path

    preprocessed_img = letterbox(raw_img, img_size, stride=stride)[0]

    # Convert
    preprocessed_img = preprocessed_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    preprocessed_img = np.ascontiguousarray(preprocessed_img, dtype=np.float32)

    preprocessed_img /= 255.0

    return preprocessed_img, raw_img


def process_predictions(pred, img, raw_image, conf_thres, iou_thres):

    pred = non_max_suppression(pred, conf_thres, iou_thres)

    detections = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        gn = torch.tensor(raw_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_image.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                xywh = [xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[2], xywh[3]]
                detections.append(
                    fo.Detection(
                        label='fore',
                        bounding_box=xywh,
                        confidence=conf
                    )
                )

    return detections


@click.command()
@click.option('--weights', type=str, help='Path to models weights')
@click.option('--source', type=str, help='Path to dataset directory')
@click.option('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
@click.option('--img-size', type=int, default=640, help='inference size (pixels)')
@click.option('--conf_thres', type=float, default=0.25, help='object confidence threshold')
@click.option('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
def main(weights: str, source: str, device: str, img_size: int, conf_thres, iou_thres):
    dataset = load_dataset(source)

    model = attempt_load(weights, map_location=device)
    model.eval()

    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    for sample in tqdm(dataset):
        file_path = sample['filepath']
        img, raw_image = load_image(file_path, img_size, stride)

        img = torch.from_numpy(img).to(device)

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]

        detections = process_predictions(pred, img, raw_image, conf_thres, iou_thres)

        sample[model.yaml_file.split('.')[0]] = fo.Detections(detections=detections)
        sample.save()

    session = fo.launch_app(dataset, remote=True)
    session.wait()


if __name__ == "__main__":
    main()
