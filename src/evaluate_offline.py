import hydra
import pyrootutils
import torch
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.utils import read_json


@hydra.main(version_base='1.3', config_path='../configs/', config_name='evaluate_offline.yaml')
def main(cfg: DictConfig):
    pred_annotations = read_json(cfg.pred_ann_file)
    target_annotations = read_json(cfg.target_ann_file)

    if cfg.check_correspondence:
        assert is_in_correspondence(pred_annotations, target_annotations)

    pred = make_bboxes_batch(pred_annotations)
    target = make_bboxes_batch(target_annotations)

    metrics = hydra.utils.instantiate(cfg.metrics)
    for metric in metrics:
        metric_val = metric(pred, target)
        print(f'{metric.__class__.__name__}: {metric_val}')


def is_in_correspondence(pred_annotations, target_annotations):
    if len(pred_annotations) != len(target_annotations):
        return False

    for pred_ann, target_ann in zip(pred_annotations, target_annotations):
        if pred_ann['filename'] != target_ann['filename']:
            return False

    return True


def make_bboxes_batch(annotations: list):
    # TODO: bruh
    batch = []
    for ann in annotations:
        sample = dict()
        sample['boxes'] = torch.FloatTensor(ann['bboxes'])
        sample['scores'] = torch.ones((len(ann['bboxes'])), dtype=torch.float)
        sample['labels'] = torch.zeros((len(ann['bboxes'])), dtype=torch.int)
        batch.append(sample)

    return batch


if __name__ == '__main__':
    main()
