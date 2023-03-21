import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]


def unnormalize_imagenet(image: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=image.device)  # [3,]
    std = torch.tensor(IMAGENET_STD, device=image.device)  # [3,]

    if image.ndim == 4:
        assert image.shape[1] == 3
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif image.ndim == 3:
        assert image.shape[0] == 3
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

    image = image * std + mean
    return image

