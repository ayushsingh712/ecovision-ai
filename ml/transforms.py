"""Shared image transforms for training and serving."""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from ml.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def build_train_transform(input_size: int = INPUT_SIZE) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                input_size, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.RandomErasing(p=0.1, scale=(0.02, 0.12), value=0),
        ]
    )


def build_eval_transform(input_size: int = INPUT_SIZE) -> v2.Compose:
    resize_size = int(round(input_size * 256 / 224))
    return v2.Compose(
        [
            v2.Resize(resize_size, antialias=True),
            v2.CenterCrop(input_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
