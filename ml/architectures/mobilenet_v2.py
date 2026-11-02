"""MobileNetV2 transfer-learning model."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, count: int = 3) -> None:
    freeze_backbone(model)
    for block in model.features[-count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def make_parameter_groups(model: nn.Module, lr_head: float, lr_backbone: float):
    return [
        {
            "params": [p for p in model.features.parameters() if p.requires_grad],
            "lr": lr_backbone,
        },
        {
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": lr_head,
        },
    ]
