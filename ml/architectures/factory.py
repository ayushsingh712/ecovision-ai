"""Architecture registry shared by training and inference."""

from __future__ import annotations

import torch

from ml.architectures.custom_cnn import build_custom_cnn
from ml.architectures.mobilenet_v2 import build_mobilenet_v2

ARCHITECTURES = {
    "custom_cnn": build_custom_cnn,
    "mobilenet_v2": build_mobilenet_v2,
}


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = False,
    dropout: float | None = None,
) -> torch.nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch}")
    builder = ARCHITECTURES[arch]
    kwargs: dict = {"num_classes": num_classes}
    if dropout is not None:
        kwargs["dropout"] = dropout
    if arch == "mobilenet_v2":
        kwargs["pretrained"] = pretrained
    return builder(**kwargs)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def target_layer_name(arch: str) -> str:
    if arch == "mobilenet_v2":
        return "features[-1]"
    if arch == "custom_cnn":
        return "features[-1] (final ConvBlock)"
    raise ValueError(f"Unknown architecture: {arch}")
