"""Dependency-light Grad-CAM implementation used by the CLI and backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


@dataclass
class GradCAMResult:
    class_index: int
    cam: np.ndarray
    heatmap: Image.Image
    overlay: Image.Image
    original: Image.Image


def get_target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    if arch == "mobilenet_v2":
        return model.features[-1]
    if arch == "custom_cnn":
        return model.features[-1]
    raise ValueError(f"Unsupported architecture for Grad-CAM: {arch}")


def _normalise_cam(cam: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        cam = torch.relu(cam)
        minimum = cam.min()
        maximum = cam.max()
        if float(maximum - minimum) < 1e-12:
            return torch.zeros_like(cam)
        return (cam - minimum) / (maximum - minimum)


class GradCAM:
    """Single-image Grad-CAM generator. Hooks are always removed."""

    def __init__(self, model: torch.nn.Module, arch: str) -> None:
        self.model = model
        self.arch = arch
        self.target_layer = get_target_layer(model, arch)
        self._lock = threading.Lock()

    def generate(
        self,
        tensor: torch.Tensor,
        class_index: int | None = None,
        original: Image.Image | None = None,
    ) -> GradCAMResult:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects exactly one image.")

        activations: list[torch.Tensor] = []
        handle: Any = None

        def forward_hook(_module, _inputs, output):
            activations.append(output)
            output.retain_grad()

        with self._lock:
            was_training = self.model.training
            self.model.eval()
            handle = self.target_layer.register_forward_hook(forward_hook)
            try:
                with torch.enable_grad():
                    work = tensor.detach().clone().requires_grad_(True)
                    self.model.zero_grad(set_to_none=True)
                    logits = self.model(work)
                    if class_index is None:
                        class_index = int(logits.argmax(dim=1).item())
                    score = logits[0, class_index]
                    score.backward()
                    if not activations:
                        raise RuntimeError("Grad-CAM target layer was not executed.")
                    acts = activations[0][0]
                    grads = activations[0].grad
                    if grads is None:
                        raise RuntimeError("Grad-CAM gradients were not captured.")
                    grads = grads[0]
                    weights = grads.mean(dim=(1, 2))
                    cam = (weights[:, None, None] * acts).sum(dim=0)
                    cam = _normalise_cam(cam).detach().cpu().numpy()
            finally:
                if handle is not None:
                    handle.remove()
                self.model.zero_grad(set_to_none=True)
                self.model.train(was_training)

        cam = np.asarray(cam, dtype=np.float32)
        heatmap_rgb = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
        base = original.convert("RGB") if original is not None else None
        if base is None:
            size = int(tensor.shape[-1])
            base = Image.new("RGB", (size, size), (255, 255, 255))
        else:
            size = base.width
        heatmap = Image.fromarray(heatmap_rgb, mode="RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        )
        overlay = Image.blend(base, heatmap, alpha=0.45)
        return GradCAMResult(
            class_index=int(class_index),
            cam=cam,
            heatmap=heatmap,
            overlay=overlay,
            original=base,
        )


def build_three_panel(result: GradCAMResult, class_name: str) -> np.ndarray:
    """Create a side-by-side original / heatmap / overlay RGB array."""
    width, height = result.original.size
    canvas = np.full((height, width * 3 + 24, 3), 255, dtype=np.uint8)
    canvas[:, :width] = np.asarray(result.original)
    canvas[:, width + 12 : width * 2 + 12] = np.asarray(result.heatmap)
    canvas[:, width * 2 + 24 :] = np.asarray(result.overlay)
    return canvas
