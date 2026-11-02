"""Integration checks against a real trained checkpoint (skipped when absent)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from ml.architectures.factory import build_model
from ml.artifacts import load_checkpoint, resolve_checkpoint
from ml.classes import CLASS_NAMES
from ml.config import DATASETS_DIR, MODELS_DIR
from ml.gradcam import GradCAM
from ml.transforms import build_eval_transform

CHECKPOINT = resolve_checkpoint(MODELS_DIR)
MANIFEST = DATASETS_DIR / "manifest.csv"

pytestmark = pytest.mark.skipif(
    CHECKPOINT is None or not MANIFEST.exists(),
    reason="No trained checkpoint or dataset available.",
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return load_checkpoint(CHECKPOINT)


@pytest.fixture(scope="module")
def model(payload):
    network = build_model(payload["arch"], len(CLASS_NAMES), pretrained=False)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network


@pytest.fixture(scope="module")
def sample_path() -> Path:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                path = DATASETS_DIR / row["processed_rel"]
                if path.exists():
                    return path
    pytest.skip("No test-split image found on disk.")


def test_checkpoint_is_real_not_synthetic(payload):
    assert payload["stage"] in {"smoke", "final"}
    assert payload["class_names"] == list(CLASS_NAMES)
    assert payload["state_dict"], "Checkpoint contains no parameters"


def test_inference_produces_valid_distribution(model, payload, sample_path):
    image = ImageOps.exif_transpose(Image.open(sample_path)).convert("RGB")
    tensor = build_eval_transform(payload["input_size"])(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].numpy()
    assert probabilities.shape == (len(CLASS_NAMES),)
    assert np.isfinite(probabilities).all()
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
    assert int(probabilities.argmax()) == int(np.argmax(logits[0].numpy()))


def test_gradcam_heatmap_is_non_degenerate(model, payload, sample_path):
    image = ImageOps.exif_transpose(Image.open(sample_path)).convert("RGB")
    tensor = build_eval_transform(payload["input_size"])(image).unsqueeze(0)
    result = GradCAM(model, payload["arch"]).generate(tensor, original=image)
    assert result.cam.ndim == 2
    assert min(result.cam.shape) > 0
    assert float(result.cam.var()) > 0, "Grad-CAM produced a flat heatmap"
    assert 0 <= result.class_index < len(CLASS_NAMES)
    assert result.overlay.size == image.size
