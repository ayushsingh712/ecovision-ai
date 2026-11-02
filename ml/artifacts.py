"""Checkpoint and manifest serialization shared by training and serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ml.classes import CLASS_NAMES, validate_class_names
from ml.config import INPUT_SIZE, MODEL_VERSION
from ml.utils import utc_now_iso, write_json

FINAL_CHECKPOINTS = ("mobilenet_v2_best.pt", "custom_cnn_best.pt")
SMOKE_CHECKPOINTS = ("smoke/mobilenet_v2_smoke.pt", "smoke/custom_cnn_smoke.pt")


def build_checkpoint(
    model: torch.nn.Module,
    arch: str,
    stage: str,
    history: list[dict[str, Any]],
    split_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_dict = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    return {
        "format_version": 1,
        "arch": arch,
        "class_names": list(CLASS_NAMES),
        "input_size": INPUT_SIZE,
        "stage": stage,
        "model_version": MODEL_VERSION,
        "created_at": utc_now_iso(),
        "torch_version": str(torch.__version__),
        "split_hash": split_hash,
        "history": history,
        "state_dict": state_dict,
        "extra": extra or {},
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    # Older checkpoints stored torch.__version__ as a TorchVersion instance.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"arch", "class_names", "state_dict", "input_size"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    validate_class_names(payload["class_names"])
    return payload


def write_manifest(
    models_dir: str | Path,
    active: str,
    stage: str,
    arch: str,
    metrics_path: str | None = None,
    validation_macro_f1: float | None = None,
) -> Path:
    target = Path(models_dir) / "model_manifest.json"
    payload = {
        "active": active,
        "stage": stage,
        "arch": arch,
        "model_version": MODEL_VERSION,
        "class_names": list(CLASS_NAMES),
        "metrics_path": metrics_path,
        "validation_macro_f1": validation_macro_f1,
        "created_at": utc_now_iso(),
    }
    return write_json(target, payload)


def read_manifest(models_dir: str | Path) -> dict[str, Any]:
    target = Path(models_dir) / "model_manifest.json"
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def resolve_checkpoint(
    models_dir: str | Path, explicit: str | Path | None = None
) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        return None

    root = Path(models_dir)
    manifest = read_manifest(root)
    active = manifest.get("active")
    if active:
        candidate = root / active
        if candidate.exists():
            return candidate

    for name in FINAL_CHECKPOINTS:
        candidate = root / name
        if candidate.exists():
            return candidate
    for name in SMOKE_CHECKPOINTS:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None
