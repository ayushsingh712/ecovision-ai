"""Model loading, inference, Grad-CAM, and model metadata."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from backend.core.config import Settings
from backend.core.logging import get_logger
from ml.architectures.factory import build_model, count_parameters, target_layer_name
from ml.artifacts import load_checkpoint, resolve_checkpoint
from ml.classes import CLASS_NAMES, validate_class_names
from ml.config import INPUT_SIZE, dataset_meta
from ml.gradcam import GradCAM, GradCAMResult
from ml.transforms import build_eval_transform
from ml.utils import resolve_device

logger = get_logger("ecovision.model")

ARCH_DISPLAY = {
    "mobilenet_v2": "MobileNetV2 (transfer learning)",
    "custom_cnn": "Custom CNN (from scratch)",
}
ARCH_DESCRIPTION = {
    "mobilenet_v2": (
        "MobileNetV2 initialized with ImageNet weights, a frozen-backbone head-training "
        "stage, then fine-tuning of the final three inverted-residual blocks."
    ),
    "custom_cnn": (
        "A four-block convolutional network trained from scratch with batch "
        "normalization, global average pooling, and dropout."
    ),
}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    arch: str
    class_names: tuple[str, ...]
    stage: str
    version: str
    checkpoint: Path
    input_size: int
    trained_at: str | None
    extra: dict[str, Any] = field(default_factory=dict)
    gradcam: GradCAM | None = None


@dataclass
class InferenceResult:
    predicted_index: int
    predicted_class: str
    confidence: float
    top3: list[dict[str, Any]]
    gradcam: GradCAMResult
    stage: str
    version: str
    device: str


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded: LoadedModel | None = None
        self.metrics: dict[str, Any] | None = None
        self._transform = build_eval_transform()
        self._device = torch.device("cpu")
        self._load_error: str | None = None
        self._arch_catalog: list[dict[str, Any]] | None = None

    @property
    def available(self) -> bool:
        return self.loaded is not None

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> bool:
        checkpoint = resolve_checkpoint(
            self.settings.model_dir, self.settings.model_path
        )
        if checkpoint is None:
            self._load_error = "No checkpoint found."
            logger.warning(
                "No model checkpoint found in %s. The API will run in degraded mode.",
                self.settings.model_dir,
            )
            return False

        try:
            payload = load_checkpoint(checkpoint)
            class_names = validate_class_names(payload["class_names"])
            arch = str(payload["arch"])
            model = build_model(
                arch, len(class_names), pretrained=False
            )
            model.load_state_dict(payload["state_dict"])
            device = resolve_device(self.settings.device)
            if device.type == "cpu":
                torch.set_num_threads(min(4, os.cpu_count() or 1))
            model.to(device)
            model.eval()

            self._device = device
            self.loaded = LoadedModel(
                model=model,
                arch=arch,
                class_names=class_names,
                stage=str(payload.get("stage", "unknown")),
                version=str(payload.get("model_version", "unknown")),
                checkpoint=checkpoint,
                input_size=int(payload.get("input_size", INPUT_SIZE)),
                trained_at=payload.get("created_at"),
                extra=payload.get("extra") or {},
                gradcam=GradCAM(model, arch),
            )
            self.metrics = self._load_metrics(self.loaded.stage)
            logger.info(
                "Loaded %s checkpoint %s (stage=%s, device=%s)",
                arch,
                checkpoint.name,
                self.loaded.stage,
                device,
            )
            if self.loaded.stage != "final":
                logger.warning(
                    "Active checkpoint stage is '%s', not 'final'. "
                    "Final evaluation is still pending.",
                    self.loaded.stage,
                )
            return True
        except Exception as exc:
            self._load_error = str(exc)
            self.loaded = None
            logger.exception("Failed to load model checkpoint %s", checkpoint)
            return False

    def _load_metrics(self, stage: str) -> dict[str, Any] | None:
        folder = "final" if stage == "final" else "smoke"
        candidate = self.settings.evaluation_dir / folder / "metrics.json"
        if not candidate.exists():
            return None
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read metrics file %s", candidate)
            return None

    def predict(self, image: Image.Image, display: Image.Image) -> InferenceResult:
        if self.loaded is None:
            raise RuntimeError("Model is not loaded.")
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self.loaded.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

        top = torch.topk(probabilities, k=min(3, len(self.loaded.class_names)))
        top3 = [
            {
                "label": self.loaded.class_names[int(index)],
                "confidence": round(float(score), 6),
            }
            for score, index in zip(top.values, top.indices)
        ]
        predicted_index = int(top.indices[0])
        predicted_class = self.loaded.class_names[predicted_index]
        confidence = float(top.values[0])

        assert self.loaded.gradcam is not None
        gradcam = self.loaded.gradcam.generate(
            tensor, class_index=predicted_index, original=display
        )
        return InferenceResult(
            predicted_index=predicted_index,
            predicted_class=predicted_class,
            confidence=round(confidence, 6),
            top3=top3,
            gradcam=gradcam,
            stage=self.loaded.stage,
            version=self.loaded.version,
            device=str(self._device),
        )

    def _architecture_catalog(self) -> list[dict[str, Any]]:
        if self._arch_catalog is None:
            catalog = []
            for name in ("custom_cnn", "mobilenet_v2"):
                model = build_model(name, len(CLASS_NAMES), pretrained=False)
                catalog.append(
                    {
                        "name": name,
                        "display_name": ARCH_DISPLAY[name],
                        "parameters": count_parameters(model),
                        "input_size": INPUT_SIZE,
                        "target_layer": target_layer_name(name),
                        "description": ARCH_DESCRIPTION[name],
                    }
                )
            self._arch_catalog = catalog
        return self._arch_catalog

    def model_info(self) -> dict[str, Any]:
        dataset = dataset_meta()
        classes = [
            {"index": index, "name": name}
            for index, name in enumerate(CLASS_NAMES)
        ]
        base = {
            "model_loaded": self.available,
            "model_name": None,
            "model_stage": None,
            "model_version": None,
            "is_final": False,
            "device": None,
            "checkpoint": None,
            "trained_at": None,
            "architecture": None,
            "architectures": self._architecture_catalog(),
            "classes": classes,
            "dataset": dataset,
            "metrics": self.metrics,
            "comparison": (self.metrics or {}).get("comparison"),
            "limitations": self._limitations(),
        }
        if self.loaded is None:
            return base

        loaded = self.loaded
        base.update(
            {
                "model_name": ARCH_DISPLAY.get(loaded.arch, loaded.arch),
                "model_stage": loaded.stage,
                "model_version": loaded.version,
                "is_final": loaded.stage == "final",
                "device": str(self._device),
                "checkpoint": loaded.checkpoint.name,
                "trained_at": loaded.trained_at,
                "architecture": {
                    "name": loaded.arch,
                    "display_name": ARCH_DISPLAY.get(loaded.arch, loaded.arch),
                    "parameters": count_parameters(loaded.model),
                    "input_size": loaded.input_size,
                    "target_layer": target_layer_name(loaded.arch),
                    "description": ARCH_DESCRIPTION.get(loaded.arch, ""),
                },
            }
        )
        return base

    def _limitations(self) -> list[str]:
        limitations = [
            "Trained on landfill photography from a single Australian facility; accuracy may drop on household photos, different lighting, or other regions.",
            "The seven target classes are coarser than the nine RealWaste classes; Textile Trash is merged into General waste.",
            "The model assumes one dominant waste item per image and does not detect multiple objects.",
            "Recycling guidance is general and may not match local municipal rules, bin colors, or acceptance lists.",
            "Feedback is stored unverified and is not used for automatic retraining.",
        ]
        if self.loaded is None:
            limitations.insert(0, "No trained checkpoint is installed, so predictions are unavailable.")
        elif self.loaded.stage != "final":
            limitations.insert(
                0,
                "The active checkpoint is a development smoke model trained on a small subset; its predictions and metrics do not represent final model quality.",
            )
        return limitations
