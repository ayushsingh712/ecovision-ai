"""Canonical class order shared by training, serving, and the UI."""

from __future__ import annotations

import json
from pathlib import Path

CLASS_NAMES: tuple[str, ...] = (
    "Plastic",
    "Paper",
    "Cardboard",
    "Glass",
    "Metal",
    "Organic",
    "General",
)
CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


def validate_class_names(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(names)
    if normalized != CLASS_NAMES:
        raise ValueError(
            f"Class order mismatch. Expected {CLASS_NAMES}, received {normalized}."
        )
    return normalized


def load_class_names(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = payload["class_names"] if isinstance(payload, dict) else payload
    return validate_class_names(names)


def write_class_names(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": list(CLASS_NAMES),
        "num_classes": NUM_CLASSES,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
