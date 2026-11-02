"""Load and validate the maintained recycling advice guide."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.classes import CLASS_NAMES

REQUIRED_KEYS = {
    "bin_type",
    "bin_color",
    "disposal_method",
    "instructions",
    "environmental_impact",
    "examples",
    "cautions",
}


class AdviceService:
    def __init__(self, path: Path) -> None:
        self.path = path
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.version: str = payload["version"]
        self.last_reviewed: str = payload["last_reviewed"]
        self.disclaimer: str = payload["disclaimer"]
        self.region_note: str = payload["region_note"]
        self.sources: list[dict[str, str]] = payload.get("sources", [])
        self.categories: dict[str, dict[str, Any]] = payload["categories"]

        missing = set(CLASS_NAMES) - set(self.categories)
        if missing:
            raise ValueError(f"Recycling guide is missing classes: {sorted(missing)}")
        for label, entry in self.categories.items():
            absent = REQUIRED_KEYS - set(entry)
            if absent:
                raise ValueError(
                    f"Recycling guide entry '{label}' is missing: {sorted(absent)}"
                )

    def for_class(self, label: str) -> dict[str, Any]:
        entry = self.categories[label]
        return {
            "bin_type": entry["bin_type"],
            "bin_color": entry["bin_color"],
            "disposal_method": entry["disposal_method"],
            "instructions": list(entry["instructions"]),
            "environmental_impact": entry["environmental_impact"],
            "examples": list(entry["examples"]),
            "cautions": list(entry["cautions"]),
            "scope": "general",
            "region_note": self.region_note,
            "disclaimer": self.disclaimer,
            "sources": entry.get("sources", self.sources),
        }
