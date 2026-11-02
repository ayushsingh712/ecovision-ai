"""Leakage and distribution guarantees for the prepared dataset splits."""

from __future__ import annotations

import csv
import json

import pytest

from ml.classes import CLASS_NAMES
from ml.config import DATASETS_DIR

MANIFEST = DATASETS_DIR / "manifest.csv"
SPLITS = DATASETS_DIR / "splits.json"

pytestmark = pytest.mark.skipif(
    not (MANIFEST.exists() and SPLITS.exists()),
    reason="Dataset splits are not prepared.",
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def splits_payload() -> dict:
    return json.loads(SPLITS.read_text(encoding="utf-8"))


def test_every_class_present_in_every_split(rows):
    for split in ("train", "val", "test"):
        present = {row["label"] for row in rows if row["split"] == split}
        assert present == set(CLASS_NAMES), f"{split} is missing {set(CLASS_NAMES) - present}"


def test_counts_match_splits_metadata(rows, splits_payload):
    counts = {split: 0 for split in ("train", "val", "test")}
    per_class = {
        name: {split: 0 for split in ("train", "val", "test")} for name in CLASS_NAMES
    }
    for row in rows:
        counts[row["split"]] += 1
        per_class[row["label"]][row["split"]] += 1
    assert counts == splits_payload["counts"]
    assert sum(counts.values()) == splits_payload["total_images"]
    assert per_class == splits_payload["per_class"]


def test_no_exact_duplicates_across_splits(rows):
    seen: dict[str, str] = {}
    for row in rows:
        previous = seen.setdefault(row["sha256"], row["split"])
        assert previous == row["split"], (
            f"sha256 {row['sha256']} appears in both {previous} and {row['split']}"
        )


def test_no_perceptual_duplicates_across_splits(rows):
    seen: dict[str, str] = {}
    for row in rows:
        previous = seen.setdefault(row["group_id"], row["split"])
        assert previous == row["split"], (
            f"group {row['group_id']} appears in both {previous} and {row['split']}"
        )


def test_class_weights_are_bounded_and_cover_all_classes(splits_payload):
    weights = splits_payload["class_weights"]
    assert set(weights) == set(CLASS_NAMES)
    train_total = splits_payload["counts"]["train"]
    for name, weight in weights.items():
        assert 0 < weight <= 4.0
        expected = train_total / (len(CLASS_NAMES) * splits_payload["per_class"][name]["train"])
        assert weight == pytest.approx(min(4.0, expected))
