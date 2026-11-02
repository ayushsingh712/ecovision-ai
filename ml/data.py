"""Dataset discovery, hashing, manifests, and PyTorch datasets."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from ml.classes import CLASS_NAMES, CLASS_TO_INDEX
from ml.config import DATASETS_DIR, dataset_meta

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_NUMBER_SUFFIX = re.compile(r"[_\-\s]*\d+$")


def normalize_label(value: str) -> str:
    cleaned = value.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


_SOURCE_LOOKUP = {
    normalize_label(name).lower(): name
    for name in dataset_meta()["source_classes"].keys()
}


def infer_source_class(image_path: Path, raw_root: Path) -> str | None:
    for candidate in (image_path.parent.name, image_path.parent.parent.name):
        match = _SOURCE_LOOKUP.get(normalize_label(candidate).lower())
        if match:
            return match

    stem = _NUMBER_SUFFIX.sub("", image_path.stem)
    match = _SOURCE_LOOKUP.get(normalize_label(stem).lower())
    if match:
        return match
    return None


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "__MACOSX" not in path.parts
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(grayscale, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(frozen=True)
class Sample:
    processed_rel: str
    original_rel: str
    sha256: str
    dhash: int
    group_id: str
    source_class: str
    label: str
    split: str

    @property
    def label_index(self) -> int:
        return CLASS_TO_INDEX[self.label]

    @property
    def processed_path(self) -> Path:
        return DATASETS_DIR / self.processed_rel


class WasteDataset(Dataset):
    def __init__(self, samples: list[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.processed_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image), sample.label_index


def write_manifest(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "processed_rel",
        "original_rel",
        "sha256",
        "dhash",
        "group_id",
        "source_class",
        "label",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "processed_rel": sample.processed_rel,
                    "original_rel": sample.original_rel,
                    "sha256": sample.sha256,
                    "dhash": sample.dhash,
                    "group_id": sample.group_id,
                    "source_class": sample.source_class,
                    "label": sample.label,
                    "split": sample.split,
                }
            )


def load_samples(
    manifest_path: Path,
    split: str | None = None,
    labels: set[str] | None = None,
) -> list[Sample]:
    samples: list[Sample] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["split"] != split:
                continue
            if labels is not None and row["label"] not in labels:
                continue
            samples.append(
                Sample(
                    processed_rel=row["processed_rel"],
                    original_rel=row["original_rel"],
                    sha256=row["sha256"],
                    dhash=int(row["dhash"]),
                    group_id=row["group_id"],
                    source_class=row["source_class"],
                    label=row["label"],
                    split=row["split"],
                )
            )
    return samples


def limit_per_class(
    samples: list[Sample], max_per_class: int, seed: int = 42
) -> list[Sample]:
    if max_per_class <= 0:
        return samples
    rng = random.Random(seed)
    buckets: dict[str, list[Sample]] = {name: [] for name in CLASS_NAMES}
    for sample in samples:
        buckets[sample.label].append(sample)
    limited: list[Sample] = []
    for name in CLASS_NAMES:
        bucket = buckets[name]
        if len(bucket) > max_per_class:
            bucket = sorted(bucket, key=lambda item: item.sha256)
            bucket = rng.sample(bucket, max_per_class)
        limited.extend(bucket)
    return limited


def class_counts(samples: list[Sample]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        counts[sample.label] += 1
    return counts


def compute_class_weights(samples: list[Sample], cap: float = 4.0) -> list[float]:
    counts = class_counts(samples)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot compute class weights for an empty sample set.")
    weights = []
    for name in CLASS_NAMES:
        count = counts[name]
        weights.append(min(cap, total / (len(CLASS_NAMES) * count)) if count else 0.0)
    return weights
