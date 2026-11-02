#!/usr/bin/env python3
"""Map RealWaste to seven classes, build leakage-safe splits, and cache images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageOps  # noqa: E402

from ml.classes import CLASS_NAMES  # noqa: E402
from ml.config import (  # noqa: E402
    CACHE_SIZE,
    CLASS_NAMES_PATH,
    DATASET_META_PATH,
    DATASETS_DIR,
    MANIFEST_CSV,
    NEAR_DUPLICATE_HAMMING,
    PROCESSED_DIR,
    SPLIT_FRACTIONS,
    SPLITS_JSON,
)
from ml.data import (  # noqa: E402
    Sample,
    compute_class_weights,
    dhash,
    find_images,
    hamming,
    infer_source_class,
    sha256_file,
    write_manifest,
)
from ml.classes import write_class_names  # noqa: E402
from ml.utils import utc_now_iso, write_json  # noqa: E402


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def group_near_duplicates(records: list[dict], threshold: int) -> dict[int, str]:
    """Group near-duplicate frames within each mapped class."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    group_ids: dict[int, str] = {}
    for label, items in by_label.items():
        uf = UnionFind()
        for item in items:
            uf.find(item["sha256"])
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                if hamming(left["dhash"], right["dhash"]) <= threshold:
                    uf.union(left["sha256"], right["sha256"])
        buckets: dict[str, list[str]] = defaultdict(list)
        for item in items:
            buckets[uf.find(item["sha256"])].append(item["sha256"])
        for members in buckets.values():
            digest = hashlib.sha256(
                "|".join(sorted(members)).encode("utf-8")
            ).hexdigest()[:16]
            group_id = f"{label.lower().replace(' ', '-')}-{digest}"
            for sha in members:
                group_ids[sha] = group_id
    return group_ids


def split_groups(
    groups: dict[str, list[list[dict]]], seed: int
) -> dict[str, str]:
    """Assign whole duplicate groups to splits with per-class stratified ratios."""
    assignment: dict[str, str] = {}
    for class_index, label in enumerate(CLASS_NAMES):
        class_groups = list(groups.get(label, []))
        rng = random.Random(seed + class_index)
        rng.shuffle(class_groups)
        total = sum(len(group) for group in class_groups)
        train_target = int(round(total * SPLIT_FRACTIONS["train"]))
        val_target = int(round(total * SPLIT_FRACTIONS["val"]))
        counts = {"train": 0, "val": 0, "test": 0}

        def select_split() -> str:
            deficits = {
                "train": max(train_target - counts["train"], 0),
                "val": max(val_target - counts["val"], 0),
                "test": max(total - train_target - val_target - counts["test"], 0),
            }
            priority = {"train": 2, "val": 1, "test": 0}
            return max(deficits, key=lambda split: (deficits[split], priority[split]))

        for group in class_groups:
            split = select_split()
            counts[split] += len(group)
            for member in group:
                assignment[member["sha256"]] = split
    return assignment


def cache_image(source: Path, target: Path, size: int) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=92, optimize=True)


def assert_no_leakage(samples: list[Sample]) -> None:
    shas: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        shas[sample.split].add(sample.sha256)
        groups[sample.split].add(sample.group_id)
    for left in ("train", "val", "test"):
        for right in ("train", "val", "test"):
            if left >= right:
                continue
            sha_overlap = shas[left] & shas[right]
            group_overlap = groups[left] & groups[right]
            if sha_overlap or group_overlap:
                raise AssertionError(
                    f"Leakage between {left} and {right}: "
                    f"{len(sha_overlap)} duplicate images, "
                    f"{len(group_overlap)} duplicate groups."
                )
    for split in ("train", "val", "test"):
        labels = {sample.label for sample in samples if sample.split == split}
        missing = set(CLASS_NAMES) - labels
        if missing:
            raise AssertionError(f"Split '{split}' is missing classes: {sorted(missing)}")


def print_summary(samples: list[Sample]) -> None:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        table[sample.label][sample.split] += 1
    print("\nSplit summary")
    print(f"{'Class':<12}{'Train':>8}{'Val':>8}{'Test':>8}{'Total':>8}")
    print("-" * 44)
    totals = Counter()
    for label in CLASS_NAMES:
        counts = table[label]
        total = sum(counts.values())
        totals.update(counts)
        print(
            f"{label:<12}{counts['train']:>8}{counts['val']:>8}"
            f"{counts['test']:>8}{total:>8}"
        )
    print("-" * 44)
    print(
        f"{'Total':<12}{totals['train']:>8}{totals['val']:>8}"
        f"{totals['test']:>8}{sum(totals.values()):>8}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATASETS_DIR / "raw")
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-size", type=int, default=CACHE_SIZE)
    parser.add_argument("--near-duplicate-hamming", type=int, default=NEAR_DUPLICATE_HAMMING)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta = json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
    raw_dir: Path = args.raw_dir
    datasets_dir: Path = args.datasets_dir
    processed_dir = datasets_dir / "processed"

    images = find_images(raw_dir)
    if not images:
        print(f"No images found in {raw_dir}. Run download_dataset.py first.")
        return 1

    records: list[dict] = []
    unmapped: list[str] = []
    seen_shas: set[str] = set()
    duplicate_count = 0
    corrupt: list[str] = []

    print(f"Scanning {len(images)} images...")
    for path in images:
        source = infer_source_class(path, raw_dir)
        if source is None:
            unmapped.append(str(path))
            continue
        label = next(
            target
            for target, sources in meta["class_mapping"].items()
            if source in sources
        )
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            sha = digest.hexdigest()
            if sha in seen_shas:
                duplicate_count += 1
                continue
            seen_shas.add(sha)
            perceptual = dhash(path)
        except Exception as exc:
            corrupt.append(f"{path}: {exc}")
            continue
        records.append(
            {
                "path": path,
                "sha256": sha,
                "dhash": perceptual,
                "source_class": source,
                "label": label,
            }
        )

    if corrupt:
        print(f"Skipping {len(corrupt)} unreadable images.")
    if unmapped:
        print(f"Skipping {len(unmapped)} images with unknown source labels.")

    group_ids = group_near_duplicates(records, args.near_duplicate_hamming)
    group_members: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        record["group_id"] = group_ids[record["sha256"]]
        group_members[record["group_id"]].append(record)

    groups_by_label: dict[str, list[list[dict]]] = defaultdict(list)
    for members in group_members.values():
        groups_by_label[members[0]["label"]].append(members)
    assignments = split_groups(groups_by_label, args.seed)

    samples: list[Sample] = []
    for record in records:
        source_path: Path = record["path"]
        cached_rel = f"processed/{record['label']}/{record['sha256'][:16]}.jpg"
        cache_path = datasets_dir / cached_rel
        cache_image(source_path, cache_path, args.cache_size)
        original_rel = str(source_path.relative_to(datasets_dir))
        samples.append(
            Sample(
                processed_rel=cached_rel,
                original_rel=original_rel,
                sha256=record["sha256"],
                dhash=record["dhash"],
                group_id=record["group_id"],
                source_class=record["source_class"],
                label=record["label"],
                split=assignments[record["sha256"]],
            )
        )

    assert_no_leakage(samples)
    samples.sort(key=lambda item: (item.split, CLASS_NAMES.index(item.label), item.sha256))
    manifest_path = datasets_dir / "manifest.csv"
    write_manifest(manifest_path, samples)

    train_samples = [sample for sample in samples if sample.split == "train"]
    class_weights = compute_class_weights(train_samples)
    split_payload = {
        "seed": args.seed,
        "fractions": SPLIT_FRACTIONS,
        "near_duplicate_hamming": args.near_duplicate_hamming,
        "cache_size": args.cache_size,
        "total_images": len(samples),
        "exact_duplicates_removed": duplicate_count,
        "near_duplicate_groups": len(group_members),
        "counts": {
            split: sum(1 for sample in samples if sample.split == split)
            for split in ("train", "val", "test")
        },
        "per_class": {
            label: {
                split: sum(
                    1
                    for sample in samples
                    if sample.label == label and sample.split == split
                )
                for split in ("train", "val", "test")
            }
            for label in CLASS_NAMES
        },
        "class_weights": dict(zip(CLASS_NAMES, class_weights)),
        "split_hash": hashlib.sha256(
            "|".join(
                f"{sample.sha256}:{sample.split}"
                for sample in sorted(samples, key=lambda item: item.sha256)
            ).encode("utf-8")
        ).hexdigest(),
        "created_at": utc_now_iso(),
    }
    write_json(datasets_dir / "splits.json", split_payload)
    write_class_names(CLASS_NAMES_PATH)

    report = {
        "scanned": len(images),
        "mapped": len(samples),
        "unmapped": unmapped[:50],
        "unmapped_count": len(unmapped),
        "corrupt": corrupt[:50],
        "corrupt_count": len(corrupt),
        "exact_duplicates_removed": duplicate_count,
        "near_duplicate_groups": len(group_members),
        "created_at": utc_now_iso(),
    }
    write_json(datasets_dir / "PREPARE_REPORT.json", report)
    print_summary(samples)
    print(f"\nManifest: {manifest_path}")
    print(f"Splits:   {datasets_dir / 'splits.json'}")
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights))}")
    print("Leakage checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
