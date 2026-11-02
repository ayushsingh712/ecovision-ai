#!/usr/bin/env python3
"""Download and validate the RealWaste dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from ml.config import DATASET_META_PATH, DATASETS_DIR, INSPECTION_JSON, RAW_DIR  # noqa: E402
from ml.data import find_images, infer_source_class  # noqa: E402
from ml.utils import utc_now_iso, write_json  # noqa: E402

DEFAULT_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
EXPECTED_MIN_BYTES = 600 * 1024 * 1024
HF_DATASET_ID = "shahzaibvohra/realwaste"
HF_META_URL = f"https://huggingface.co/api/datasets/{HF_DATASET_ID}?blobs=true"
HF_RESOLVE_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/"


def _fmt_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MB"


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "EcoVisionAI/1.0"})
    print(f"Downloading dataset from {url}")
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        started = time.perf_counter()
        with part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    speed = downloaded / max(time.perf_counter() - started, 1e-6)
                    print(
                        f"\r  {percent:5.1f}%  {_fmt_bytes(downloaded)} / "
                        f"{_fmt_bytes(total)}  ({_fmt_bytes(int(speed))}/s)",
                        end="",
                        flush=True,
                    )
    print()
    if part.stat().st_size < EXPECTED_MIN_BYTES:
        raise RuntimeError(
            f"Downloaded archive is unexpectedly small ({_fmt_bytes(part.stat().st_size)})."
        )
    part.replace(target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "EcoVisionAI/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _download_one(entry: dict, target_dir: Path) -> bool:
    name = entry["name"]
    destination = target_dir / name
    expected_hash = entry["sha256"]
    if destination.exists():
        if expected_hash is None or _sha256(destination) == expected_hash:
            return False

    quoted = urllib.parse.quote(name)
    request = urllib.request.Request(
        HF_RESOLVE_URL + quoted, headers={"User-Agent": "EcoVisionAI/1.0"}
    )
    part = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, part.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(response, handle, length=1024 * 256)
            if expected_hash and _sha256(part) != expected_hash:
                raise RuntimeError(f"Checksum mismatch for {name}")
            part.replace(destination)
            return True
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            part.unlink(missing_ok=True)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {name}: {last_error}")


def download_hf(raw_dir: Path, expected_total: int, workers: int = 16) -> int:
    """Download the verified Hugging Face mirror of RealWaste.

    The UCI file server uses chunked transfer without byte-range support, so a
    dropped connection restarts the whole 657 MB archive. This mirror exposes
    the same 4,752 files individually with per-file SHA-256 checksums.
    """
    metadata = _fetch_json(HF_META_URL)
    entries = []
    for sibling in metadata.get("siblings", []):
        name = sibling["rfilename"]
        if name.lower().endswith(".jpg"):
            lfs = sibling.get("lfs") or {}
            entries.append(
                {"name": name, "sha256": lfs.get("oid"), "size": lfs.get("size")}
            )
    if len(entries) != expected_total:
        raise RuntimeError(
            f"Mirror lists {len(entries)} images; expected {expected_total}."
        )

    target = raw_dir / "realwaste_hf"
    target.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {len(entries)} images from huggingface.co/datasets/{HF_DATASET_ID}"
    )

    downloaded = 0
    completed = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, entry, target): entry for entry in entries
        }
        try:
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
                with lock:
                    completed += 1
                    if completed % 200 == 0 or completed == len(entries):
                        print(
                            f"\r  {completed}/{len(entries)} checked, "
                            f"{downloaded} downloaded",
                            end="",
                            flush=True,
                        )
        except Exception:
            for future in futures:
                future.cancel()
            raise
    print()
    return len(entries)


def safe_extract(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt zip entry: {bad}")
        root = destination.resolve()
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if "__MACOSX" in name or name.startswith("._"):
                continue
            target = (destination / name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in archive: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted += 1
    return extracted


def inspect(raw_dir: Path, meta: dict) -> dict:
    images = find_images(raw_dir)
    source_counts: Counter[str] = Counter()
    unmapped: list[str] = []
    corrupt: list[str] = []
    sizes: Counter[tuple[int, int]] = Counter()

    for path in images:
        source = infer_source_class(path, raw_dir)
        if source is None:
            unmapped.append(str(path.relative_to(raw_dir)))
            continue
        source_counts[source] += 1
        try:
            with Image.open(path) as image:
                sizes[image.size] += 1
        except Exception:
            corrupt.append(str(path.relative_to(raw_dir)))

    expected = meta["source_classes"]
    differences = {
        name: {"expected": expected.get(name, 0), "found": source_counts.get(name, 0)}
        for name in sorted(set(expected) | set(source_counts))
        if expected.get(name, 0) != source_counts.get(name, 0)
    }
    report = {
        "inspected_at": utc_now_iso(),
        "total_images": len(images),
        "mapped_images": sum(source_counts.values()),
        "source_counts": dict(sorted(source_counts.items())),
        "expected_counts": expected,
        "differences": differences,
        "unmapped_images": unmapped[:50],
        "unmapped_count": len(unmapped),
        "corrupt_images": corrupt[:50],
        "corrupt_count": len(corrupt),
        "sample_dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in sizes.most_common()
        ],
        "dataset": meta["dataset"],
        "license": meta["license"],
    }
    write_json(INSPECTION_JSON, report)
    return report


def print_report(report: dict) -> None:
    print("\nRealWaste inspection")
    print("-" * 52)
    for name, count in report["source_counts"].items():
        expected = report["expected_counts"].get(name, 0)
        marker = "" if expected == count else f"  (expected {expected})"
        print(f"  {name:<22} {count:>5}{marker}")
    print("-" * 52)
    print(f"  {'Total mapped':<22} {report['mapped_images']:>5}")
    print(f"  {'Unmapped':<22} {report['unmapped_count']:>5}")
    print(f"  {'Corrupt':<22} {report['corrupt_count']:>5}")
    if report["sample_dimensions"]:
        dims = report["sample_dimensions"][0]
        print(f"  Typical dimensions: {dims['width']}x{dims['height']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--zip-path", type=Path, help="Use an already-downloaded archive.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--source",
        choices=["uci", "hf"],
        default="uci",
        help=(
            "uci: original archive (default). hf: verified Hugging Face mirror, "
            "resumable per file."
        ),
    )
    parser.add_argument("--workers", type=int, default=16, help="HF download workers.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Inspect but do not fail when counts differ.",
    )
    args = parser.parse_args()

    meta = json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = find_images(raw_dir)
    if existing and not args.force:
        print(f"Found {len(existing)} existing images in {raw_dir}; skipping download.")
    elif args.source == "hf":
        download_hf(raw_dir, meta["total_images"], workers=max(1, args.workers))
    else:
        archive = args.zip_path
        if archive is None:
            candidates = sorted(raw_dir.glob("*.zip"))
            archive = candidates[0] if candidates else raw_dir / "realwaste.zip"
        if not archive.exists() or archive.stat().st_size < EXPECTED_MIN_BYTES:
            archive = download(args.url, raw_dir / "realwaste.zip")
        else:
            print(f"Using existing archive: {archive}")
        extracted = safe_extract(archive, raw_dir)
        print(f"Extracted {extracted} files into {raw_dir}.")

    report = inspect(raw_dir, meta)
    print_report(report)

    if report["differences"] or report["unmapped_count"] or report["corrupt_count"]:
        print("\nInspection found unexpected files. Review ml/datasets/INSPECTION.json.")
        if not args.allow_count_mismatch:
            return 1
    if report["mapped_images"] != meta["total_images"]:
        print(
            f"\nExpected {meta['total_images']} images, found {report['mapped_images']}."
        )
        if not args.allow_count_mismatch:
            return 1
    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
