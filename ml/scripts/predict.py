#!/usr/bin/env python3
"""Classify one image or every image in a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402

from ml.architectures.factory import build_model  # noqa: E402
from ml.artifacts import load_checkpoint, resolve_checkpoint  # noqa: E402
from ml.classes import CLASS_NAMES  # noqa: E402
from ml.config import MODELS_DIR  # noqa: E402
from ml.data import find_images  # noqa: E402
from ml.transforms import build_eval_transform  # noqa: E402
from ml.utils import resolve_device  # noqa: E402


def _collect(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images.extend(find_images(path))
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Image files or folders.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.models_dir, args.checkpoint)
    if checkpoint is None:
        print(f"No checkpoint found in {args.models_dir}.", file=sys.stderr)
        return 1

    payload = load_checkpoint(checkpoint)
    arch = payload["arch"]
    stage = payload.get("stage", "unknown")
    model = build_model(arch, len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    device = resolve_device(args.device)
    model.to(device).eval()
    transform = build_eval_transform()

    images = _collect(args.paths)
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    if not args.json:
        print(f"Checkpoint: {checkpoint} (arch={arch}, stage={stage})")
        if stage != "final":
            print("WARNING: development smoke model; predictions are not final quality.")

    results = []
    with torch.inference_mode():
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                print(f"Skipping unreadable image {image_path}: {exc}", file=sys.stderr)
                continue
            tensor = transform(rgb).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
            top = torch.topk(probabilities, k=min(args.top_k, len(CLASS_NAMES)))
            predictions = [
                {
                    "class": CLASS_NAMES[int(index)],
                    "confidence": round(float(score), 4),
                }
                for score, index in zip(top.values, top.indices)
            ]
            results.append({"image": str(image_path), "predictions": predictions})
            if not args.json:
                print(f"\n{image_path}")
                for item in predictions:
                    bar = "#" * int(item["confidence"] * 30)
                    print(f"  {item['class']:<10} {item['confidence'] * 100:6.2f}%  {bar}")

    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
