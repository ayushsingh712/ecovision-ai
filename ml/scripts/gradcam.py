#!/usr/bin/env python3
"""Generate Grad-CAM heatmaps and overlays for sample images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402

from ml.architectures.factory import build_model  # noqa: E402
from ml.artifacts import load_checkpoint, resolve_checkpoint  # noqa: E402
from ml.classes import CLASS_NAMES  # noqa: E402
from ml.config import DATASETS_DIR, EVAL_DIR, MODELS_DIR  # noqa: E402
from ml.data import Sample, find_images, load_samples  # noqa: E402
from ml.gradcam import GradCAM, build_three_panel  # noqa: E402
from ml.transforms import build_eval_transform  # noqa: E402
from ml.utils import resolve_device  # noqa: E402

DISPLAY_SIZE = 384


def _display(image: Image.Image) -> Image.Image:
    return ImageOps.fit(
        image.convert("RGB"),
        (DISPLAY_SIZE, DISPLAY_SIZE),
        Image.Resampling.LANCZOS,
    )


def _per_class_samples(
    datasets_dir: Path, per_class: int, split: str
) -> list[Sample]:
    samples = load_samples(datasets_dir / "manifest.csv", split=split)
    buckets: dict[str, list[Sample]] = {name: [] for name in CLASS_NAMES}
    for sample in samples:
        buckets[sample.label].append(sample)
    selected: list[Sample] = []
    for name in CLASS_NAMES:
        bucket = sorted(buckets[name], key=lambda item: item.sha256)
        selected.extend(bucket[:per_class])
    return selected


def _run(
    model,
    gradcam: GradCAM,
    transform,
    device: torch.device,
    image: Image.Image,
    class_index: int | None,
) -> tuple[int, float, np.ndarray, object]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    predicted = int(probabilities.argmax())
    target = predicted if class_index is None else class_index
    result = gradcam.generate(tensor, class_index=target, original=image)
    return predicted, float(probabilities[predicted]), np.asarray(result.cam), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Image files or folders.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "smoke" / "gradcam_samples")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Instead of paths, take N test-split images per class.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-index", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.models_dir, args.checkpoint)
    if checkpoint is None:
        print(f"No checkpoint found in {args.models_dir}.", file=sys.stderr)
        return 1

    payload = load_checkpoint(checkpoint)
    arch = payload["arch"]
    model = build_model(arch, len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    device = resolve_device(args.device)
    model.to(device).eval()
    transform = build_eval_transform()
    gradcam = GradCAM(model, arch)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {checkpoint} (stage={payload.get('stage')})")
    print(f"Output: {args.output}")

    items: list[tuple[Path, str, int | None]] = []
    for path in args.paths:
        if path.is_dir():
            items.extend((image, image.stem, args.class_index) for image in find_images(path))
        elif path.is_file():
            items.append((path, path.stem, args.class_index))
        else:
            print(f"Skipping missing path: {path}", file=sys.stderr)

    if args.samples_per_class > 0:
        for sample in _per_class_samples(args.data_dir, args.samples_per_class, args.split):
            items.append((sample.processed_path, f"{sample.label}_{sample.sha256[:8]}", None))

    if not items:
        print("No images to process.", file=sys.stderr)
        return 1

    for image_path, stem, class_index in items:
        try:
            with Image.open(image_path) as raw:
                image = _display(ImageOps.exif_transpose(raw))
        except (UnidentifiedImageError, OSError) as exc:
            print(f"Skipping unreadable image {image_path}: {exc}", file=sys.stderr)
            continue

        predicted, confidence, cam, result = _run(
            model, gradcam, transform, device, image, class_index
        )
        variance = float(cam.var())
        if variance <= 0:
            print(f"WARNING: degenerate heatmap for {image_path}", file=sys.stderr)

        panel = build_three_panel(result, CLASS_NAMES[predicted])
        target = args.output / f"{stem}_{CLASS_NAMES[predicted].lower()}.png"
        Image.fromarray(panel).save(target)
        result.overlay.save(args.output / f"{stem}_overlay.png")
        print(
            f"{image_path.name:<28} -> {CLASS_NAMES[predicted]:<10} "
            f"{confidence * 100:6.2f}%  cam_var={variance:.5f}  saved {target.name}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
