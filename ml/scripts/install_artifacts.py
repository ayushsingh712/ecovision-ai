#!/usr/bin/env python3
"""Validate and install final training artifacts produced by the Colab notebook."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.architectures.factory import build_model  # noqa: E402
from ml.artifacts import load_checkpoint, read_manifest, write_manifest  # noqa: E402
from ml.classes import CLASS_NAMES, validate_class_names  # noqa: E402
from ml.config import EVAL_DIR, MODELS_DIR  # noqa: E402

MODEL_FILES = {
    "mobilenet_v2": "mobilenet_v2_best.pt",
    "custom_cnn": "custom_cnn_best.pt",
}
MANIFEST_NAME = "model_manifest.json"
EVAL_SUFFIXES = (".json", ".md", ".txt", ".png", ".csv")


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Corrupt zip entry: {bad}")
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if "__MACOSX" in name or Path(name).name.startswith("._"):
                continue
            target = (destination / name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in archive: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def _find(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _discover_models(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for arch, filename in MODEL_FILES.items():
        match = _find(root, filename)
        if match is None:
            raise RuntimeError(f"Archive is missing {filename}")
        found[arch] = match
    return found


def _validate_checkpoint(arch: str, path: Path) -> dict:
    payload = load_checkpoint(path)
    if payload["arch"] != arch:
        raise RuntimeError(
            f"{path.name} declares arch '{payload['arch']}', expected '{arch}'."
        )
    if payload.get("stage") != "final":
        raise RuntimeError(
            f"{path.name} has stage '{payload.get('stage')}', expected 'final'. "
            "Smoke checkpoints cannot be installed as final."
        )
    validate_class_names(payload["class_names"])
    if list(payload["class_names"]) != list(CLASS_NAMES):
        raise RuntimeError(f"{path.name} class order does not match the app.")
    model = build_model(arch, len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    return payload


def _validate_manifest(root: Path, models: dict[str, Path]) -> dict:
    path = _find(root, MANIFEST_NAME)
    if path is None:
        print("No model_manifest.json in archive; one will be generated.")
        first = next(iter(models))
        return {
            "active": models[first].name,
            "stage": "final",
            "arch": first,
            "model_version": "realwaste-7class-v1",
            "class_names": list(CLASS_NAMES),
            "metrics_path": "evaluation/final/metrics.json",
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_class_names(manifest.get("class_names") or [])
    if manifest.get("stage") != "final":
        raise RuntimeError("Manifest stage must be 'final'.")
    active = manifest.get("active", "")
    if Path(active).name not in {path.name for path in models.values()}:
        raise RuntimeError(
            f"Manifest active checkpoint '{active}' is not one of the installed models."
        )
    return manifest


def _validate_metrics(root: Path) -> tuple[dict, Path]:
    path = _find(root, "metrics.json")
    if path is None:
        raise RuntimeError("Archive is missing evaluation metrics.json.")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("stage") != "final":
        raise RuntimeError(
            f"metrics.json stage is '{metrics.get('stage')}', expected 'final'."
        )
    models = metrics.get("models") or {}
    for arch in MODEL_FILES:
        if arch not in models:
            raise RuntimeError(f"metrics.json is missing results for {arch}.")
        accuracy = models[arch].get("metrics", {}).get("accuracy")
        if accuracy is None:
            raise RuntimeError(f"metrics.json has no accuracy for {arch}.")
    if not metrics.get("comparison"):
        raise RuntimeError("metrics.json is missing the comparison table.")
    images = metrics.get("test_split", {}).get("images", 0)
    if not images:
        raise RuntimeError("metrics.json reports an empty test split.")
    return metrics, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.archive.is_file():
        print(f"Archive not found: {args.archive}", file=sys.stderr)
        return 1

    final_eval_dir = args.eval_dir / "final"
    targets = [args.models_dir / name for name in MODEL_FILES.values()]
    if (targets[0].exists() or targets[1].exists()) and not args.force:
        print(
            "Final checkpoints already exist. Re-run with --force to replace them.",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        extracted = Path(tmp)
        _safe_extract(args.archive, extracted)

        models = _discover_models(extracted)
        payloads = {}
        for arch, path in models.items():
            payloads[arch] = _validate_checkpoint(arch, path)
            parameters = sum(
                tensor.numel() for tensor in payloads[arch]["state_dict"].values()
            )
            print(
                f"OK {path.name}: arch={arch}, stage=final, "
                f"{parameters:,} parameters"
            )

        manifest = _validate_manifest(extracted, models)
        metrics, metrics_path = _validate_metrics(extracted)
        print(
            f"OK metrics: test images={metrics['test_split']['images']}, "
            f"models={', '.join(metrics['models'])}"
        )
        for arch, entry in metrics["models"].items():
            data = entry["metrics"]
            print(
                f"   {arch}: accuracy={data['accuracy']:.4f} "
                f"macro_f1={data['macro_f1']:.4f}"
            )

        eval_root = metrics_path.parent
        eval_files = [
            path
            for path in eval_root.rglob("*")
            if path.is_file() and path.suffix.lower() in EVAL_SUFFIXES
        ]
        if not eval_files:
            raise RuntimeError("No evaluation artifacts found next to metrics.json.")

        if args.dry_run:
            print("\nDry run complete; nothing was copied.")
            return 0

        args.models_dir.mkdir(parents=True, exist_ok=True)
        final_eval_dir.mkdir(parents=True, exist_ok=True)
        for arch, path in models.items():
            shutil.copy2(path, args.models_dir / path.name)
        for path in eval_files:
            relative = path.relative_to(eval_root)
            destination = final_eval_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    manifest_path = args.models_dir / MANIFEST_NAME
    existing = read_manifest(args.models_dir)
    write_manifest(
        args.models_dir,
        active=manifest["active"],
        stage="final",
        arch=manifest.get("arch") or existing.get("arch") or "mobilenet_v2",
        metrics_path=manifest.get("metrics_path") or "evaluation/final/metrics.json",
        validation_macro_f1=manifest.get("validation_macro_f1"),
    )
    print(f"\nInstalled checkpoints into {args.models_dir}")
    print(f"Installed evaluation into {final_eval_dir}")
    print(f"Manifest: {manifest_path}")
    print("Restart the API to serve the final model.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Artifact validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
