#!/usr/bin/env python3
"""Evaluate both trained models on the held-out test split and compare them."""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader  # noqa: E402

from ml.architectures.factory import build_model, count_parameters  # noqa: E402
from ml.artifacts import load_checkpoint  # noqa: E402
from ml.classes import CLASS_NAMES  # noqa: E402
from ml.config import (  # noqa: E402
    DATASETS_DIR,
    EVAL_DIR,
    MODELS_DIR,
    SEED,
    dataset_meta,
)
from ml.data import Sample, WasteDataset, class_counts  # noqa: E402
from ml.training_utils import (  # noqa: E402
    load_split_samples,
    make_run_paths,
    split_hash,
)
from ml.transforms import build_eval_transform  # noqa: E402
from ml.utils import resolve_device, seed_everything, utc_now_iso, write_json  # noqa: E402

ARCHS = ("custom_cnn", "mobilenet_v2")
ARCH_LABELS = {"custom_cnn": "Custom CNN", "mobilenet_v2": "MobileNetV2"}


def _metric_block(y_true, y_pred) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(macro[0]), 6),
        "macro_recall": round(float(macro[1]), 6),
        "macro_f1": round(float(macro[2]), 6),
        "weighted_precision": round(float(weighted[0]), 6),
        "weighted_recall": round(float(weighted[1]), 6),
        "weighted_f1": round(float(weighted[2]), 6),
        "per_class": {
            name: {
                "precision": round(float(per_class[0][index]), 6),
                "recall": round(float(per_class[1][index]), 6),
                "f1": round(float(per_class[2][index]), 6),
                "support": int(per_class[3][index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def _predict_all(
    model: torch.nn.Module, samples: list[Sample], device: torch.device, batch_size: int
) -> tuple[list[int], list[int]]:
    dataset = WasteDataset(samples, build_eval_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def _measure_latency(
    model: torch.nn.Module,
    samples: list[Sample],
    device: torch.device,
    iterations: int = 30,
) -> dict:
    transform = build_eval_transform()
    tensors = []
    for sample in samples[: iterations + 3]:
        with Image.open(sample.processed_path) as image:
            tensor = transform(ImageOps.exif_transpose(image).convert("RGB"))
        tensors.append(tensor.unsqueeze(0).to(device))
    if not tensors:
        return {"mean_ms": None, "median_ms": None}

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    timings = []
    with torch.inference_mode():
        for tensor in tensors[:3]:
            model(tensor)
        for tensor in tensors[3:]:
            _sync()
            started = time.perf_counter()
            model(tensor)
            _sync()
            timings.append((time.perf_counter() - started) * 1000)
    return {
        "mean_ms": round(statistics.fmean(timings), 2),
        "median_ms": round(statistics.median(timings), 2),
        "images": len(timings),
    }


def _plot_confusion(matrix: list[list[int]], arch: str, stage: str, target: Path) -> None:
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(values, cmap="Greens")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if values[row, column] > threshold else "#12372a",
            )
    ax.set_title(f"{ARCH_LABELS[arch]} confusion matrix ({stage} test split)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(metrics: dict, stage: str, target: Path) -> None:
    keys = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")]
    archs = [arch for arch in ARCHS if arch in metrics]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = {"custom_cnn": "#7fb685", "mobilenet_v2": "#1f6f43"}
    for offset, arch in enumerate(archs):
        values = [metrics[arch][key] for key, _ in keys]
        positions = x + (offset - (len(archs) - 1) / 2) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=ARCH_LABELS[arch],
            color=colors.get(arch, "#1f6f43"),
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in keys])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"Model comparison on the held-out test split ({stage})")
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _comparison_rows(results: dict) -> list[dict]:
    rows = []
    for arch in ARCHS:
        if arch not in results:
            continue
        result = results[arch]
        metrics = result["metrics"]
        rows.append(
            {
                "architecture": arch,
                "model": ARCH_LABELS[arch],
                "parameters": result["parameters"],
                "checkpoint_mb": result["checkpoint_mb"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "cpu_latency_ms": result["latency"]["mean_ms"],
                "training_seconds": result["training_seconds"],
                "best_epoch": result["best_epoch"],
            }
        )
    return rows


def _comparison_markdown(rows: list[dict], stage: str) -> str:
    lines = [
        f"# Model comparison ({stage} test split)",
        "",
        "| Model | Parameters | Checkpoint | Accuracy | Macro F1 | Weighted F1 | CPU latency | Training time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        latency = (
            f"{row['cpu_latency_ms']:.1f} ms" if row["cpu_latency_ms"] else "n/a"
        )
        training = (
            f"{row['training_seconds'] / 60:.1f} min"
            if row["training_seconds"]
            else "n/a"
        )
        lines.append(
            f"| {row['model']} | {row['parameters']:,} | "
            f"{row['checkpoint_mb']:.1f} MB | {row['accuracy']:.3f} | "
            f"{row['macro_f1']:.3f} | {row['weighted_f1']:.3f} | "
            f"{latency} | {training} |"
        )
    if stage != "final":
        lines.extend(
            [
                "",
                "> **Development smoke run.** These numbers come from models trained on a "
                "small stratified subset for pipeline validation. They do not represent "
                "final model quality. Run the Colab notebook and re-run evaluation with "
                "`--stage final` after installing the artifacts.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["smoke", "final"],
        default="smoke",
        help="Which checkpoints and output folder to use.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-per-class", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arch", action="append", choices=ARCHS)
    args = parser.parse_args()

    seed_everything(SEED)
    stage = args.stage
    archs = tuple(args.arch) if args.arch else ARCHS
    device = resolve_device(args.device)
    print(f"Evaluating {', '.join(archs)} | stage={stage} | device={device}")

    split_digest = split_hash(args.data_dir)
    samples = load_split_samples(
        args.data_dir,
        smoke=stage == "smoke",
        max_per_class=args.max_per_class,
        seed=SEED,
    )
    test_samples = samples["test"]
    print(f"Test images: {len(test_samples)} | counts: {class_counts(test_samples)}")

    results: dict[str, dict] = {}
    for arch in archs:
        run = make_run_paths(arch, stage, args.models_dir, args.eval_dir)
        if not run.checkpoint.exists():
            print(f"Missing checkpoint for {arch}: {run.checkpoint}")
            return 1
        payload = load_checkpoint(run.checkpoint)
        model = build_model(arch, len(CLASS_NAMES), pretrained=False)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()

        started = time.perf_counter()
        y_true, y_pred = _predict_all(model, test_samples, device, args.batch_size)
        elapsed = time.perf_counter() - started
        metrics = _metric_block(y_true, y_pred)
        latency = _measure_latency(model, test_samples, device)

        run.eval_dir.mkdir(parents=True, exist_ok=True)
        report_text = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=list(CLASS_NAMES),
            digits=4,
            zero_division=0,
        )
        header = (
            f"{ARCH_LABELS[arch]} — {stage} test split\n"
            f"Images: {len(test_samples)}\n"
            f"Split hash: {split_digest[:16]}...\n"
            f"Evaluated: {utc_now_iso()}\n\n"
        )
        (run.eval_dir / f"classification_report_{arch}.txt").write_text(
            header + report_text, encoding="utf-8"
        )
        _plot_confusion(
            metrics["confusion_matrix"],
            arch,
            stage,
            run.eval_dir / f"confusion_matrix_{arch}.png",
        )

        extra = payload.get("extra") or {}
        results[arch] = {
            "stage": stage,
            "checkpoint": str(run.checkpoint.name),
            "parameters": count_parameters(model),
            "checkpoint_mb": round(run.checkpoint.stat().st_size / (1024 * 1024), 2),
            "test_images": len(test_samples),
            "metrics": metrics,
            "latency": latency,
            "evaluation_seconds": round(elapsed, 2),
            "training_seconds": extra.get("duration_seconds"),
            "best_epoch": extra.get("best_epoch"),
            "best_phase": extra.get("best_phase"),
            "trained_at": payload.get("created_at"),
            "split_hash": payload.get("split_hash"),
        }
        print(
            f"\n{ARCH_LABELS[arch]}: accuracy={metrics['accuracy']:.4f} "
            f"macro-F1={metrics['macro_f1']:.4f} "
            f"weighted-F1={metrics['weighted_f1']:.4f}"
        )

    rows = _comparison_rows(results)
    output = {
        "stage": stage,
        "evaluated_at": utc_now_iso(),
        "dataset": dataset_meta()["dataset"],
        "test_split": {
            "images": len(test_samples),
            "counts": class_counts(test_samples),
            "split_hash": split_digest,
        },
        "models": results,
        "comparison": rows,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
        "notes": (
            "Smoke-run metrics validate the pipeline only; they are not final results."
            if stage != "final"
            else "Final metrics from the complete RealWaste training run."
        ),
    }
    eval_dir = args.eval_dir / ("smoke" if stage == "smoke" else "final")
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir / "metrics.json", output)
    (eval_dir / "comparison.md").write_text(
        _comparison_markdown(rows, stage), encoding="utf-8"
    )
    _plot_comparison(
        {arch: results[arch]["metrics"] for arch in results},
        stage,
        eval_dir / "comparison.png",
    )

    print(f"\nWrote metrics: {eval_dir / 'metrics.json'}")
    print(f"Wrote comparison: {eval_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
