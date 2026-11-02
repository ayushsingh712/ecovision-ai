"""Shared helpers for the two training entry points."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ml.classes import CLASS_NAMES  # noqa: E402
from ml.config import EVAL_DIR, MODELS_DIR  # noqa: E402
from ml.data import (  # noqa: E402
    Sample,
    WasteDataset,
    class_counts,
    limit_per_class,
    load_samples,
)
from ml.transforms import build_eval_transform, build_train_transform  # noqa: E402
from ml.utils import write_json  # noqa: E402


@dataclass(frozen=True)
class RunPaths:
    stage: str
    arch: str
    models_dir: Path
    eval_dir: Path
    checkpoint: Path
    history: Path
    curves: Path
    config: Path


def make_run_paths(
    arch: str,
    stage: str,
    models_root: Path = MODELS_DIR,
    eval_root: Path = EVAL_DIR,
) -> RunPaths:
    if stage == "smoke":
        models_dir = models_root / "smoke"
        eval_dir = eval_root / "smoke"
        suffix = "smoke"
    else:
        models_dir = models_root
        eval_dir = eval_root / "final"
        suffix = "best"
    return RunPaths(
        stage=stage,
        arch=arch,
        models_dir=models_dir,
        eval_dir=eval_dir,
        checkpoint=models_dir / f"{arch}_{suffix}.pt",
        history=eval_dir / f"{arch}_history.json",
        curves=eval_dir / f"{arch}_training_curves.png",
        config=eval_dir / f"{arch}_training_config.json",
    )


def split_hash(datasets_dir: Path) -> str:
    path = datasets_dir / "splits.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["split_hash"]


def load_split_samples(
    datasets_dir: Path,
    *,
    smoke: bool,
    max_per_class: int,
    seed: int,
    max_per_split: int = 0,
) -> dict[str, list[Sample]]:
    manifest = datasets_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing {manifest}. Run ml/scripts/prepare_dataset.py first."
        )
    result: dict[str, list[Sample]] = {}
    for split in ("train", "val", "test"):
        samples = load_samples(manifest, split=split)
        if max_per_split > 0:
            samples = samples[:max_per_split]
        if smoke:
            samples = limit_per_class(samples, max_per_class, seed=seed)
        if not samples:
            raise ValueError(f"Split '{split}' is empty.")
        result[split] = samples
    return result


def build_dataloaders(
    samples: dict[str, list[Sample]],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> dict[str, DataLoader]:
    train_dataset = WasteDataset(samples["train"], build_train_transform())
    val_dataset = WasteDataset(samples["val"], build_eval_transform())
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        "val": DataLoader(val_dataset, shuffle=False, **common),
    }


def print_split_summary(samples: dict[str, list[Sample]]) -> None:
    print("\nTraining data")
    print(f"{'Class':<12}{'Train':>8}{'Val':>8}{'Test':>8}")
    print("-" * 36)
    for label in CLASS_NAMES:
        counts = [
            sum(1 for sample in samples[split] if sample.label == label)
            for split in ("train", "val", "test")
        ]
        print(f"{label:<12}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}")
    print("-" * 36)
    totals = [class_counts(samples[split]) for split in ("train", "val", "test")]
    print(
        f"{'Total':<12}{sum(totals[0].values()):>8}"
        f"{sum(totals[1].values()):>8}{sum(totals[2].values()):>8}"
    )


def plot_history(history: list[dict[str, Any]], arch: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    series = [("loss", "Loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    for axis, (key, label) in zip(axes, series):
        axis.plot(
            epochs,
            [record[f"train_{key}"] for record in history],
            label="Train",
            linewidth=2,
        )
        axis.plot(
            epochs,
            [record[f"val_{key}"] for record in history],
            label="Validation",
            linewidth=2,
        )
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    phases = {record["phase"] for record in history}
    if len(phases) > 1:
        first_phase = history[0]["phase"]
        boundary = next(
            index for index, record in enumerate(history) if record["phase"] != first_phase
        )
        for axis in axes:
            axis.axvline(
                history[boundary]["epoch"] - 0.5,
                color="#94a3b8",
                linestyle="--",
                linewidth=1,
            )
    fig.suptitle(f"{arch} training history", fontsize=14)
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def save_run(
    *,
    model: torch.nn.Module,
    trainer,
    run: RunPaths,
    config: dict[str, Any],
    split_digest: str,
    duration_seconds: float,
    extra: dict[str, Any] | None = None,
) -> None:
    from ml.artifacts import build_checkpoint, read_manifest, save_checkpoint, write_manifest

    trainer.restore_best()
    payload = build_checkpoint(
        model=model,
        arch=run.arch,
        stage=run.stage,
        history=trainer.history,
        split_hash=split_digest,
        extra={
            "best_epoch": trainer.best_epoch,
            "best_phase": trainer.best_phase,
            "best_validation_macro_f1": trainer.best_metric,
            "duration_seconds": round(duration_seconds, 2),
            "config": config,
            **(extra or {}),
        },
    )
    save_checkpoint(run.checkpoint, payload)
    write_json(run.history, trainer.history)
    write_json(run.config, config)
    plot_history(trainer.history, run.arch, run.curves)

    manifest_root = run.models_dir.parent if run.stage == "smoke" else run.models_dir
    manifest = read_manifest(manifest_root)
    if run.stage == "final" or manifest.get("stage") != "final":
        active = (
            f"smoke/{run.checkpoint.name}"
            if run.stage == "smoke"
            else run.checkpoint.name
        )
        active_arch = run.arch
        active_metric = trainer.best_metric
        if run.stage == "smoke":
            preferred = run.models_dir / "mobilenet_v2_smoke.pt"
            if preferred.exists() and preferred.name != run.checkpoint.name:
                from ml.artifacts import load_checkpoint

                active = f"smoke/{preferred.name}"
                active_arch = "mobilenet_v2"
                active_metric = float(
                    load_checkpoint(preferred).get("extra", {}).get(
                        "best_validation_macro_f1", 0.0
                    )
                )
        write_manifest(
            manifest_root,
            active=active,
            stage=run.stage,
            arch=active_arch,
            validation_macro_f1=active_metric,
        )

    print(f"\nSaved checkpoint: {run.checkpoint}")
    print(f"Best epoch: {trainer.best_epoch} ({trainer.best_phase})")
    print(f"Best validation macro-F1: {trainer.best_metric:.4f}")
    print(f"History: {run.history}")
    print(f"Curves:  {run.curves}")
    print(f"Config:  {run.config}")
