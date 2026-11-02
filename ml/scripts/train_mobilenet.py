#!/usr/bin/env python3
"""Train MobileNetV2 with a frozen head stage followed by fine-tuning."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau  # noqa: E402

from ml.architectures.factory import build_model, count_parameters  # noqa: E402
from ml.architectures.mobilenet_v2 import (  # noqa: E402
    freeze_backbone,
    make_parameter_groups,
    trainable_parameter_count,
    unfreeze_last_blocks,
)
from ml.classes import CLASS_NAMES, NUM_CLASSES, write_class_names  # noqa: E402
from ml.config import (  # noqa: E402
    CLASS_NAMES_PATH,
    DATASETS_DIR,
    EVAL_DIR,
    MODELS_DIR,
    SEED,
    dataset_meta,
)
from ml.data import compute_class_weights  # noqa: E402
from ml.training_utils import (  # noqa: E402
    build_dataloaders,
    load_split_samples,
    make_run_paths,
    print_split_summary,
    save_run,
    split_hash,
)
from ml.trainer import Trainer  # noqa: E402
from ml.utils import human_duration, resolve_device, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--stage-a-epochs", type=int, default=None)
    parser.add_argument("--stage-b-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze-blocks", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=60)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage = "smoke" if args.smoke else "final"
    stage_a_epochs = (
        args.stage_a_epochs if args.stage_a_epochs is not None else (1 if args.smoke else 4)
    )
    stage_b_epochs = (
        args.stage_b_epochs if args.stage_b_epochs is not None else (2 if args.smoke else 12)
    )
    patience = 2 if args.smoke else args.patience
    seed_everything(args.seed)
    device = resolve_device(args.device)
    run = make_run_paths("mobilenet_v2", stage, args.models_dir, args.eval_dir)
    pretrained = not args.no_pretrained

    print(f"Device: {device}")
    print(f"Stage:  {stage}")
    print(f"Pretrained ImageNet weights: {pretrained}")

    samples = load_split_samples(
        args.datasets_dir,
        smoke=args.smoke,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    print_split_summary(samples)
    loaders = build_dataloaders(
        samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )

    model = build_model(
        "mobilenet_v2", NUM_CLASSES, pretrained=pretrained, dropout=args.dropout
    ).to(device)
    total_params = count_parameters(model)
    print(f"Parameters: {total_params:,}")

    weights = torch.tensor(
        compute_class_weights(samples["train"]), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    trainer = Trainer(model, criterion, device)

    started = time.perf_counter()

    freeze_backbone(model)
    trainable, total = trainable_parameter_count(model)
    print(f"Stage A trainable parameters: {trainable:,} / {total:,}")
    optimizer = AdamW(
        [p for p in model.classifier.parameters() if p.requires_grad],
        lr=args.lr_head,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(stage_a_epochs, 1))
    trainer.fit(
        loaders["train"],
        loaders["val"],
        epochs=stage_a_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        phase="frozen_backbone",
        patience=max(stage_a_epochs, 1),
        freeze_bn=True,
    )
    if trainer.best_state is not None:
        model.load_state_dict(trainer.best_state)

    unfreeze_last_blocks(model, count=args.unfreeze_blocks)
    trainable, total = trainable_parameter_count(model)
    print(f"Stage B trainable parameters: {trainable:,} / {total:,}")
    optimizer = AdamW(
        make_parameter_groups(model, args.lr_head, args.lr_backbone),
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    trainer.fit(
        loaders["train"],
        loaders["val"],
        epochs=stage_b_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        phase="fine_tune",
        patience=patience,
        freeze_bn=True,
        start_epoch=len(trainer.history),
    )
    duration = time.perf_counter() - started

    trainable_final, total_final = trainable_parameter_count(model)
    config = {
        "arch": "mobilenet_v2",
        "stage": stage,
        "pretrained": pretrained,
        "stage_a_epochs": stage_a_epochs,
        "stage_b_epochs": stage_b_epochs,
        "unfreeze_blocks": args.unfreeze_blocks,
        "batch_size": args.batch_size,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR -> ReduceLROnPlateau",
        "label_smoothing": 0.1,
        "class_weights": dict(
            zip(CLASS_NAMES, compute_class_weights(samples["train"]))
        ),
        "seed": args.seed,
        "device": str(device),
        "total_parameters": total_final,
        "trainable_parameters": trainable_final,
        "dataset": dataset_meta()["dataset"],
        "torch_version": torch.__version__,
        "samples": {
            split: {
                "total": len(items),
                "per_class": {
                    label: sum(1 for sample in items if sample.label == label)
                    for label in CLASS_NAMES
                },
            }
            for split, items in samples.items()
        },
    }
    write_class_names(CLASS_NAMES_PATH)
    save_run(
        model=model,
        trainer=trainer,
        run=run,
        config=config,
        split_digest=split_hash(args.datasets_dir),
        duration_seconds=duration,
    )
    print(f"Training duration: {human_duration(duration)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
