"""Shared training loop with macro-F1 model selection and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.history = history if history is not None else []
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_phase = ""
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_summary: dict[str, Any] | None = None

    def _forward(self, images: torch.Tensor):
        if self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=True):
                return self.model(images)
        return self.model(images)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        *,
        training: bool,
        freeze_bn: bool,
        grad_clip: float | None,
        scaler: torch.amp.GradScaler | None,
    ) -> EpochMetrics:
        self.model.train(training)
        if training and freeze_bn:
            self.model.features.eval()

        total_loss = 0.0
        total_samples = 0
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = labels.size(0)

            with torch.set_grad_enabled(training):
                outputs = self._forward(images)
                loss = self.criterion(outputs, labels)

                if training:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), grad_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), grad_clip
                            )
                        optimizer.step()

            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            predictions.append(outputs.detach().argmax(dim=1).cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(targets) if targets else np.array([])
        y_pred = np.concatenate(predictions) if predictions else np.array([])
        return EpochMetrics(
            loss=total_loss / max(total_samples, 1),
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        phase: str = "train",
        patience: int = 5,
        grad_clip: float | None = 1.0,
        freeze_bn: bool = False,
        start_epoch: int = 0,
    ) -> list[dict[str, Any]]:
        scaler = None
        if self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda", enabled=True)

        stale_epochs = 0
        for offset in range(epochs):
            epoch = start_epoch + offset
            train_metrics = self._run_epoch(
                train_loader,
                optimizer,
                training=True,
                freeze_bn=freeze_bn,
                grad_clip=grad_clip,
                scaler=scaler,
            )
            val_metrics = self._run_epoch(
                val_loader,
                None,
                training=False,
                freeze_bn=False,
                grad_clip=None,
                scaler=None,
            )

            learning_rate = optimizer.param_groups[0]["lr"]
            record = {
                "phase": phase,
                "epoch": epoch + 1,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "train_macro_f1": train_metrics.macro_f1,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
                "learning_rate": learning_rate,
            }
            self.history.append(record)

            improved = val_metrics.macro_f1 > self.best_metric + 1e-6
            if improved:
                self.best_metric = val_metrics.macro_f1
                self.best_epoch = epoch + 1
                self.best_phase = phase
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.best_summary = dict(record)
                stale_epochs = 0
            else:
                stale_epochs += 1

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.macro_f1)
            elif scheduler is not None:
                scheduler.step()

            print(
                f"[{phase}] epoch {epoch + 1}/{start_epoch + epochs} "
                f"train_loss={train_metrics.loss:.4f} "
                f"val_loss={val_metrics.loss:.4f} "
                f"val_acc={val_metrics.accuracy:.4f} "
                f"val_macro_f1={val_metrics.macro_f1:.4f}",
                flush=True,
            )

            if stale_epochs >= patience:
                print(
                    f"[{phase}] early stopping after {stale_epochs} stale epochs.",
                    flush=True,
                )
                break

        return self.history

    def restore_best(self) -> None:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
