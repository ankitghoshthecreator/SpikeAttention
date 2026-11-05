"""PyTorch training harness with validation-based early stopping and balanced accuracy tracking."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikeattn.statistics.classification_metrics import ClassificationMetrics
from spikeattn.training.losses import ClassWeightedCrossEntropy

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitors a validation metric and triggers early termination when improvement halts."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_state: dict[str, Any] | None = None
        self.early_stop = False

    def step(self, score: float, model: nn.Module) -> bool:
        """Update tracker with the latest epoch score.

        Returns:
            True if score improved and model checkpoint was saved; False otherwise.
        """
        if self.mode == "max":
            improved = self.best_score is None or score > self.best_score + self.min_delta
        else:
            improved = self.best_score is None or score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def restore_best(self, model: nn.Module) -> None:
        """Load best recorded weights into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    """Robust training coordinator supporting gradient clipping, class weighting, and early stopping.

    Args:
        device: Target execution device ('cuda' or 'cpu').
        criterion: Optional custom loss criterion; defaults to ClassWeightedCrossEntropy.
        grad_clip: Maximum gradient norm (default 1.0).
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        criterion: nn.Module | None = None,
        grad_clip: float = 1.0,
    ) -> None:
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; refusing to silently run on CPU.")
        self.device = requested_device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.grad_clip = grad_clip

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 25,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        class_weights: dict[int, int] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Train neural network model across specified epochs with validation monitoring.

        Args:
            model: PyTorch model exposing forward / forward_logits.
            train_loader: DataLoader yielding (signals, labels).
            val_loader: Optional validation DataLoader for early stopping.
            epochs: Maximum training epochs.
            lr: Initial learning rate for AdamW.
            weight_decay: L2 penalty for AdamW.
            patience: Early stopping patience epochs.
            class_weights: Class counts or weights for weighted cross-entropy.

        Returns:
            Dictionary with training history: train_loss, val_loss, val_balanced_acc, best_epoch.
        """
        model = model.to(self.device)

        if class_weights is not None:
            criterion = ClassWeightedCrossEntropy(class_weights).to(self.device)
        else:
            criterion = self.criterion.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        early_stopping = EarlyStopping(patience=patience, mode="max")

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_balanced_acc": [],
            "val_accuracy": [],
        }
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            # 1. Training Phase
            model.train()
            running_loss = 0.0
            total_train_samples = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                if hasattr(model, "forward_logits"):
                    logits = model.forward_logits(batch_x)
                else:
                    logits = model(batch_x)

                loss = criterion(logits, batch_y)
                loss.backward()

                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                optimizer.step()

                batch_size = len(batch_y)
                running_loss += float(loss.item()) * batch_size
                total_train_samples += batch_size

            epoch_train_loss = running_loss / max(1, total_train_samples)
            history["train_loss"].append(epoch_train_loss)
            scheduler.step()

            # 2. Validation Phase
            if val_loader is not None:
                val_metrics = self.evaluate(model, val_loader, criterion)
                history["val_loss"].append(val_metrics["loss"])
                history["val_balanced_acc"].append(val_metrics["balanced_accuracy"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                improved = early_stopping.step(val_metrics["balanced_accuracy"], model)
                if improved:
                    best_epoch = epoch

                logger.info(
                    "Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Val BalAcc: %.4f%s",
                    epoch,
                    epochs,
                    epoch_train_loss,
                    val_metrics["loss"],
                    val_metrics["balanced_accuracy"],
                    " (Best)" if improved else "",
                )

                if early_stopping.early_stop:
                    logger.info("Early stopping triggered at epoch %d (patience=%d)", epoch, patience)
                    break
            else:
                logger.info("Epoch %d/%d - Train Loss: %.4f", epoch, epochs, epoch_train_loss)

        # Restore optimal checkpoint if validation was monitored
        if val_loader is not None and early_stopping.best_state is not None:
            early_stopping.restore_best(model)

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_balanced_acc": early_stopping.best_score or 0.0,
        }

    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
    ) -> dict[str, float]:
        """Compute evaluation metrics over a DataLoader."""
        model.eval()
        crit = criterion or self.criterion
        crit = crit.to(self.device)

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if hasattr(model, "forward_logits"):
                    logits = model.forward_logits(batch_x)
                else:
                    logits = model(batch_x)

                loss = crit(logits, batch_y)
                total_loss += float(loss.item()) * len(batch_y)
                total_samples += len(batch_y)

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch_y.cpu().numpy().tolist())

        y_true = np.asarray(all_targets, dtype=np.int64)
        y_pred = np.asarray(all_preds, dtype=np.int64)

        bacc = ClassificationMetrics.balanced_accuracy(y_true, y_pred, n_classes=5)
        acc = ClassificationMetrics.accuracy(y_true, y_pred)
        mean_loss = total_loss / max(1, total_samples)

        return {
            "loss": mean_loss,
            "balanced_accuracy": bacc,
            "accuracy": acc,
        }
