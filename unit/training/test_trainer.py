"""Unit tests for Trainer, EarlyStopping, and ClassWeightedCrossEntropy."""

import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spikeattn.training.losses import ClassWeightedCrossEntropy
from spikeattn.training.trainer import EarlyStopping, Trainer


class _ToyClassifier(nn.Module):
    """Tiny linear model for fast unit testing of the training harness."""

    def __init__(self, in_features: int = 10, n_classes: int = 5) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestClassWeightedCrossEntropy:
    def test_weights_computation(self):
        # 10 samples of class 0, 1 of class 1
        counts = {0: 10, 1: 1, 2: 2, 3: 1, 4: 1}
        weights = ClassWeightedCrossEntropy.compute_weights(counts, n_classes=5)
        assert len(weights) == 5
        # Class 1 should have much higher weight than class 0
        assert weights[1] > weights[0]

    def test_loss_evaluation(self):
        loss_fn = ClassWeightedCrossEntropy(class_counts={0: 10, 1: 2, 2: 2, 3: 1, 4: 1})
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0.0


class TestEarlyStopping:
    def test_improves_and_resets(self):
        es = EarlyStopping(patience=3, mode="max")
        model = _ToyClassifier()
        assert es.step(0.5, model) is True
        assert es.counter == 0
        assert es.early_stop is False

    def test_triggers_early_stopping(self):
        es = EarlyStopping(patience=2, mode="max")
        model = _ToyClassifier()
        es.step(0.8, model)
        es.step(0.7, model)  # counter = 1
        assert es.early_stop is False
        es.step(0.6, model)  # counter = 2 >= patience
        assert es.early_stop is True

    def test_restore_best(self):
        es = EarlyStopping(patience=3, mode="max")
        model = _ToyClassifier()

        # Save initial weights
        es.step(0.9, model)

        # Mutate model weights
        with torch.no_grad():
            model.linear.weight.fill_(100.0)

        # Restore
        es.restore_best(model)
        assert not torch.allclose(model.linear.weight, torch.full_like(model.linear.weight, 100.0))


class TestTrainer:
    def _create_synthetic_data(self):
        X = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        return loader

    def test_parameters_update_after_fit(self):
        loader = self._create_synthetic_data()
        model = _ToyClassifier()
        initial_params = copy.deepcopy(model.linear.weight.data)

        trainer = Trainer(device="cpu")
        trainer.fit(model, train_loader=loader, epochs=2, lr=0.1)

        # Weights must have updated
        assert not torch.allclose(model.linear.weight.data, initial_params)

    def test_early_stopping_triggered_with_val_loader(self):
        train_loader = self._create_synthetic_data()
        val_loader = self._create_synthetic_data()
        model = _ToyClassifier()

        trainer = Trainer(device="cpu")
        history = trainer.fit(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=20,
            patience=2,
        )

        assert "history" in history
        # Should stop well before 20 epochs because of patience=2
        assert len(history["history"]["train_loss"]) < 20
