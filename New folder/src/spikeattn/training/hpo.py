"""Hyperparameter tuning with equal budget per model on validation patient splits."""

from __future__ import annotations

from collections.abc import Callable
import copy
import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spikeattn.training.trainer import Trainer

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Explores hyperparameter configurations evaluated strictly on the validation patient split.

    Avoids test-set data snooping by scoring trials purely on validation balanced accuracy.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.trainer = Trainer(device=device)

    def tune(
        self,
        model_factory: Callable[..., Any],
        param_grid: dict[str, list[Any]],
        train_signals: np.ndarray,
        train_labels: np.ndarray,
        val_signals: np.ndarray,
        val_labels: np.ndarray,
        n_trials: int = 5,
        epochs_per_trial: int = 10,
        batch_size: int = 128,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute random or grid search across candidate parameter configurations.

        Args:
            model_factory: Function returning an uninitialized model given **kwargs.
            param_grid: Dictionary of parameter names to lists of candidate values.
            train_signals: Training array [N_train, 2, 360].
            train_labels: Training labels [N_train].
            val_signals: Validation array [N_val, 2, 360].
            val_labels: Validation labels [N_val].
            n_trials: Number of trial configurations to evaluate.
            epochs_per_trial: Epoch budget allocated per trial.
            batch_size: Batch size for DataLoader.
            seed: Random seed for configuration sampling.

        Returns:
            Dictionary with best_params, best_val_score, and trial_records.
        """
        rng = np.random.default_rng(seed)

        # Build candidate configurations
        keys = list(param_grid.keys())
        all_combinations = []
        if keys:
            import itertools
            for vals in itertools.product(*(param_grid[k] for k in keys)):
                all_combinations.append(dict(zip(keys, vals, strict=True)))

        if not all_combinations:
            all_combinations = [{}]

        # Sample at most n_trials
        if len(all_combinations) > n_trials:
            sampled_indices = rng.choice(len(all_combinations), size=n_trials, replace=False)
            candidate_configs = [all_combinations[i] for i in sampled_indices]
        else:
            candidate_configs = all_combinations

        train_ds = TensorDataset(
            torch.from_numpy(train_signals).float(),
            torch.from_numpy(train_labels).long(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(val_signals).float(),
            torch.from_numpy(val_labels).long(),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_score = -1.0
        best_params = {}
        trial_records = []

        for idx, cfg in enumerate(candidate_configs, 1):
            logger.info("Trial %d/%d with config: %s", idx, len(candidate_configs), cfg)
            model = model_factory(**cfg)

            lr = cfg.get("lr", 1e-3)
            patience = cfg.get("patience", 3)

            res = self.trainer.fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_trial,
                lr=lr,
                patience=patience,
            )

            score = res.get("best_val_balanced_acc", 0.0)
            trial_records.append({
                "trial": idx,
                "params": cfg,
                "best_val_balanced_acc": score,
                "best_epoch": res.get("best_epoch", 0),
            })

            if score > best_score:
                best_score = score
                best_params = copy.deepcopy(cfg)

        logger.info("HPO complete: Best BalAcc=%.4f with %s", best_score, best_params)
        return {
            "best_params": best_params,
            "best_val_score": best_score,
            "trials": trial_records,
        }
