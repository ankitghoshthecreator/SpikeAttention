"""Resumable experiment runner executing patient-grouped multi-seed cross-validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from spikeattn.evaluation.efficiency_profiler import EfficiencyProfiler
from spikeattn.evaluation.model_evaluator import ModelEvaluator
from spikeattn.evaluation.results_store import ResultsStore
from spikeattn.models.baselines.cnn_baseline import CNNBaseline
from spikeattn.models.baselines.rnn_baseline import RNNBaseline
from spikeattn.models.baselines.svm_baseline import SVMBaseline
from spikeattn.models.snn.plain_snn import PlainSNN
from spikeattn.models.snn.st_snn import SpikeAttentionSNN
from spikeattn.prediction.predictor import Predictor
from spikeattn.reporting.report_builder import ReportBuilder
from spikeattn.statistics.experiment_analyzer import ExperimentAnalyzer
from spikeattn.training.trainer import Trainer
from spikeattn.utils.config import load_config
from spikeattn.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "svm": SVMBaseline,
    "cnn": CNNBaseline,
    "rnn": RNNBaseline,
    "plain_snn": PlainSNN,
    "st_snn": SpikeAttentionSNN,
}


def load_processed_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Find and load preprocessed beats .npz and beats_meta.csv."""
    npz_files = list(data_dir.glob("beats_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No processed beats_*.npz file found in {data_dir}. Run 'main.py data preprocess' first.")

    npz_path = npz_files[0]
    meta_path = data_dir / "beats_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file {meta_path} not found.")

    logger.info("Loading processed dataset from %s", npz_path)
    data = np.load(npz_path)
    signals = data["signals"]  # [N, 2, 360]
    labels = data["labels"]    # [N]
    meta_df = pd.read_csv(meta_path)
    return signals, labels, meta_df


def run_experiment(
    config_dir: Path = Path("configs"),
    data_dir: Path = Path("data/processed"),
    results_root: Path = Path("results"),
    run_id: str = "run_default",
    selected_models: list[str] | None = None,
    seeds_override: list[int] | None = None,
    max_folds: int | None = None,
    epochs_override: int | None = None,
    batch_size: int = 128,
) -> Path:
    """Execute complete resumable cross-validation experiment across models, seeds, and folds."""
    exp_cfg = load_config(config_dir / "experiment.yaml")
    ds_cfg = load_config(config_dir / "dataset.yaml")
    analysis_cfg = load_config(config_dir / "analysis.yaml")

    seeds = seeds_override or exp_cfg.get("seeds", [0, 1, 2, 3, 4])
    n_splits = exp_cfg.get("n_splits", 5)
    device = exp_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    models_to_run = selected_models or list(MODEL_MAP.keys())

    run_dir = results_root / run_id
    store = ResultsStore(run_dir)

    signals, labels, meta_df = load_processed_dataset(data_dir)
    splits_dir = data_dir / "splits"

    logger.info("Starting experiment %s with models=%s, seeds=%s, splits=%d, device=%s",
                run_id, models_to_run, seeds, n_splits, device)

    trainer = Trainer(device=device)

    # Main CV loops
    for seed in seeds:
        folds_to_run = range(n_splits) if max_folds is None else range(min(n_splits, max_folds))
        for fold in folds_to_run:
            split_file = splits_dir / f"seed_{seed}_fold_{fold}.json"
            if not split_file.exists():
                logger.warning("Split file %s missing, skipping.", split_file)
                continue

            with open(split_file, encoding="utf-8") as f:
                split_info = json.load(f)

            train_pgs = set(split_info["train_patient_group_ids"])
            val_pgs = set(split_info["val_patient_group_ids"])
            test_pgs = set(split_info["test_patient_group_ids"])

            # Map to row indices in meta_df
            train_mask = meta_df["patient_group_id"].isin(train_pgs).to_numpy()
            val_mask = meta_df["patient_group_id"].isin(val_pgs).to_numpy()
            test_mask = meta_df["patient_group_id"].isin(test_pgs).to_numpy()

            X_train, y_train = signals[train_mask], labels[train_mask]
            X_val, y_val = signals[val_mask], labels[val_mask]
            X_test, y_test = signals[test_mask], labels[test_mask]
            test_meta = meta_df[test_mask].reset_index(drop=True)

            # Class counts for weighted loss
            unique, counts = np.unique(y_train, return_counts=True)
            class_counts = dict(zip(unique.tolist(), counts.tolist(), strict=True))

            train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
            val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            for model_name in models_to_run:
                if store.is_fold_completed(model_name, seed, fold):
                    logger.info("Fold completed (skipping): %s seed=%d fold=%d", model_name, seed, fold)
                    continue

                logger.info("Training %s on seed=%d fold=%d (train=%d, val=%d, test=%d)",
                            model_name, seed, fold, len(X_train), len(X_val), len(X_test))

                model_cls = MODEL_MAP[model_name]
                model = model_cls()

                epochs = epochs_override or (15 if model_name != "svm" else 1)

                if model_name == "svm":
                    model.fit(X_train, y_train)
                else:
                    trainer.fit(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs,
                        lr=1e-3,
                        patience=4,
                        class_weights=class_counts,
                    )

                # Inference on test set
                bundle = Predictor.predict(
                    model=model,
                    signals=X_test,
                    metadata=test_meta,
                    seed=seed,
                    fold=fold,
                    batch_size=batch_size,
                    device=device,
                )
                store.save_predictions(bundle)

                # Save hierarchical metrics
                patient_metrics = ModelEvaluator.compute_per_patient(bundle)
                store.save_per_patient_metrics(patient_metrics)

                fold_metrics = ModelEvaluator.compute_per_fold(bundle)
                store.save_per_fold_metrics(fold_metrics)

                # Profile hardware efficiency
                sample_in = X_test[: min(10, len(X_test))]
                eff_df = EfficiencyProfiler.profile_model(
                    model=model,
                    sample_input=sample_in,
                    seed=seed,
                    fold=fold,
                    device=device,
                )
                store.save_measurements(eff_df)

    logger.info("Cross-validation training loops finished. Running aggregations and analysis...")

    # Aggregate all predictions
    pred_files = list(store.predictions_dir.glob("*.csv"))
    if pred_files:
        all_preds = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
        # Per-seed
        seed_dfs = []
        for (m, s), grp in all_preds.groupby(["model", "seed"]):
            seed_dfs.append(ModelEvaluator.compute_per_seed(grp))
        if seed_dfs:
            store.save_per_seed_metrics(pd.concat(seed_dfs, ignore_index=True))

        # Pooled
        pooled_df = ModelEvaluator.compute_pooled(all_preds)
        store.save_pooled_metrics(pooled_df)

    # Execute statistical analysis and report generation
    try:
        analyzer = ExperimentAnalyzer(run_dir, analysis_cfg)
        bundle_res = analyzer.run()
        report_builder = ReportBuilder()
        report_path = run_dir / "report.md"
        report_builder.build_report(bundle_res, report_path, run_id=run_id)
        logger.info("Report generated successfully at %s", report_path)
    except Exception as exc:
        logger.warning("Statistical analysis / reporting encountered error: %s", exc)

    return run_dir


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Run SPIKE-ATTENTION cross-validation experiments")
    parser.add_argument("--run-id", type=str, default="run_benchmark", help="Run identifier")
    parser.add_argument("--models", nargs="+", default=None, help="Models to train: svm cnn rnn plain_snn st_snn")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run")
    parser.add_argument("--max-folds", type=int, default=None, help="Maximum folds per seed")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch budget per model")
    args = parser.parse_args()

    run_experiment(
        run_id=args.run_id,
        selected_models=args.models,
        seeds_override=args.seeds,
        max_folds=args.max_folds,
        epochs_override=args.epochs,
    )
