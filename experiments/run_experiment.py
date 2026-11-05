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
from spikeattn.evaluation.model_evaluator import CLASS_NAMES, ModelEvaluator
from spikeattn.evaluation.results_store import ResultsStore
from spikeattn.visualization.performance_plots import plot_confusion_matrix, plot_evaluation_comparisons
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

    # Validate lengths and row alignment
    if len(signals) != len(labels) or len(signals) != len(meta_df):
        raise ValueError(
            f"Dataset length mismatch in {data_dir}: signals={len(signals)}, "
            f"labels={len(labels)}, metadata={len(meta_df)}"
        )

    # Establish canonical ground-truth label column 'y_true' in meta_df
    if "y_true" not in meta_df.columns:
        if "class_id" in meta_df.columns:
            meta_df["y_true"] = meta_df["class_id"].to_numpy(dtype=np.int64)
        elif "label" in meta_df.columns:
            meta_df["y_true"] = meta_df["label"].to_numpy(dtype=np.int64)
        else:
            meta_df["y_true"] = labels.astype(np.int64)

    # Validate that meta_df['y_true'] matches labels
    if not np.array_equal(meta_df["y_true"].to_numpy(dtype=np.int64), labels):
        raise ValueError("Ground-truth labels in meta_df do not match labels in .npz archive!")

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
        raise RuntimeError(
            "CUDA is required by configs/experiment.yaml but is unavailable in this Python environment. "
            "Install a CUDA-enabled PyTorch build; refusing to silently run on CPU."
        )

    models_to_run = selected_models or list(MODEL_MAP.keys())

    run_dir = results_root / run_id
    store = ResultsStore(run_dir)

    signals, labels, meta_df = load_processed_dataset(data_dir)
    splits_dir = data_dir / "splits"

    logger.info("Starting experiment %s with models=%s, seeds=%s, splits=%d, device=%s",
                run_id, models_to_run, seeds, n_splits, device)

    trainer = Trainer(device=device)
    fold_status: list[dict[str, Any]] = [
        {"seed": int(seed), "fold": int(fold), "model": model, "status": "queued", "reason": ""}
        for seed in seeds
        for fold in (range(n_splits) if max_folds is None else range(min(n_splits, max_folds)))
        for model in models_to_run
    ]
    status_path = run_dir / "fold_status.csv"
    def set_fold_status(seed: int, fold: int, model: str, status: str, reason: str = "") -> None:
        for entry in fold_status:
            if (entry["seed"], entry["fold"], entry["model"]) == (int(seed), int(fold), model):
                entry.update(status=status, reason=reason)
                break
        pd.DataFrame(fold_status).to_csv(status_path, index=False)
    pd.DataFrame(fold_status).to_csv(status_path, index=False)
    curve_payloads: dict[str, dict[str, Any]] = {"validation": {}, "test": {}}

    # Main CV loops
    for seed in seeds:
        folds_to_run = range(n_splits) if max_folds is None else range(min(n_splits, max_folds))
        for fold in folds_to_run:
            split_file = splits_dir / f"seed_{seed}_fold_{fold}.json"
            if not split_file.exists():
                for model_name in models_to_run:
                    set_fold_status(seed, fold, model_name, "failed", f"Missing split: {split_file}")
                logger.error("Required split file missing: %s", split_file)
                continue

            try:
                with open(split_file, encoding="utf-8") as f:
                    split_info = json.load(f)
            except Exception as exc:
                for model_name in models_to_run:
                    set_fold_status(seed, fold, model_name, "failed", f"Invalid split metadata: {exc}")
                logger.exception("Could not load split metadata %s", split_file)
                continue

            train_pgs = set(split_info["train_patient_group_ids"])
            val_pgs = set(split_info["val_patient_group_ids"])
            test_pgs = set(split_info["test_patient_group_ids"])
            all_groups = set(meta_df["patient_group_id"].astype(str))
            overlap = (train_pgs & val_pgs) | (train_pgs & test_pgs) | (val_pgs & test_pgs)
            missing_groups = all_groups - (train_pgs | val_pgs | test_pgs)
            unknown_groups = (train_pgs | val_pgs | test_pgs) - all_groups
            if overlap or missing_groups or unknown_groups:
                reason = f"Invalid patient partition: overlap={sorted(overlap)}, missing={sorted(missing_groups)}, unknown={sorted(unknown_groups)}"
                for model_name in models_to_run:
                    set_fold_status(seed, fold, model_name, "failed", reason)
                logger.error("%s (seed=%s fold=%s)", reason, seed, fold)
                continue
            (run_dir / "splits").mkdir(exist_ok=True)
            (run_dir / "splits" / f"seed_{seed}_fold_{fold}.json").write_text(
                json.dumps(split_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            # Map to row indices in meta_df
            train_mask = meta_df["patient_group_id"].isin(train_pgs).to_numpy()
            val_mask = meta_df["patient_group_id"].isin(val_pgs).to_numpy()
            test_mask = meta_df["patient_group_id"].isin(test_pgs).to_numpy()

            X_train, y_train = signals[train_mask], labels[train_mask]
            X_val, y_val = signals[val_mask], labels[val_mask]
            X_test, y_test = signals[test_mask], labels[test_mask]
            test_meta = meta_df[test_mask].reset_index(drop=True)
            val_meta = meta_df[val_mask].reset_index(drop=True)

            if not np.array_equal(test_meta["y_true"].to_numpy(dtype=np.int64), y_test):
                raise ValueError("Alignment error: test_meta['y_true'] does not match y_test!")

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
                    set_fold_status(seed, fold, model_name, "completed", "resumed from saved test predictions")
                    continue

                model_device = "cpu" if model_name == "svm" else device
                logger.info("Training %s on seed=%d fold=%d device=%s (train=%d, val=%d, test=%d)",
                            model_name, seed, fold, model_device, len(X_train), len(X_val), len(X_test))

                try:
                    set_fold_status(seed, fold, model_name, "running")
                    model_cls = MODEL_MAP[model_name]
                    model = model_cls()

                    epochs = epochs_override or (15 if model_name != "svm" else 1)

                    if model_name == "svm":
                        model.fit(X_train, y_train)
                    else:
                        trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader,
                                    epochs=epochs, lr=1e-3, patience=4, class_weights=class_counts)

                # Inference on test set
                    validation_bundle = Predictor.predict(
                        model=model, signals=X_val, metadata=val_meta, seed=seed, fold=fold,
                        batch_size=batch_size, device=model_device, y_true=y_val)
                    store.save_predictions(validation_bundle, split="validation")
                    v_summary, v_class, v_cm, v_curves = ModelEvaluator.compute_classification(validation_bundle)
                    store.save_classification_artifacts("validation", v_summary, v_class, v_cm, v_curves)
                    curve_payloads["validation"][model_name] = v_curves
                    plot_evaluation_comparisons(pd.DataFrame([v_summary]), v_class, {model_name: v_curves},
                        run_dir / "visualizations" / "validation" / f"seed_{seed}_fold_{fold}" / model_name, "validation")

                    bundle = Predictor.predict(model=model, signals=X_test, metadata=test_meta,
                        seed=seed, fold=fold, batch_size=batch_size, device=model_device, y_true=y_test)
                    store.save_predictions(bundle, split="test")

                # Save hierarchical metrics
                    patient_metrics = ModelEvaluator.compute_per_patient(bundle)
                    store.save_per_patient_metrics(patient_metrics)

                    fold_metrics = ModelEvaluator.compute_per_fold(bundle)
                    store.save_per_fold_metrics(fold_metrics)
                    t_summary, t_class, t_cm, t_curves = ModelEvaluator.compute_classification(bundle)
                    store.save_classification_artifacts("test", t_summary, t_class, t_cm, t_curves)
                    curve_payloads["test"][model_name] = t_curves
                    vis_dir = run_dir / "visualizations" / "test" / f"seed_{seed}_fold_{fold}"
                    plot_evaluation_comparisons(pd.DataFrame([t_summary]), t_class, {model_name: t_curves},
                        vis_dir / model_name, "test")
                    for normalize, suffix in ((False, "raw"), (True, "normalized")):
                        plot_confusion_matrix(np.asarray(t_cm[CLASS_NAMES].values),
                            vis_dir / f"{model_name}_confusion_{suffix}.png", normalize=normalize)

                # Profile hardware efficiency
                    sample_in = X_test[: min(10, len(X_test))]
                    eff_df = EfficiencyProfiler.profile_model(model=model, sample_input=sample_in,
                        seed=seed, fold=fold, device=model_device)
                    store.save_measurements(eff_df)
                    set_fold_status(seed, fold, model_name, "completed")
                except Exception as exc:
                    logger.exception("Model fold failed: model=%s seed=%s fold=%s", model_name, seed, fold)
                    set_fold_status(seed, fold, model_name, "failed", f"{type(exc).__name__}: {exc}")

    logger.info("Cross-validation training loops finished. Running aggregations and analysis...")

    # Aggregate all predictions
    pd.DataFrame(fold_status).to_csv(status_path, index=False)
    for split in ("validation", "test"):
        metrics_path = run_dir / "metrics" / f"{split}_metrics.csv"
        class_path = run_dir / "metrics" / f"{split}_per_class.csv"
        if metrics_path.exists() and class_path.exists():
            metrics_frame, classes_frame = pd.read_csv(metrics_path), pd.read_csv(class_path)
            plot_evaluation_comparisons(metrics_frame, classes_frame, curve_payloads[split],
                                        run_dir / "visualizations" / split / "comparative", split)
    pred_files = list((store.predictions_dir / "test").glob("*.csv"))
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
        pooled_df.to_csv(run_dir / "metrics" / "pooled_oof.csv", index=False)

    # Execute statistical analysis and report generation
    try:
        analyzer = ExperimentAnalyzer(run_dir, analysis_cfg)
        bundle_res = analyzer.run()
        report_builder = ReportBuilder()
        report_path = run_dir / "report.md"
        val_path, test_path = run_dir / "metrics" / "validation_metrics.csv", run_dir / "metrics" / "test_metrics.csv"
        val_rows = pd.read_csv(val_path).replace({np.nan: None}).to_dict(orient="records") if val_path.exists() else []
        test_rows = pd.read_csv(test_path).replace({np.nan: None}).to_dict(orient="records") if test_path.exists() else []
        report_builder.build_report(bundle_res, report_path, run_id=run_id,
                                    validation_metrics=val_rows, test_metrics=test_rows)
        logger.info("Report generated successfully at %s", report_path)
    except Exception as exc:
        logger.warning("Statistical analysis / reporting encountered error: %s", exc)

    if any(row.get("status") == "failed" for row in fold_status):
        raise RuntimeError(f"One or more requested model folds failed; see {status_path}")
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
