"""ResultsStore: persistent storage manager ensuring strict compliance with Results Contract v1.0.0."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from spikeattn.prediction.predictor import PredictionBundle


class ResultsStore:
    """Manages the disk persistence of model predictions, multi-level metrics, and hardware measurements.

    Directory Layout (Results Contract v1.0.0):
        results/<run_id>/
            predictions/
                <model>_seed<seed>.csv
            metrics/
                per_patient.csv
                per_fold.csv
                per_seed.csv
                pooled.csv
            efficiency/
                measurements.csv
            manifest.json
    """

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.predictions_dir = self.run_dir / "predictions"
        self.metrics_dir = self.run_dir / "metrics"
        self.efficiency_dir = self.run_dir / "efficiency"

        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.efficiency_dir.mkdir(parents=True, exist_ok=True)

    def is_fold_completed(self, model: str, seed: int, fold: int) -> bool:
        """Check if predictions for the given (model, seed, fold) already exist."""
        pred_file = self.predictions_dir / "test" / f"{model}_seed{seed}.csv"
        if not pred_file.exists():
            legacy_file = self.predictions_dir / f"{model}_seed{seed}.csv"
            pred_file = legacy_file if legacy_file.exists() else pred_file
        if not pred_file.exists():
            return False
        try:
            df = pd.read_csv(pred_file, usecols=["fold"])
            return fold in df["fold"].values
        except Exception:
            return False

    def save_predictions(self, bundle: PredictionBundle, split: str = "test") -> Path:
        """Save validation or held-out test predictions without mixing split roles."""
        if split not in {"validation", "test"}:
            raise ValueError(f"Unknown prediction split: {split}")
        split_dir = self.predictions_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        pred_file = split_dir / f"{bundle.model_name}_seed{bundle.seed}.csv"
        new_df = bundle.to_dataframe()

        if pred_file.exists():
            existing = pd.read_csv(pred_file)
            # Remove any previous entries for this exact fold to prevent duplication
            existing = existing[existing["fold"] != bundle.fold]
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(pred_file, index=False)
        else:
            new_df.to_csv(pred_file, index=False)

        return pred_file

    def save_classification_artifacts(self, split: str, summary: dict[str, Any], per_class: pd.DataFrame,
                                      confusion: pd.DataFrame, curves: dict[str, Any]) -> None:
        """Persist split-scoped metrics, class details, confusion matrices and probability curves."""
        if split not in {"validation", "test"}:
            raise ValueError(f"Unknown evaluation split: {split}")
        self._append_or_write(self.metrics_dir / f"{split}_metrics.csv", pd.DataFrame([summary]),
                              ["model", "seed", "fold"])
        classes = per_class.copy()
        for key in ("model", "seed", "fold"):
            classes[key] = summary[key]
        self._append_or_write(self.metrics_dir / f"{split}_per_class.csv", classes,
                              ["model", "seed", "fold", "class"])
        conf = confusion.copy()
        conf["model"], conf["seed"], conf["fold"] = summary["model"], summary["seed"], summary["fold"]
        self._append_or_write(self.metrics_dir / f"{split}_confusion_matrices.csv", conf,
                              ["model", "seed", "fold", "true_class"])
        curve_dir = self.run_dir / "probability_metrics" / split
        curve_dir.mkdir(parents=True, exist_ok=True)
        path = curve_dir / f"{summary['model']}_seed{summary['seed']}_fold{summary['fold']}.json"
        def finite_values(value):
            if isinstance(value, dict): return {k: finite_values(v) for k, v in value.items()}
            if isinstance(value, list): return [finite_values(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value): return None
            return value
        with path.open("w", encoding="utf-8") as f:
            json.dump(finite_values(curves), f, indent=2, allow_nan=False)

    def _append_or_write(self, path: Path, df: pd.DataFrame, dedup_cols: list[str]) -> None:
        """Write DataFrame or append without duplicate primary keys.

        Deduplication removes existing rows whose *composite* key (all dedup_cols
        together) matches any row in the incoming DataFrame.  Each column is compared
        independently only within the context of the full key — this avoids the
        mistake of treating each column as an independent filter, which would remove
        rows from other models when they share the same seed/fold values.
        """
        if path.exists():
            existing = pd.read_csv(path)
            # Build a set of composite key tuples from the incoming DataFrame
            valid_dedup_cols = [c for c in dedup_cols if c in existing.columns and c in df.columns]
            if valid_dedup_cols:
                incoming_keys: set[tuple] = set(
                    df[valid_dedup_cols].itertuples(index=False, name=None)
                )
                # Keep existing rows whose composite key is NOT in the incoming set
                keep_mask = ~existing[valid_dedup_cols].apply(
                    lambda row: tuple(row) in incoming_keys, axis=1
                )
                combined = pd.concat([existing[keep_mask], df], ignore_index=True)
            else:
                combined = pd.concat([existing, df], ignore_index=True)
            combined.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

    def save_per_patient_metrics(self, df: pd.DataFrame) -> Path:
        """Append or write metrics/per_patient.csv."""
        out = self.metrics_dir / "per_patient.csv"
        self._append_or_write(out, df, dedup_cols=["model", "seed", "fold"])
        return out

    def save_per_fold_metrics(self, df: pd.DataFrame) -> Path:
        """Append or write metrics/per_fold.csv."""
        out = self.metrics_dir / "per_fold.csv"
        self._append_or_write(out, df, dedup_cols=["model", "seed", "fold"])
        return out

    def save_per_seed_metrics(self, df: pd.DataFrame) -> Path:
        """Append or write metrics/per_seed.csv."""
        out = self.metrics_dir / "per_seed.csv"
        self._append_or_write(out, df, dedup_cols=["model", "seed"])
        return out

    def save_pooled_metrics(self, df: pd.DataFrame) -> Path:
        """Write metrics/pooled.csv."""
        out = self.metrics_dir / "pooled.csv"
        df.to_csv(out, index=False)
        return out

    def save_measurements(self, df: pd.DataFrame) -> Path:
        """Append or write efficiency/measurements.csv without duplicating model/seed/fold/metric combos."""
        out = self.efficiency_dir / "measurements.csv"
        self._append_or_write(out, df, dedup_cols=["model", "seed", "fold", "metric"])
        return out

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        """Write manifest.json."""
        out = self.run_dir / "manifest.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return out
