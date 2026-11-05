"""ResultsStore: persistent storage manager ensuring strict compliance with Results Contract v1.0.0."""

from __future__ import annotations

import json
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
        pred_file = self.predictions_dir / f"{model}_seed{seed}.csv"
        if not pred_file.exists():
            return False
        try:
            df = pd.read_csv(pred_file, usecols=["fold"])
            return fold in df["fold"].values
        except Exception:
            return False

    def save_predictions(self, bundle: PredictionBundle) -> Path:
        """Save out-of-fold prediction records into predictions/<model>_seed<seed>.csv."""
        pred_file = self.predictions_dir / f"{bundle.model_name}_seed{bundle.seed}.csv"
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

    def _append_or_write(self, path: Path, df: pd.DataFrame, dedup_cols: list[str]) -> None:
        """Write DataFrame or append without duplicate primary keys."""
        if path.exists():
            existing = pd.read_csv(path)
            # Filter out overlapping records
            mask = pd.Series([True] * len(existing))
            for col in dedup_cols:
                if col in existing.columns and col in df.columns:
                    mask = mask & (~existing[col].isin(df[col].unique()))
            combined = pd.concat([existing[mask], df], ignore_index=True)
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
        """Append or write efficiency/measurements.csv."""
        out = self.efficiency_dir / "measurements.csv"
        if out.exists():
            existing = pd.read_csv(out)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_csv(out, index=False)
        else:
            df.to_csv(out, index=False)
        return out

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        """Write manifest.json."""
        out = self.run_dir / "manifest.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return out
