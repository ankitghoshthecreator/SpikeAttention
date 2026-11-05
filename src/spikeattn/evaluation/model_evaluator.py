"""ModelEvaluator: aggregates out-of-fold predictions into per-patient, per-fold, per-seed, and pooled metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spikeattn.prediction.predictor import PredictionBundle
from spikeattn.statistics.classification_metrics import ClassificationMetrics


class ModelEvaluator:
    """Evaluates classification performance across the hierarchical levels mandated by Results Contract v1.0.0."""

    @staticmethod
    def compute_per_patient(bundle: PredictionBundle) -> pd.DataFrame:
        """Compute metrics aggregated at the patient group unit level.

        Returns DataFrame with columns:
            model, seed, fold, patient_group_id, n_beats, balanced_accuracy, accuracy
        """
        df = bundle.to_dataframe()
        rows = []

        for pg_id, group in df.groupby("patient_group_id"):
            yt = group["y_true"].to_numpy(dtype=np.int64)
            yp = group["y_pred"].to_numpy(dtype=np.int64)

            bacc = ClassificationMetrics.balanced_accuracy(yt, yp, n_classes=5)
            acc = ClassificationMetrics.accuracy(yt, yp)

            rows.append({
                "model": bundle.model_name,
                "seed": bundle.seed,
                "fold": bundle.fold,
                "patient_group_id": pg_id,
                "n_beats": int(len(yt)),
                "balanced_accuracy": float(bacc),
                "accuracy": float(acc),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def compute_per_fold(bundle: PredictionBundle) -> pd.DataFrame:
        """Compute summary metrics over the entire fold test set.

        Returns DataFrame with columns:
            model, seed, fold, n_patients, n_beats, accuracy, balanced_accuracy, macro_f1, weighted_f1
        """
        yt = np.asarray(bundle.y_true, dtype=np.int64)
        yp = np.asarray(bundle.y_pred, dtype=np.int64)
        cm = ClassificationMetrics.confusion_matrix(yt, yp, n_classes=5)

        row = {
            "model": bundle.model_name,
            "seed": bundle.seed,
            "fold": bundle.fold,
            "n_patients": int(len(set(bundle.patient_group_ids))),
            "n_beats": int(len(yt)),
            "accuracy": float(ClassificationMetrics.accuracy(yt, yp)),
            "balanced_accuracy": float(ClassificationMetrics.balanced_accuracy(yt, yp, n_classes=5)),
            "macro_f1": float(ClassificationMetrics.f1(cm, average="macro")),
            "weighted_f1": float(ClassificationMetrics.f1(cm, average="weighted")),
        }
        return pd.DataFrame([row])

    @staticmethod
    def compute_per_seed(df_seed_predictions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate all fold predictions for a single seed into a per-seed evaluation row.

        Returns DataFrame with columns:
            model, seed, n_patients, n_beats, accuracy, balanced_accuracy, macro_f1, weighted_f1
        """
        model = df_seed_predictions["model"].iloc[0]
        seed = int(df_seed_predictions["seed"].iloc[0])

        yt = df_seed_predictions["y_true"].to_numpy(dtype=np.int64)
        yp = df_seed_predictions["y_pred"].to_numpy(dtype=np.int64)
        cm = ClassificationMetrics.confusion_matrix(yt, yp, n_classes=5)

        row = {
            "model": model,
            "seed": seed,
            "n_patients": int(df_seed_predictions["patient_group_id"].nunique()),
            "n_beats": int(len(yt)),
            "accuracy": float(ClassificationMetrics.accuracy(yt, yp)),
            "balanced_accuracy": float(ClassificationMetrics.balanced_accuracy(yt, yp, n_classes=5)),
            "macro_f1": float(ClassificationMetrics.f1(cm, average="macro")),
            "weighted_f1": float(ClassificationMetrics.f1(cm, average="weighted")),
        }
        return pd.DataFrame([row])

    @staticmethod
    def compute_pooled(df_all_predictions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate predictions across all seeds and folds for each model.

        Returns DataFrame with columns:
            model, n_patients, n_beats, accuracy, balanced_accuracy, macro_f1, weighted_f1
        """
        rows = []
        for model, grp in df_all_predictions.groupby("model"):
            yt = grp["y_true"].to_numpy(dtype=np.int64)
            yp = grp["y_pred"].to_numpy(dtype=np.int64)
            cm = ClassificationMetrics.confusion_matrix(yt, yp, n_classes=5)

            rows.append({
                "model": model,
                "n_patients": int(grp["patient_group_id"].nunique()),
                "n_beats": int(len(yt)),
                "accuracy": float(ClassificationMetrics.accuracy(yt, yp)),
                "balanced_accuracy": float(ClassificationMetrics.balanced_accuracy(yt, yp, n_classes=5)),
                "macro_f1": float(ClassificationMetrics.f1(cm, average="macro")),
                "weighted_f1": float(ClassificationMetrics.f1(cm, average="weighted")),
            })

        return pd.DataFrame(rows)
