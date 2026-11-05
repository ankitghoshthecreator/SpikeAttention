"""Classification metrics for beat predictions and patient level summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from spikeattn.prediction.predictor import PredictionBundle
from spikeattn.statistics.classification_metrics import ClassificationMetrics

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


class ModelEvaluator:
    """Compute fixed-label (N,S,V,F,Q) evaluation summaries."""

    @staticmethod
    def compute_classification(bundle: PredictionBundle) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        y = np.asarray(bundle.y_true, dtype=np.int64)
        pred = np.asarray(bundle.y_pred, dtype=np.int64)
        probs = np.asarray(bundle.probabilities, dtype=np.float64)
        cm = ClassificationMetrics.confusion_matrix(y, pred, n_classes=5)
        per_class = ClassificationMetrics.report(y, pred, CLASS_NAMES, 5)
        class_rows = []
        curve_data: dict[str, Any] = {"averaging": "macro over classes with both positive and negative examples; absent/undefined classes excluded and listed", "classes": {}}
        aucs, aps = [], []
        for c, name in enumerate(CLASS_NAMES):
            yt = (y == c).astype(np.uint8)
            support = int(yt.sum())
            predicted_support = int((pred == c).sum())
            roc_auc = ap = np.nan
            if 0 < support < len(y):
                roc_auc = float(roc_auc_score(yt, probs[:, c]))
                ap = float(average_precision_score(yt, probs[:, c]))
                aucs.append(roc_auc)
                aps.append(ap)
                fpr, tpr, _ = roc_curve(yt, probs[:, c])
                precision, recall, _ = precision_recall_curve(yt, probs[:, c])
                roc_idx = np.linspace(0, len(fpr) - 1, min(1000, len(fpr)), dtype=int)
                pr_idx = np.linspace(0, len(recall) - 1, min(1000, len(recall)), dtype=int)
                curve_data["classes"][name] = {"status": "defined", "roc_auc_ovr": roc_auc, "average_precision": ap,
                    "fpr": fpr[roc_idx].tolist(), "tpr": tpr[roc_idx].tolist(),
                    "precision": precision[pr_idx].tolist(), "recall": recall[pr_idx].tolist(),
                    "curve_points_max": 1000, "source_probabilities": "split-scoped predictions CSV"}
            else:
                reason = "no positive examples" if support == 0 else "no negative examples"
                curve_data["classes"][name] = {"status": "undefined", "reason": reason}
            p, r, f = per_class[name]["precision"], per_class[name]["recall"], per_class[name]["f1"]
            class_rows.append({"model": bundle.model_name, "class": name, "precision": p, "recall": r, "f1": f, "support": support,
                               "predicted_support": predicted_support, "roc_auc_ovr": roc_auc,
                               "classification_metric_status": ("no_true_examples_and_zero_predicted_examples; zero_division=0" if support == 0 and predicted_support == 0 else
                                   "no_true_examples; recall=0 by zero_division=0" if support == 0 else
                                   "no_predicted_examples; precision=0 by zero_division=0" if predicted_support == 0 else "defined"),
                               "average_precision": ap, "probability_metric_status": curve_data["classes"][name]["status"]})
        macro_precision = float(np.mean([x["precision"] for x in class_rows]))
        macro_recall = float(np.mean([x["recall"] for x in class_rows]))
        macro_f1 = float(np.mean([x["f1"] for x in class_rows]))
        total = sum(x["support"] for x in class_rows)
        weighted_precision = float(sum(x["precision"] * x["support"] for x in class_rows) / total) if total else np.nan
        weighted_recall = float(sum(x["recall"] * x["support"] for x in class_rows) / total) if total else np.nan
        weighted_f1 = float(sum(x["f1"] * x["support"] for x in class_rows) / total) if total else np.nan
        curve_data["macro_roc_auc_ovr"] = float(np.mean(aucs)) if aucs else None
        curve_data["macro_average_precision"] = float(np.mean(aps)) if aps else None
        curve_data["undefined_classes"] = [n for n, x in curve_data["classes"].items() if x["status"] == "undefined"]
        summary = {"model": bundle.model_name, "seed": bundle.seed, "fold": bundle.fold, "n_beats": int(len(y)),
            "n_patients": len(set(bundle.patient_group_ids)), "accuracy": ClassificationMetrics.accuracy(y, pred),
            "balanced_accuracy": ClassificationMetrics.balanced_accuracy(y, pred, 5),
            "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1,
            "weighted_precision": weighted_precision, "weighted_recall": weighted_recall, "weighted_f1": weighted_f1,
            "macro_roc_auc_ovr": curve_data["macro_roc_auc_ovr"],
            "macro_pr_auc": curve_data["macro_average_precision"],
            "auc_averaging": curve_data["averaging"], "undefined_auc_classes": ";".join(curve_data["undefined_classes"]),
            "classification_zero_division": 0,
            "confusion_matrix": cm.tolist()}
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).rename_axis("true_class").reset_index()
        return summary, pd.DataFrame(class_rows), cm_df, curve_data

    @staticmethod
    def compute_per_patient(bundle: PredictionBundle) -> pd.DataFrame:
        df = bundle.to_dataframe()
        rows = []
        for pg_id, group in df.groupby("patient_group_id"):
            yt = group.y_true.to_numpy(dtype=np.int64); yp = group.y_pred.to_numpy(dtype=np.int64)
            cm = ClassificationMetrics.confusion_matrix(yt, yp, 5)
            rows.append({"model": bundle.model_name, "seed": bundle.seed, "fold": bundle.fold,
                "patient_group_id": pg_id, "n_beats": len(yt),
                "balanced_accuracy": ClassificationMetrics.balanced_accuracy(yt, yp, 5),
                "accuracy": ClassificationMetrics.accuracy(yt, yp), "macro_f1": ClassificationMetrics.f1(cm, "macro")})
        return pd.DataFrame(rows)

    @staticmethod
    def compute_per_fold(bundle: PredictionBundle) -> pd.DataFrame:
        summary, _, _, _ = ModelEvaluator.compute_classification(bundle)
        return pd.DataFrame([summary])

    @staticmethod
    def compute_per_seed(df: pd.DataFrame) -> pd.DataFrame:
        y = df.y_true.to_numpy(dtype=np.int64); pred = df.y_pred.to_numpy(dtype=np.int64)
        cm = ClassificationMetrics.confusion_matrix(y, pred, 5)
        return pd.DataFrame([{"model": df.model.iloc[0], "seed": int(df.seed.iloc[0]),
            "n_patients": int(df.patient_group_id.nunique()), "n_beats": len(y),
            "accuracy": ClassificationMetrics.accuracy(y, pred),
            "balanced_accuracy": ClassificationMetrics.balanced_accuracy(y, pred, 5),
            "macro_f1": ClassificationMetrics.f1(cm, "macro"), "weighted_f1": ClassificationMetrics.f1(cm, "weighted"),
            "aggregation": "out_of_fold_prediction_pool; not an independent test set"}])

    @staticmethod
    def compute_pooled(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for model, group in df.groupby("model"):
            y = group.y_true.to_numpy(dtype=np.int64); pred = group.y_pred.to_numpy(dtype=np.int64)
            cm = ClassificationMetrics.confusion_matrix(y, pred, 5)
            rows.append({"model": model, "n_patients": int(group.patient_group_id.nunique()), "n_beats": len(y),
                "accuracy": ClassificationMetrics.accuracy(y, pred),
                "balanced_accuracy": ClassificationMetrics.balanced_accuracy(y, pred, 5),
                "macro_f1": ClassificationMetrics.f1(cm, "macro"), "weighted_f1": ClassificationMetrics.f1(cm, "weighted"),
                "aggregation": "out_of_fold_prediction_pool; not an independent test set"})
        return pd.DataFrame(rows)
