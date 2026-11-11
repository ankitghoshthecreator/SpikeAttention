"""Classification metrics: confusion matrix decomposition, precision, recall, specificity, F1, and balanced accuracy."""

from typing import Any

import numpy as np


class ClassificationMetrics:
    """Computes multiclass and one-vs-rest classification performance metrics."""

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> np.ndarray:
        """Compute integer confusion matrix of shape [n_classes, n_classes].

        Rows represent true classes; columns represent predicted classes.
        """
        yt = np.asarray(y_true, dtype=np.int64)
        yp = np.asarray(y_pred, dtype=np.int64)
        if len(yt) != len(yp):
            raise ValueError(f"y_true ({len(yt)}) and y_pred ({len(yp)}) must have identical lengths")

        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(yt, yp, strict=True):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t, p] += 1
        return cm

    @staticmethod
    def per_class_counts(cm: np.ndarray) -> dict[int, dict[str, int]]:
        """Extract True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN)."""
        k = cm.shape[0]
        total = np.sum(cm)
        counts = {}
        for c in range(k):
            tp = int(cm[c, c])
            fn = int(np.sum(cm[c, :]) - tp)
            fp = int(np.sum(cm[:, c]) - tp)
            tn = int(total - tp - fn - fp)
            counts[c] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        return counts

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Overall classification accuracy = correct / total."""
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        if len(yt) == 0:
            return 0.0
        return float(np.mean(yt == yp))

    @staticmethod
    def precision(cm: np.ndarray, average: str = "macro") -> float | np.ndarray:
        """Precision = TP / (TP + FP)."""
        k = cm.shape[0]
        tp = np.diag(cm)
        pred_pos = np.sum(cm, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec_per_class = np.where(pred_pos > 0, tp / pred_pos, 0.0)

        if average is None:
            return prec_per_class
        elif average == "macro":
            return float(np.mean(prec_per_class))
        elif average == "weighted":
            weights = np.sum(cm, axis=1)
            total = np.sum(weights)
            return float(np.sum(prec_per_class * weights) / total) if total > 0 else 0.0
        else:
            raise ValueError(f"Unknown average '{average}'")

    @staticmethod
    def recall(cm: np.ndarray, average: str = "macro") -> float | np.ndarray:
        """Recall (Sensitivity) = TP / (TP + FN)."""
        k = cm.shape[0]
        tp = np.diag(cm)
        actual_pos = np.sum(cm, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rec_per_class = np.where(actual_pos > 0, tp / actual_pos, 0.0)

        if average is None:
            return rec_per_class
        elif average == "macro":
            return float(np.mean(rec_per_class))
        elif average == "weighted":
            weights = np.sum(cm, axis=1)
            total = np.sum(weights)
            return float(np.sum(rec_per_class * weights) / total) if total > 0 else 0.0
        else:
            raise ValueError(f"Unknown average '{average}'")

    @staticmethod
    def specificity(cm: np.ndarray, average: str = "macro") -> float | np.ndarray:
        """Specificity = TN / (TN + FP)."""
        counts = ClassificationMetrics.per_class_counts(cm)
        k = cm.shape[0]
        spec_per_class = np.zeros(k, dtype=np.float64)
        for c in range(k):
            tn = counts[c]["TN"]
            fp = counts[c]["FP"]
            spec_per_class[c] = tn / (tn + fp) if (tn + fp) > 0 else 1.0

        if average is None:
            return spec_per_class
        elif average == "macro":
            return float(np.mean(spec_per_class))
        else:
            return float(np.mean(spec_per_class))

    @staticmethod
    def f1(cm: np.ndarray, average: str = "macro") -> float | np.ndarray:
        """F1-score = 2 * (precision * recall) / (precision + recall)."""
        prec = ClassificationMetrics.precision(cm, average=None)
        rec = ClassificationMetrics.recall(cm, average=None)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_per_class = np.where((prec + rec) > 0, 2.0 * (prec * rec) / (prec + rec), 0.0)

        if average is None:
            return f1_per_class
        elif average == "macro":
            return float(np.mean(f1_per_class))
        elif average == "weighted":
            weights = np.sum(cm, axis=1)
            total = np.sum(weights)
            return float(np.sum(f1_per_class * weights) / total) if total > 0 else 0.0
        else:
            raise ValueError(f"Unknown average '{average}'")

    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
        """Balanced accuracy = mean recall over the classes actually present in y_true."""
        yt = np.asarray(y_true, dtype=np.int64)
        yp = np.asarray(y_pred, dtype=np.int64)
        present_classes = np.unique(yt)
        if len(present_classes) == 0:
            return 0.0

        cm = ClassificationMetrics.confusion_matrix(yt, yp, n_classes=n_classes)
        rec = ClassificationMetrics.recall(cm, average=None)
        present_recalls = rec[present_classes]
        return float(np.mean(present_recalls))

    @staticmethod
    def report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list[str] | None = None,
        n_classes: int = 5,
    ) -> dict[str, Any]:
        """Compute full classification report with precision, recall, f1, and support."""
        names = target_names or ["N", "S", "V", "F", "Q"]
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred, n_classes=n_classes)
        prec = ClassificationMetrics.precision(cm, average=None)
        rec = ClassificationMetrics.recall(cm, average=None)
        f1_vals = ClassificationMetrics.f1(cm, average=None)
        support = np.sum(cm, axis=1)

        result: dict[str, Any] = {}
        for c in range(min(n_classes, len(names))):
            name = names[c]
            result[name] = {
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "f1": float(f1_vals[c]),
                "support": int(support[c]),
            }

        result["accuracy"] = ClassificationMetrics.accuracy(y_true, y_pred)
        result["balanced_accuracy"] = ClassificationMetrics.balanced_accuracy(y_true, y_pred, n_classes=n_classes)
        result["macro_f1"] = ClassificationMetrics.f1(cm, average="macro")
        result["weighted_f1"] = ClassificationMetrics.f1(cm, average="weighted")
        result["confusion_matrix"] = cm.tolist()
        return result
