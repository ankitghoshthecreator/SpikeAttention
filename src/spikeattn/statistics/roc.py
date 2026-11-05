"""Receiver Operating Characteristic (ROC) analysis: ROC Curve, trapezoidal area, and optimal Youden index."""

from typing import Any

import numpy as np


class ROCAnalyzer:
    """Computes one-vs-rest ROC Curve coordinates, Area Under the ROC Curve, and Youden index thresholds."""

    @staticmethod
    def curve(y_true_binary: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds.

        Args:
            y_true_binary: Binary array of shape [N] where 1 is positive, 0 is negative.
            y_score: Continuous probability scores of shape [N].

        Returns:
            (fpr, tpr, thresholds)
        """
        yt = np.asarray(y_true_binary, dtype=np.int64)
        ys = np.asarray(y_score, dtype=np.float64)

        if len(yt) != len(ys):
            raise ValueError("y_true and y_score must have identical length")

        n_pos = np.sum(yt == 1)
        n_neg = np.sum(yt == 0)

        # Handle degenerate cases: no positive or negative samples
        if n_pos == 0 or n_neg == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

        # Sort descending by score
        order = np.argsort(-ys)
        yt_sorted = yt[order]
        ys_sorted = ys[order]

        # Identify unique threshold points where score changes
        distinct_mask = np.diff(ys_sorted) != 0
        threshold_indices = np.where(distinct_mask)[0]
        threshold_indices = np.append(threshold_indices, len(yt_sorted) - 1)

        tps = np.cumsum(yt_sorted)[threshold_indices]
        fps = (1 + threshold_indices) - tps

        # Prepend point (0, 0)
        tpr = np.concatenate([[0.0], tps / n_pos])
        fpr = np.concatenate([[0.0], fps / n_neg])
        thresholds = np.concatenate([[ys_sorted[0] + 1.0], ys_sorted[threshold_indices]])

        return fpr, tpr, thresholds

    @staticmethod
    def area(fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Compute Area Under the ROC Curve via numerical trapezoidal integration.

        Integrates TPR with respect to FPR: integral TPR d(FPR).
        """
        f = np.asarray(fpr, dtype=np.float64)
        t = np.asarray(tpr, dtype=np.float64)
        # Trapezoidal rule: sum (f[i] - f[i-1]) * (t[i] + t[i-1]) / 2
        return float(np.trapezoid(t, f)) if hasattr(np, "trapezoid") else float(np.trapz(t, f))

    @staticmethod
    def youden_index(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float, float]:
        """Compute optimal classification threshold maximizing Youden's J statistic: J = TPR - FPR.

        Returns:
            (best_threshold, max_j, optimal_operating_point)
        """
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx])
        max_j = float(j_scores[best_idx])
        optimal_point = (float(fpr[best_idx]), float(tpr[best_idx]))
        return best_thresh, max_j, optimal_point

    @staticmethod
    def one_vs_rest(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        class_names: list[str] | None = None,
        n_classes: int = 5,
    ) -> dict[str, Any]:
        """Perform one-vs-rest ROC Curve analysis across all classes.

        Computes per-class ROC Curve coordinates, Area Under the ROC Curve,
        optimal Youden thresholds, and the macro-averaged Area Under the ROC Curve.
        """
        names = class_names or ["N", "S", "V", "F", "Q"]
        yt = np.asarray(y_true, dtype=np.int64)
        scores = np.asarray(y_scores, dtype=np.float64)

        curves = {}
        areas = []

        for c in range(min(n_classes, scores.shape[1])):
            name = names[c]
            binary_true = (yt == c).astype(np.int64)
            class_scores = scores[:, c]

            n_pos = int(np.sum(binary_true == 1))
            if n_pos == 0:
                curves[name] = {
                    "fpr": [0.0, 1.0],
                    "tpr": [0.0, 1.0],
                    "area_under_roc_curve": None,
                    "youden_threshold": None,
                    "support": 0,
                    "note": f"Class {name} absent in ground-truth labels",
                }
                continue

            fpr, tpr, thresh = ROCAnalyzer.curve(binary_true, class_scores)
            area_val = ROCAnalyzer.area(fpr, tpr)
            best_th, max_j, opt_pt = ROCAnalyzer.youden_index(fpr, tpr, thresh)

            curves[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresh.tolist(),
                "area_under_roc_curve": area_val,
                "youden_threshold": best_th,
                "max_youden_j": max_j,
                "optimal_operating_point": opt_pt,
                "support": n_pos,
            }
            areas.append(area_val)

        macro_area = float(np.mean(areas)) if areas else None

        return {
            "per_class": curves,
            "macro_area_under_roc_curve": macro_area,
            "classes_evaluated": len(areas),
        }
