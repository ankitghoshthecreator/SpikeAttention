"""Multiple testing correction: Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg."""

from typing import Any

import numpy as np


class MultipleComparisonCorrector:
    """Corrects p-values across pre-registered hypothesis test families."""

    @staticmethod
    def bonferroni(p_values: np.ndarray | list[float]) -> np.ndarray:
        """Single-step Bonferroni correction: p_adj = min(1.0, p * m)."""
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        if m == 0:
            return np.array([], dtype=np.float64)
        return np.clip(p * m, 0.0, 1.0)

    @staticmethod
    def holm(p_values: np.ndarray | list[float]) -> np.ndarray:
        """Step-down Holm-Bonferroni correction (controls FWER strongly).

        p_adj_(i) = min(1.0, max_{j <= i} (m - j + 1) * p_(j))
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        if m == 0:
            return np.array([], dtype=np.float64)

        order = np.argsort(p)
        sorted_p = p[order]
        adjusted = np.empty(m, dtype=np.float64)

        running_max = 0.0
        for i in range(m):
            multiplier = m - i
            val = sorted_p[i] * multiplier
            running_max = max(running_max, val)
            adjusted[i] = min(1.0, running_max)

        # Restore original order
        result = np.empty(m, dtype=np.float64)
        result[order] = adjusted
        return result

    @staticmethod
    def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
        """Step-up Benjamini-Hochberg procedure (controls False Discovery Rate).

        p_adj_(i) = min(1.0, min_{j >= i} (m / j) * p_(j))
        """
        p = np.asarray(p_values, dtype=np.float64)
        m = len(p)
        if m == 0:
            return np.array([], dtype=np.float64)

        order = np.argsort(p)
        sorted_p = p[order]
        adjusted = np.empty(m, dtype=np.float64)

        running_min = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            val = (sorted_p[i] * m) / rank
            running_min = min(running_min, val)
            adjusted[i] = min(1.0, running_min)

        result = np.empty(m, dtype=np.float64)
        result[order] = adjusted
        return result

    @staticmethod
    def correct(
        p_values: np.ndarray | list[float],
        method: str = "holm",
    ) -> np.ndarray:
        """Apply the specified correction method."""
        method = method.lower()
        if method == "holm":
            return MultipleComparisonCorrector.holm(p_values)
        elif method == "bonferroni":
            return MultipleComparisonCorrector.bonferroni(p_values)
        elif method in {"bh", "benjamini_hochberg", "fdr"}:
            return MultipleComparisonCorrector.benjamini_hochberg(p_values)
        else:
            raise ValueError(f"Unknown correction method '{method}'; choose 'holm', 'bonferroni', or 'benjamini_hochberg'")

    @staticmethod
    def correct_families(
        family_p_dict: dict[str, list[float]],
        methods: dict[str, str],
    ) -> dict[str, np.ndarray]:
        """Apply pre-specified correction methods per hypothesis family."""
        out = {}
        for family, p_vals in family_p_dict.items():
            method = methods.get(family, "holm")
            out[family] = MultipleComparisonCorrector.correct(p_vals, method=method)
        return out
