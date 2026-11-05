"""Descriptive statistics: parametric and non-parametric summary measures."""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spikeattn.exceptions import InsufficientDataError


class DescriptiveStatistics:
    """Compute central tendency, dispersion, and distribution profiles."""

    @staticmethod
    def summarize(x: np.ndarray | list[float]) -> dict[str, float]:
        """Compute comprehensive location, scale, and shape summary statistics.

        Returns dictionary with mean, std, median, min, max, quartiles (q25, q75),
        iqr, variance, skewness, and kurtosis.
        """
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            raise InsufficientDataError("Cannot summarize empty or non-finite array")

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        median = float(np.median(arr))
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = float(q75 - q25)
        var = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
        skew = float(stats.skew(arr)) if len(arr) >= 3 else 0.0
        kurt = float(stats.kurtosis(arr)) if len(arr) >= 4 else 0.0

        return {
            "n": int(len(arr)),
            "mean": mean,
            "std": std,
            "variance": var,
            "median": median,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": iqr,
            "skewness": skew,
            "kurtosis": kurt,
        }

    @staticmethod
    def mode(x: np.ndarray | list[float]) -> float:
        """Compute the statistical mode."""
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            raise InsufficientDataError("Cannot compute mode on empty data")
        res = stats.mode(arr, keepdims=False)
        return float(res.mode)

    @staticmethod
    def quartiles(x: np.ndarray | list[float]) -> tuple[float, float, float]:
        """Return Q1 (25th percentile), Q2 (median), Q3 (75th percentile)."""
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            raise InsufficientDataError("Cannot compute quartiles on empty data")
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        return float(q1), float(q2), float(q3)

    @staticmethod
    def iqr(x: np.ndarray | list[float]) -> float:
        """Compute the Interquartile Range (IQR = Q3 - Q1)."""
        q1, _, q3 = DescriptiveStatistics.quartiles(x)
        return float(q3 - q1)

    @staticmethod
    def class_distribution(y: np.ndarray | list[Any]) -> dict[Any, int]:
        """Compute counts of categorical labels."""
        counts = Counter(y)
        return dict(counts)

    @staticmethod
    def summarize_frame(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
        """Produce tidy descriptive statistics DataFrame, optionally grouped."""
        if df.empty:
            return pd.DataFrame()

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if by is not None and by in num_cols:
            num_cols.remove(by)

        rows = []
        if by is None:
            for col in num_cols:
                summary = DescriptiveStatistics.summarize(df[col].dropna())
                summary["variable"] = col
                rows.append(summary)
        else:
            for group_val, sub_df in df.groupby(by):
                for col in num_cols:
                    if len(sub_df[col].dropna()) == 0:
                        continue
                    summary = DescriptiveStatistics.summarize(sub_df[col].dropna())
                    summary["group"] = group_val
                    summary["variable"] = col
                    rows.append(summary)

        return pd.DataFrame(rows)
