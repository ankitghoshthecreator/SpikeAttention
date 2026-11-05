"""Statistical inference plots: Forest plots for confidence intervals and paired difference distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forest_ci(
    intervals_dict: dict[str, dict[str, Any]],
    output_path: Path | str,
    metric_label: str = "Balanced Accuracy",
) -> Path:
    """Render Forest plot illustrating point estimates and 95% bootstrap confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 4 + len(intervals_dict) * 0.4))

    models = list(intervals_dict.keys())
    y_pos = np.arange(len(models))

    points = [intervals_dict[m].get("point", 0.0) for m in models]
    lowers = [intervals_dict[m].get("lower", 0.0) for m in models]
    uppers = [intervals_dict[m].get("upper", 0.0) for m in models]

    xerr_low = [p - lo for p, lo in zip(points, lowers, strict=True)]
    xerr_high = [hi - p for p, hi in zip(points, uppers, strict=True)]

    ax.errorbar(
        points, y_pos,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#1f77b4",
        ecolor="#333333",
        elinewidth=2,
        capsize=4,
        capthick=1.5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.upper() for m in models])
    ax.set_xlabel(f"{metric_label} (Mean and 95% Clustered Bootstrap CI)")
    ax.set_title("Model Performance with Clustered Patient Bootstrap Confidence Intervals")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_paired_differences(
    diffs: np.ndarray | list[float],
    model_a: str,
    model_b: str,
    output_path: Path | str,
) -> Path:
    """Render boxplot and jittered strip plot of patient-level performance differences."""
    arr = np.asarray(diffs, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(arr, vert=True, patch_artist=True, boxprops=dict(facecolor="#d1e5f0", color="#2166ac"))

    # Jittered scatter overlay
    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(arr))
    ax.scatter(np.ones_like(arr) + jitter, arr, alpha=0.6, color="#4393c3", s=25)

    ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.5, label="Null Effect (Delta = 0)")
    ax.set_ylabel(f"Balanced Accuracy Difference ({model_a} - {model_b})")
    ax.set_xticklabels([f"{model_a.upper()} vs {model_b.upper()}"])
    ax.set_title("Patient-Level Paired Differences Distribution")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
