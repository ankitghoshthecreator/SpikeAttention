"""Distribution visualization: class frequencies and patient-wise data partitioning."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(
    class_counts: dict[str, int] | dict[int, int],
    output_path: Path | str,
    class_names: list[str] | None = None,
) -> Path:
    """Render class frequency bar chart showing extreme class imbalance."""
    names = class_names or ["N", "S", "V", "F", "Q"]
    fig, ax = plt.subplots(figsize=(8, 5))

    keys = list(range(len(names))) if all(isinstance(k, int) for k in class_counts) else names
    vals = [class_counts.get(k, 0) for k in keys]

    bars = ax.bar(names, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax.set_ylabel("Number of Heartbeats")
    ax.set_xlabel("AAMI EC57 Rhythm Class")
    ax.set_title("MIT-BIH Dataset Class Distribution (AAMI EC57)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_patient_distribution(
    beats_meta_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Render horizontal bar chart of heartbeats contributed per patient group."""
    counts = beats_meta_df.groupby("patient_group_id")["beat_id"].count().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    counts.plot(kind="barh", ax=ax, color="#4c72b0")
    ax.set_xlabel("Number of Annotated Beats")
    ax.set_ylabel("Patient Group ID")
    ax.set_title("Heartbeats per Patient Group (Clustering / Stratification Unit)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
