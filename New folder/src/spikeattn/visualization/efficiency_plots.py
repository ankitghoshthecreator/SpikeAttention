"""Efficiency trade-off plots: Accuracy vs. Latency and Accuracy vs. Estimated Energy."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_vs_latency(
    merged_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Scatter plot displaying Balanced Accuracy vs. Median Inference Latency (ms)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in merged_df.iterrows():
        model = str(row["model"])
        lat = float(row.get("latency_ms", 0.0))
        acc = float(row.get("balanced_accuracy", 0.0))

        color = "red" if "st_snn" in model else "blue"
        ax.scatter(lat, acc, s=120, color=color, alpha=0.8, edgecolors="black", linewidth=1.2)
        ax.annotate(
            model.upper(),
            (lat, acc),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold" if "st_snn" in model else "normal",
        )

    ax.set_xlabel("Median Hardware Latency per Beat (ms) [Measured]")
    ax.set_ylabel("Mean Balanced Accuracy [Patient-Group Unit]")
    ax.set_title("Performance vs. Latency Trade-Off")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_accuracy_vs_energy(
    merged_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Scatter plot displaying Balanced Accuracy vs. Estimated Per-Inference Energy (uJ)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in merged_df.iterrows():
        model = str(row["model"])
        energy = float(row.get("energy_uJ", 0.0))
        acc = float(row.get("balanced_accuracy", 0.0))

        color = "forestgreen" if "snn" in model else "purple"
        ax.scatter(energy, acc, s=120, color=color, alpha=0.8, edgecolors="black", linewidth=1.2)
        ax.annotate(
            model.upper(),
            (energy, acc),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold" if "st_snn" in model else "normal",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Estimated Energy per Beat (uJ, log scale) [Estimated via CMOS Model]")
    ax.set_ylabel("Mean Balanced Accuracy [Patient-Group Unit]")
    ax.set_title("Performance vs. Energy Efficiency (Measured vs. Estimated)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
