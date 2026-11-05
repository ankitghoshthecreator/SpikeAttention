"""Performance plotting: confusion matrix heatmaps and multi-class ROC curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path | str,
    class_names: list[str] | None = None,
    normalize: bool = True,
) -> Path:
    """Render confusion matrix heatmap.

    Args:
        cm: Confusion matrix of shape [5, 5].
        output_path: Destination image path.
        class_names: Class labels (default ['N', 'S', 'V', 'F', 'Q']).
        normalize: Whether to normalize by true row counts.
    """
    names = class_names or ["N", "S", "V", "F", "Q"]
    mat = np.asarray(cm, dtype=np.float64)

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display_mat = mat / row_sums
        fmt = ".2%"
    else:
        display_mat = mat
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display_mat, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Normalized Confusion Matrix" if normalize else "Raw Confusion Matrix")

    thresh = display_mat.max() / 2.0
    for i in range(len(names)):
        for j in range(len(names)):
            val = display_mat[i, j]
            txt = f"{val:.1%}" if normalize else f"{int(val):,}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_roc_curves(
    roc_dict: dict[str, Any],
    output_path: Path | str,
    model_name: str = "st_snn",
) -> Path:
    """Render one-vs-rest ROC curves for all classes.

    TERMINOLOGY NOTE: Area is designated as 'AU (Area Under the ROC Curve)'.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AU=0.500)")

    per_class = roc_dict.get("per_class", {})
    colors = {"N": "#1f77b4", "S": "#ff7f0e", "V": "#2ca02c", "F": "#d62728", "Q": "#9467bd"}

    for name, cdict in per_class.items():
        fpr = cdict.get("fpr", [])
        tpr = cdict.get("tpr", [])
        area_val = cdict.get("area_under_roc_curve")
        if area_val is not None:
            label = f"{name} (AU = {area_val:.3f})"
        else:
            label = f"{name} (support=0)"
        ax.plot(fpr, tpr, color=colors.get(name, "black"), label=label, linewidth=1.5)

    macro_area = roc_dict.get("macro_area_under_roc_curve")
    title_suffix = f" (Macro AU = {macro_area:.3f})" if macro_area is not None else ""
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curves by Class â€” {model_name.upper()}{title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
