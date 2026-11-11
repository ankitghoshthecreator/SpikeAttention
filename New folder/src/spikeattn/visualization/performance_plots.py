"""Performance plotting: confusion matrix heatmaps and multi-class ROC curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARRHYTHMIA_CLASS_ORDER = ["N", "S", "V", "F", "Q"]


def _validate_comparison_frames(metrics: pd.DataFrame, per_class: pd.DataFrame) -> None:
    """Fail early with clear schema errors before invoking pandas plotting reshapes."""
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError("metrics must be a non-empty DataFrame")
    if not isinstance(per_class, pd.DataFrame) or per_class.empty:
        raise ValueError("per_class metrics must be a non-empty DataFrame")
    metric_columns = {"model", "balanced_accuracy", "macro_f1"}
    class_columns = {"model", "class", "precision", "recall", "f1"}
    missing_metrics = metric_columns - set(metrics.columns)
    missing_class = class_columns - set(per_class.columns)
    if missing_metrics:
        raise ValueError(f"metrics is missing required columns: {sorted(missing_metrics)}")
    if missing_class:
        raise ValueError(f"per_class metrics is missing required columns: {sorted(missing_class)}")
    for label, frame in (("metrics", metrics), ("per_class metrics", per_class)):
        if frame["model"].isna().any() or frame["model"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label} contains an empty model identifier")
    unknown_classes = set(per_class["class"].dropna().astype(str)) - set(ARRHYTHMIA_CLASS_ORDER)
    if per_class["class"].isna().any() or unknown_classes:
        raise ValueError(f"per_class metrics contains invalid class labels: {sorted(unknown_classes)}")
    metric_models = set(metrics["model"].astype(str))
    class_models = set(per_class["model"].astype(str))
    if metric_models != class_models:
        raise ValueError(f"model identifiers differ between metrics and per_class metrics: {sorted(metric_models)} vs {sorted(class_models)}")


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


def plot_evaluation_comparisons(metrics: pd.DataFrame, per_class: pd.DataFrame, curves: dict,
                               output_dir: Path | str, split: str) -> list[Path]:
    """Write model-comparison metric plots and one-vs-rest curves for one data split."""
    _validate_comparison_frames(metrics, per_class)
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True); paths = []
    for column, title, filename in (("balanced_accuracy", "Balanced accuracy", "balanced_accuracy.png"),
                                    ("macro_f1", "Macro F1", "macro_f1.png")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ordered = metrics.sort_values("model")
        ax.bar(ordered.model, ordered[column], color="#3274a1")
        ax.set_ylim(0, 1); ax.set_ylabel(title); ax.set_title(f"{split.title()} {title} by model")
        ax.tick_params(axis="x", rotation=25); fig.tight_layout()
        path = outdir / filename; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = per_class.pivot_table(index="class", columns="model", values="f1", aggfunc="mean").reindex(ARRHYTHMIA_CLASS_ORDER)
    pivot.plot(kind="bar", ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.set_title(f"{split.title()} per-class F1")
    fig.tight_layout(); path = outdir / "per_class_f1.png"; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, metric, title in zip(axes, ("precision", "recall", "f1"), ("Precision", "Recall", "F1"), strict=True):
        table = per_class.pivot_table(index="class", columns="model", values=metric, aggfunc="mean").reindex(ARRHYTHMIA_CLASS_ORDER)
        table.plot(kind="bar", ax=ax); ax.set_ylim(0, 1); ax.set_title(title); ax.set_xlabel("Class")
    axes[0].set_ylabel("Score"); fig.suptitle(f"{split.title()} per-class metrics")
    fig.tight_layout(); path = outdir / "per_class_metrics.png"; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    for model, curve in curves.items():
        for curve_type, xkey, ykey, xlabel, ylabel in (("roc", "fpr", "tpr", "False positive rate", "True positive rate"),
                                                        ("pr", "recall", "precision", "Recall", "Precision")):
            fig, ax = plt.subplots(figsize=(7, 5))
            for cls, data in curve.get("classes", {}).items():
                if data.get("status") == "defined":
                    auc = data.get("roc_auc_ovr") if curve_type == "roc" else data.get("average_precision")
                    ax.plot(data[xkey], data[ykey], label=f"{cls} (area={auc:.3f})")
            if curve_type == "roc": ax.plot([0, 1], [0, 1], "k--", alpha=.5)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=xlabel, ylabel=ylabel, title=f"{split.title()} {model} one-vs-rest {curve_type.upper()}")
            ax.legend(loc="best"); fig.tight_layout()
            path = outdir / f"{model}_{curve_type}.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths
