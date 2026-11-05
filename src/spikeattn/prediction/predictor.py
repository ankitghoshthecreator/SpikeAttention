"""Inference coordinator generating calibrated probability predictions adhering to Results Contract v1.0.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class PredictionBundle:
    """Encapsulates model predictions on an evaluation split conforming to Results Contract v1.0.0."""

    beat_ids: list[str]
    patient_group_ids: list[str]
    record_ids: list[str]
    seed: int
    fold: int
    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray  # [N, 5]

    def __post_init__(self) -> None:
        n = len(self.beat_ids)
        if len(self.patient_group_ids) != n or len(self.record_ids) != n:
            raise ValueError(
                f"PredictionBundle identifier length mismatch: beat_ids={n}, "
                f"patient_group_ids={len(self.patient_group_ids)}, record_ids={len(self.record_ids)}"
            )
        yt = np.asarray(self.y_true)
        yp = np.asarray(self.y_pred)
        probs = np.asarray(self.probabilities)
        if len(yt) != n:
            raise ValueError(f"PredictionBundle y_true length ({len(yt)}) does not match beat_ids ({n})")
        if len(yp) != n:
            raise ValueError(f"PredictionBundle y_pred length ({len(yp)}) does not match beat_ids ({n})")
        if probs.shape != (n, 5):
            raise ValueError(f"PredictionBundle probabilities shape {probs.shape} must be ({n}, 5)")

    def to_dataframe(self) -> pd.DataFrame:
        """Serializes predictions to DataFrame with columns:
        beat_id, patient_group_id, record_id, seed, fold, model, y_true, y_pred, p_N, p_S, p_V, p_F, p_Q
        """
        n = len(self.beat_ids)
        probs = np.asarray(self.probabilities, dtype=np.float64)

        data = {
            "beat_id": self.beat_ids,
            "patient_group_id": self.patient_group_ids,
            "record_id": self.record_ids,
            "seed": [self.seed] * n,
            "fold": [self.fold] * n,
            "model": [self.model_name] * n,
            "y_true": np.asarray(self.y_true, dtype=np.int64),
            "y_pred": np.asarray(self.y_pred, dtype=np.int64),
            "p_N": probs[:, 0],
            "p_S": probs[:, 1],
            "p_V": probs[:, 2],
            "p_F": probs[:, 3],
            "p_Q": probs[:, 4],
        }
        return pd.DataFrame(data)


class Predictor:
    """Manages batch inference for classical and neural models."""

    @staticmethod
    def predict(
        model: Any,
        signals: np.ndarray | torch.Tensor,
        metadata: pd.DataFrame | dict[str, Any],
        seed: int,
        fold: int,
        batch_size: int = 256,
        device: str = "cpu",
        y_true: np.ndarray | torch.Tensor | None = None,
    ) -> PredictionBundle:
        """Run inference over input signals and construct PredictionBundle.

        Args:
            model: BaseModel instance.
            signals: Array or tensor of shape [N, 2, 360].
            metadata: Metadata DataFrame or mapping containing beat_id, patient_group_id, record_id, and ground-truth labels.
            seed: Random seed of the evaluation fold.
            fold: Fold index (0..4).
            batch_size: Evaluation batch size.
            device: Computation device ('cpu' or 'cuda').
            y_true: Optional explicit ground-truth labels array of shape [N]. When provided, it is validated against metadata.

        Returns:
            PredictionBundle strictly conforming to Results Contract v1.0.0.
        """
        model_name = getattr(model, "name", "unknown_model")

        if isinstance(signals, torch.Tensor):
            arr = signals.detach().cpu().numpy()
        else:
            arr = np.asarray(signals)

        n = len(arr)
        meta_len = len(metadata) if hasattr(metadata, "__len__") else None
        if meta_len is not None and meta_len != n:
            raise ValueError(
                f"Row count mismatch: input signals has {n} samples, but metadata has {meta_len} rows."
            )

        if hasattr(model, "predict_proba"):
            probs_list = []
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                batch_x = arr[start:end]
                p = model.predict_proba(batch_x)
                probs_list.append(p)
            probs = np.vstack(probs_list) if probs_list else np.empty((0, 5), dtype=np.float64)
        else:
            raise TypeError(f"Model {model_name} does not implement predict_proba()")

        if probs.shape != (n, 5):
            raise ValueError(f"Predicted probabilities shape {probs.shape} must be ({n}, 5)")

        y_pred = np.argmax(probs, axis=-1).astype(np.int64)

        # Inspect available metadata columns / keys
        if isinstance(metadata, pd.DataFrame):
            available_cols = list(metadata.columns)
        elif isinstance(metadata, dict):
            available_cols = list(metadata.keys())
        else:
            raise TypeError(f"metadata must be a pd.DataFrame or dict, got {type(metadata)}")

        # Validate required identifiers
        for req in ("beat_id", "patient_group_id", "record_id"):
            if req not in available_cols:
                raise KeyError(
                    f"Required identifier '{req}' not found in metadata. "
                    f"Available fields: {available_cols}"
                )

        if isinstance(metadata, pd.DataFrame):
            beat_ids = metadata["beat_id"].astype(str).tolist()
            pg_ids = metadata["patient_group_id"].astype(str).tolist()
            rec_ids = metadata["record_id"].astype(str).tolist()
        else:
            beat_ids = list(map(str, metadata["beat_id"]))
            pg_ids = list(map(str, metadata["patient_group_id"]))
            rec_ids = list(map(str, metadata["record_id"]))

        # Resolve ground-truth labels
        label_col = None
        for candidate in ("y_true", "class_id", "label"):
            if candidate in available_cols:
                label_col = candidate
                break

        if y_true is not None:
            resolved_y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true, dtype=np.int64)
            if label_col is not None:
                meta_labels = (
                    metadata[label_col].to_numpy(dtype=np.int64)
                    if isinstance(metadata, pd.DataFrame)
                    else np.asarray(metadata[label_col], dtype=np.int64)
                )
                if not np.array_equal(resolved_y_true, meta_labels):
                    raise ValueError(
                        f"Explicit y_true does not match '{label_col}' in metadata."
                    )
        else:
            if label_col is None:
                raise KeyError(
                    f"Ground-truth label column not found in metadata. "
                    f"Expected canonical 'y_true' or aliases ('class_id', 'label'). "
                    f"Available fields: {available_cols}"
                )
            resolved_y_true = (
                metadata[label_col].to_numpy(dtype=np.int64)
                if isinstance(metadata, pd.DataFrame)
                else np.asarray(metadata[label_col], dtype=np.int64)
            )

        if len(resolved_y_true) != n:
            raise ValueError(
                f"Ground-truth label count ({len(resolved_y_true)}) does not match signals count ({n})."
            )

        # Validate that y_true contains valid class indices
        if n > 0:
            unique_labels = set(np.unique(resolved_y_true))
            if not unique_labels.issubset({0, 1, 2, 3, 4}):
                raise ValueError(
                    f"Ground-truth y_true contains invalid class indices: {unique_labels}. Expected subset of {{0, 1, 2, 3, 4}}."
                )

        return PredictionBundle(
            beat_ids=beat_ids,
            patient_group_ids=pg_ids,
            record_ids=rec_ids,
            seed=seed,
            fold=fold,
            model_name=model_name,
            y_true=resolved_y_true,
            y_pred=y_pred,
            probabilities=probs,
        )
