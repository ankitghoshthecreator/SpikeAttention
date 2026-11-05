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
    ) -> PredictionBundle:
        """Run inference over input signals and construct PredictionBundle.

        Args:
            model: BaseModel instance.
            signals: Array or tensor of shape [N, 2, 360].
            metadata: Metadata DataFrame or mapping containing beat_id, patient_group_id, record_id, y_true.
            seed: Random seed of the evaluation fold.
            fold: Fold index (0..4).
            batch_size: Evaluation batch size.
            device: Computation device ('cpu' or 'cuda').

        Returns:
            PredictionBundle.
        """
        model_name = getattr(model, "name", "unknown_model")

        if isinstance(signals, torch.Tensor):
            arr = signals.detach().cpu().numpy()
        else:
            arr = np.asarray(signals)

        n = len(arr)

        if hasattr(model, "predict_proba"):
            # Batch inference to avoid OOM
            probs_list = []
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                batch_x = arr[start:end]
                p = model.predict_proba(batch_x)
                probs_list.append(p)
            probs = np.vstack(probs_list)
        else:
            raise TypeError(f"Model {model_name} does not implement predict_proba()")

        y_pred = np.argmax(probs, axis=-1).astype(np.int64)

        if isinstance(metadata, pd.DataFrame):
            beat_ids = metadata["beat_id"].astype(str).tolist()
            pg_ids = metadata["patient_group_id"].astype(str).tolist()
            rec_ids = metadata["record_id"].astype(str).tolist()
            y_true = metadata["label"].to_numpy(dtype=np.int64) if "label" in metadata.columns else metadata["y_true"].to_numpy(dtype=np.int64)
        else:
            beat_ids = list(map(str, metadata["beat_id"]))
            pg_ids = list(map(str, metadata["patient_group_id"]))
            rec_ids = list(map(str, metadata["record_id"]))
            y_true = np.asarray(metadata.get("label", metadata.get("y_true")), dtype=np.int64)

        return PredictionBundle(
            beat_ids=beat_ids,
            patient_group_ids=pg_ids,
            record_ids=rec_ids,
            seed=seed,
            fold=fold,
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probs,
        )
