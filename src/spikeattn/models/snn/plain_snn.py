"""Plain Spiking Neural Network baseline architecture without attention mechanisms."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikeattn.encoding.delta_encoder import DeltaSpikeEncoder
from spikeattn.models.base_model import BaseModel
from spikeattn.models.snn.spiking_conv import SpikingConv1dBlock


class PlainSNN(nn.Module, BaseModel):
    """Plain SNN baseline model: 2 Spiking Conv1D blocks -> Global Average Pooling -> Classifier.

    Accepts continuous ECG signals [B, 2, 360] (encoded to spikes) or spike rasters [B, T, 4].
    Produces 5 class logits corresponding to AAMI classes (N, S, V, F, Q).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim1: int = 32,
        hidden_dim2: int = 64,
        n_classes: int = 5,
        tau: float = 10.0,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self._name = "plain_snn"
        self.encoder = DeltaSpikeEncoder(scale=0.5)

        self.block1 = SpikingConv1dBlock(
            in_channels=in_channels,
            out_channels=hidden_dim1,
            kernel_size=7,
            stride=2,
            tau=tau,
            alpha=alpha,
        )
        self.block2 = SpikingConv1dBlock(
            in_channels=hidden_dim1,
            out_channels=hidden_dim2,
            kernel_size=5,
            stride=2,
            tau=tau,
            alpha=alpha,
        )
        self.classifier = nn.Linear(hidden_dim2, n_classes)

    @property
    def name(self) -> str:
        return self._name

    def _ensure_spikes(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert [B, 2, L] continuous signal into [B, T, 4] spikes if not already spiked."""
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
        else:
            t = x.float()

        if t.ndim == 3 and t.shape[1] == 2:
            # Continuous signals [B, 2, L]
            if not self.encoder.fitted:
                self.encoder.fit(t)
            spikes = self.encoder.encode(t)
            return spikes.to(device=self.classifier.weight.device)
        elif t.ndim == 3 and t.shape[-1] == 4:
            # Already spike tensor [B, T, 4]
            return t.to(device=self.classifier.weight.device)
        else:
            raise ValueError(f"Expected input shape [B, 2, 360] or [B, T, 4], got {t.shape}")

    def forward_logits(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute unnormalized class logits [B, 5]."""
        spikes = self._ensure_spikes(x)

        s1, _ = self.block1(spikes)
        s2, _ = self.block2(s1)

        # Global temporal rate / average pooling across time dimension
        pooled = s2.mean(dim=1)  # [B, hidden_dim2]
        logits = self.classifier(pooled)
        return logits

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.forward_logits(x)

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return class probabilities of shape [B, 5] summing to 1.0."""
        self.eval()
        with torch.no_grad():
            logits = self.forward_logits(X)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs.astype(np.float64)

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return predicted class labels in {0, 1, 2, 3, 4}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1).astype(np.int64)

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        val_data: tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor] | None = None,
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 128,
    ) -> PlainSNN:
        """Fit model parameters on training data."""
        self.train()
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X.float()

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).long()
        else:
            y_tensor = y.long()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = self.forward_logits(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
