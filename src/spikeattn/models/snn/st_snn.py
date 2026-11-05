"""Spike-Attention Spiking Neural Network (ST-SNN) with Spatio-Temporal Attention."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikeattn.encoding.delta_encoder import DeltaSpikeEncoder
from spikeattn.models.base_model import BaseModel
from spikeattn.models.snn.attention import InterLeadAttention, TemporalAttention
from spikeattn.models.snn.spiking_conv import SpikingConv1dBlock


class SpikeAttentionSNN(nn.Module, BaseModel):
    """Hybrid Spatio-Temporal Attention Spiking Neural Network (ST-SNN).

    Architecture:
        1. Delta Spike Encoder: [B, 2, L] -> [B, T, 4] spikes
        2. Spiking Conv1D Block 1: 4 -> 32 channels (stride 2)
        3. Inter-Lead Spatial Attention: dynamic cross-lead coupling [B, T, 32]
        4. Spiking Conv1D Block 2: 32 -> 64 channels (stride 2)
        5. Temporal Self-Attention: cardiac temporal token refinement [B, T, 64]
        6. Global Temporal Rate Pooling -> Linear Head [B, 5]
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim1: int = 32,
        hidden_dim2: int = 64,
        n_classes: int = 5,
        tau: float = 10.0,
        alpha: float = 2.0,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._name = "st_snn"
        self.encoder = DeltaSpikeEncoder(scale=0.5)

        self.conv_block1 = SpikingConv1dBlock(
            in_channels=in_channels,
            out_channels=hidden_dim1,
            kernel_size=7,
            stride=2,
            tau=tau,
            alpha=alpha,
        )
        self.inter_lead_attn = InterLeadAttention(
            channels=hidden_dim1,
            dropout=dropout,
        )
        self.conv_block2 = SpikingConv1dBlock(
            in_channels=hidden_dim1,
            out_channels=hidden_dim2,
            kernel_size=5,
            stride=2,
            tau=tau,
            alpha=alpha,
        )
        self.temporal_attn = TemporalAttention(
            channels=hidden_dim2,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim2, n_classes)

        self.last_lead_weights: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return self._name

    def _ensure_spikes(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Ensure input tensor is converted to spikes [B, T, 4] on the model device."""
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
        else:
            t = x.float()

        target_device = self.classifier.weight.device

        if t.ndim == 3 and t.shape[1] == 2:
            if not self.encoder.fitted:
                self.encoder.fit(t)
            spikes = self.encoder.encode(t)
            return spikes.to(device=target_device)
        elif t.ndim == 3 and t.shape[-1] == 4:
            return t.to(device=target_device)
        else:
            raise ValueError(f"Expected input shape [B, 2, 360] or [B, T, 4], got {t.shape}")

    def forward_logits(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Compute unnormalized class logits of shape [B, 5]."""
        spikes = self._ensure_spikes(x)

        # Stage 1: Initial Spiking Convolutions
        s1, _ = self.conv_block1(spikes)  # [B, T1, 32]

        # Stage 2: Spatial Inter-Lead Attention
        refined_spatial, lead_weights = self.inter_lead_attn(s1)
        self.last_lead_weights = lead_weights.detach()

        # Stage 3: Second Spiking Convolution Block
        s2, _ = self.conv_block2(refined_spatial)  # [B, T2, 64]

        # Stage 4: Temporal Attention Refinement
        refined_temporal = self.temporal_attn(s2)  # [B, T2, 64]

        # Stage 5: Global Temporal Rate Pooling & Classification
        pooled = refined_temporal.mean(dim=1)  # [B, 64]
        logits = self.classifier(pooled)       # [B, 5]
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
        """Return class index predictions in {0, 1, 2, 3, 4}."""
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
    ) -> SpikeAttentionSNN:
        """Fit model on training set."""
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
