"""Bidirectional Recurrent Neural Network (Bi-GRU) baseline model implementing BaseModel."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikeattn.models.base_model import BaseModel


class RNNBaseline(nn.Module, BaseModel):
    """Bidirectional GRU baseline model for two-channel ECG classification.

    Architecture:
        Bi-GRU (input_size=2, hidden_size=64, num_layers=2) -> Global Mean Pooling -> Linear(128, 5)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._name = "rnn"

        self.rnn = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    @property
    def name(self) -> str:
        return self._name

    def _prepare_tensor(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
        else:
            t = x.float()

        target_device = self.classifier.weight.device
        if t.ndim != 3 or t.shape[1] != 2:
            raise ValueError(f"RNNBaseline expects input of shape [B, 2, L], got {t.shape}")
        # Transpose to [B, L, 2] for batch_first RNN
        return t.permute(0, 2, 1).to(device=target_device)

    def forward_logits(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        seq = self._prepare_tensor(x)          # [B, L, 2]
        rnn_out, _ = self.rnn(seq)             # [B, L, 128]
        pooled = rnn_out.mean(dim=1)           # [B, 128]
        return self.classifier(pooled)         # [B, 5]

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.forward_logits(x)

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward_logits(X)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs.astype(np.float64)

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
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
    ) -> RNNBaseline:
        self.train()
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.float()
        y_t = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y.long()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(self.forward_logits(bx), by)
                loss.backward()
                optimizer.step()

        return self

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
