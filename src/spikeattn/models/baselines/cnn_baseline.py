"""1D Convolutional Neural Network baseline model implementing BaseModel."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikeattn.models.base_model import BaseModel


class CNNBaseline(nn.Module, BaseModel):
    """1D Convolutional Neural Network baseline for two-channel ECG classification.

    Architecture:
        Conv1d(2 -> 32) -> BN -> ReLU -> MaxPool
        Conv1d(32 -> 64) -> BN -> ReLU -> MaxPool
        Conv1d(64 -> 128) -> BN -> ReLU -> AdaptiveAvgPool(1)
        Linear(128 -> 5)
    """

    def __init__(self, in_channels: int = 2, n_classes: int = 5) -> None:
        super().__init__()
        self._name = "cnn"

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

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
            raise ValueError(f"CNNBaseline expects input of shape [B, 2, L], got {t.shape}")
        return t.to(device=target_device)

    def forward_logits(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        t = self._prepare_tensor(x)
        feat = self.features(t).squeeze(-1)  # [B, 128]
        return self.classifier(feat)          # [B, 5]

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
    ) -> CNNBaseline:
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
