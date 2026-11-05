"""Temporal and Inter-Lead attention modules for spatio-temporal ECG feature refinement."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Scaled dot-product self-attention across the temporal dimension [B, T, C].

    Refines temporal spike trace representations by attending across the cardiac cycle time steps.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by n_heads ({n_heads})")

        self.channels = channels
        self.n_heads = n_heads
        self.d_k = channels // n_heads

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention.

        Args:
            x: Input tensor of shape [B, T, C].

        Returns:
            Refined tensor of shape [B, T, C].
        """
        B, T, C = x.shape
        residual = x

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, T, d_k]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)    # [B, H, T, T]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)                                # [B, H, T, d_k]
        context = context.transpose(1, 2).contiguous().view(B, T, C)           # [B, T, C]

        out = self.out_proj(context)
        return self.norm(residual + out)


class InterLeadAttention(nn.Module):
    """Cross-lead spatial attention capturing dynamic inter-lead coupling between Lead 0 and Lead 1.

    Divides the feature dimension into 2 lead representations (Lead 0 and Lead 1),
    projects each lead to query, key, value vectors, and computes a 2x2 attention matrix.

    Returns:
        tuple (transformed_features [B, T, C], attention_weights [B, 2, 2])
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels ({channels}) must be even to split into 2 leads")

        self.channels = channels
        self.lead_channels = channels // 2

        self.q_lead = nn.Linear(self.lead_channels, self.lead_channels, bias=False)
        self.k_lead = nn.Linear(self.lead_channels, self.lead_channels, bias=False)
        self.v_lead = nn.Linear(self.lead_channels, self.lead_channels, bias=False)
        self.out_lead = nn.Linear(self.lead_channels, self.lead_channels, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply inter-lead spatial attention.

        Args:
            x: Input tensor of shape [B, T, C] where C = 2 * lead_channels.

        Returns:
            tuple:
                out: Transformed tensor of shape [B, T, C].
                weights: 2x2 lead attention weights [B, 2, 2] summing to 1.0 along dim=-1.
        """
        B, T, C = x.shape
        residual = x

        # Split into lead 0 and lead 1: each [B, T, lead_channels]
        l0 = x[:, :, : self.lead_channels]
        l1 = x[:, :, self.lead_channels :]

        # Average pool over time to compute lead-level summary for spatial query/key: [B, 2, lead_channels]
        l0_summary = l0.mean(dim=1)  # [B, lead_channels]
        l1_summary = l1.mean(dim=1)  # [B, lead_channels]
        leads_summary = torch.stack([l0_summary, l1_summary], dim=1)  # [B, 2, lead_channels]

        q = self.q_lead(leads_summary)  # [B, 2, lead_channels]
        k = self.k_lead(leads_summary)  # [B, 2, lead_channels]

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.lead_channels)  # [B, 2, 2]
        weights = F.softmax(scores, dim=-1)  # [B, 2, 2]
        weights_dropped = self.dropout(weights)

        # Apply weights to values across time:
        # v0 = [B, T, lead_channels], v1 = [B, T, lead_channels]
        v0 = self.v_lead(l0)
        v1 = self.v_lead(l1)

        # l0_out = W_00 * v0 + W_01 * v1
        # l1_out = W_10 * v0 + W_11 * v1
        w00 = weights_dropped[:, 0, 0].view(B, 1, 1)
        w01 = weights_dropped[:, 0, 1].view(B, 1, 1)
        w10 = weights_dropped[:, 1, 0].view(B, 1, 1)
        w11 = weights_dropped[:, 1, 1].view(B, 1, 1)

        l0_new = self.out_lead(w00 * v0 + w01 * v1)
        l1_new = self.out_lead(w10 * v0 + w11 * v1)

        combined = torch.cat([l0_new, l1_new], dim=-1)  # [B, T, C]
        out = self.norm(residual + combined)
        return out, weights
