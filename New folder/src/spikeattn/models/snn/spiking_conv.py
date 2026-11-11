"""Spiking 1D convolutional block with temporal unrolling and LIF dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from spikeattn.models.snn.lif import LIFNode


class SpikingConv1dBlock(nn.Module):
    """Spiking 1D Convolutional Block: Conv1d -> BatchNorm1d -> LIFNode.

    Accepts spike or feature tensors of shape [B, T, C_in], applies 1D temporal convolution
    across the time dimension, and activates with a LIF spiking neuron layer.

    Args:
        in_channels: Number of input channels C_in.
        out_channels: Number of output feature maps C_out.
        kernel_size: Convolution kernel size along the temporal axis.
        stride: Temporal stride (default 1).
        padding: Temporal zero-padding (default 'same' or int).
        tau: Membrane time constant for the LIF layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int | str = "same",
        tau: float = 10.0,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        if padding == "same" and stride != 1:
            padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.lif = LIFNode(tau=tau, alpha=alpha)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, T, C_in].

        Returns:
            tuple (spikes, voltages) of shape [B, T_out, C_out].
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, T, C], got shape {x.shape}")

        # Permute to [B, C_in, T] for PyTorch Conv1d
        x_perm = x.permute(0, 2, 1)
        conv_out = self.conv(x_perm)
        bn_out = self.bn(conv_out)
        # Permute back to [B, T_out, C_out]
        current = bn_out.permute(0, 2, 1)

        spikes, voltages = self.lif(current)
        return spikes, voltages
