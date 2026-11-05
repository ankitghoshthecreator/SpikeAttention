"""Surrogate gradient activation functions for non-differentiable spiking thresholds."""

from __future__ import annotations

import math
import torch
from torch.autograd import Function


class _ArctanFunction(Function):
    """Autograd Function with exact Heaviside forward and smooth Arctan backward surrogate."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # d/dx (1/pi * arctan(alpha * x) + 0.5) = 1/pi * alpha / (1 + (alpha * x)^2)
        # Scaled form: (alpha / 2) / (1 + (pi * alpha * x / 2)^2)
        arg = (math.pi * alpha * x) / 2.0
        grad_input = grad_output * (alpha / 2.0) / (1.0 + arg * arg)
        return grad_input, None


def surrogate_heaviside(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Evaluate Heaviside step with smooth Arctan surrogate gradient in backward pass.

    Args:
        x: Input tensor representing membrane potential relative to threshold: (V - V_th).
        alpha: Steepness parameter controlling surrogate gradient bandwidth.

    Returns:
        Binary spike tensor of the same shape and device as x.
    """
    return _ArctanFunction.apply(x, alpha)


class ArctanSurrogate(torch.nn.Module):
    """PyTorch Module wrapper for Arctan surrogate gradient activation."""

    def __init__(self, alpha: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return surrogate_heaviside(x, self.alpha)
