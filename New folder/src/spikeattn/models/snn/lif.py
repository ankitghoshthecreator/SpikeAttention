"""Leaky Integrate-and-Fire (LIF) spiking neuron node with surrogate gradient dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn

from spikeattn.models.snn.surrogate import surrogate_heaviside


class LIFNode(nn.Module):
    """Discretized Leaky Integrate-and-Fire (LIF) neuron model with hard voltage reset.

    Dynamics:
        V[t] = V[t-1] + (dt / tau) * (-(V[t-1] - V_reset) + I[t])
        S[t] = Heaviside(V[t] - V_th)
        V[t] = V[t] * (1 - S[t]) + V_reset * S[t]

    Args:
        tau: Membrane time constant (default 10.0 ms).
        dt: Simulation time step (default 1.0 ms).
        v_th: Firing threshold potential (default 1.0).
        v_reset: Reset potential after firing (default 0.0).
        alpha: Surrogate gradient steepness parameter.
        learnable_tau: Whether the decay factor is a learnable parameter.
    """

    def __init__(
        self,
        tau: float = 10.0,
        dt: float = 1.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        alpha: float = 2.0,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        self.dt = float(dt)
        self.v_th = float(v_th)
        self.v_reset = float(v_reset)
        self.alpha = float(alpha)

        init_decay = 1.0 - (self.dt / self.tau)
        if learnable_tau:
            # Parametrized via sigmoid to guarantee decay in (0, 1)
            raw_decay = torch.logit(torch.tensor(init_decay, dtype=torch.float32))
            self.raw_decay = nn.Parameter(raw_decay)
        else:
            self.register_buffer("raw_decay", None)
            self.register_buffer("fixed_decay", torch.tensor(init_decay, dtype=torch.float32))

        self.register_buffer("v_mem", None)

    @property
    def decay(self) -> torch.Tensor:
        """Effective decay factor (1 - dt / tau) in (0, 1)."""
        if self.raw_decay is not None:
            return torch.sigmoid(self.raw_decay)
        return self.fixed_decay  # type: ignore[return-value]

    def reset_membrane_potential(self) -> None:
        """Reset internal membrane state to None (re-initialized on next forward)."""
        self.v_mem = None

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input current tensor.

        If x is 3D [B, T, C], simulates all T time steps sequentially.
        If x is 2D [B, C], executes a single time step using persistent membrane state.

        Args:
            x: Input current tensor of shape [B, T, C] or [B, C].

        Returns:
            tuple (spikes, membrane_voltages) matching the input time dimension.
        """
        if x.ndim == 3:
            B, T, C = x.shape
            spikes_list = []
            v_list = []
            v = torch.full((B, C), self.v_reset, device=x.device, dtype=x.dtype)
            decay = self.decay.to(x.device, dtype=x.dtype)

            for t in range(T):
                i_t = x[:, t, :]
                # Numerical Euler update: V[t] = beta * (V[t-1] - V_reset) + V_reset + I[t]
                v = decay * (v - self.v_reset) + self.v_reset + i_t
                s = surrogate_heaviside(v - self.v_th, self.alpha)
                # Hard reset
                v = v * (1.0 - s) + self.v_reset * s

                spikes_list.append(s)
                v_list.append(v)

            spikes = torch.stack(spikes_list, dim=1)  # [B, T, C]
            voltages = torch.stack(v_list, dim=1)     # [B, T, C]
            return spikes, voltages

        elif x.ndim == 2:
            B, C = x.shape
            if self.v_mem is None or self.v_mem.shape != (B, C) or self.v_mem.device != x.device:
                self.v_mem = torch.full((B, C), self.v_reset, device=x.device, dtype=x.dtype)

            decay = self.decay.to(x.device, dtype=x.dtype)
            v = decay * (self.v_mem - self.v_reset) + self.v_reset + x
            s = surrogate_heaviside(v - self.v_th, self.alpha)
            v = v * (1.0 - s) + self.v_reset * s
            self.v_mem = v.detach()
            return s, v

        else:
            raise ValueError(f"LIFNode expects 2D [B, C] or 3D [B, T, C] input, got ndim={x.ndim}")
