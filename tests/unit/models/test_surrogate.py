"""Unit tests for ArctanSurrogate gradient activation."""

import pytest
import torch

from spikeattn.models.snn.surrogate import ArctanSurrogate, surrogate_heaviside


class TestArctanSurrogate:
    def test_forward_heaviside_step(self):
        x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
        s = surrogate_heaviside(x, alpha=2.0)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(s, expected)

    def test_gradient_flow_non_zero(self):
        """Standard Heaviside has zero gradient everywhere; surrogate provides non-zero gradient."""
        x = torch.tensor([0.0], requires_grad=True)
        s = surrogate_heaviside(x, alpha=2.0)
        s.backward()
        assert x.grad is not None
        assert x.grad.item() > 0.0

    def test_maximum_gradient_at_threshold(self):
        """Surrogate derivative has its global maximum at x = 0 (V = V_th)."""
        x_at_thresh = torch.tensor([0.0], requires_grad=True)
        x_off_thresh = torch.tensor([1.5], requires_grad=True)

        s1 = surrogate_heaviside(x_at_thresh, alpha=2.0)
        s1.backward()

        s2 = surrogate_heaviside(x_off_thresh, alpha=2.0)
        s2.backward()

        assert x_at_thresh.grad.item() > x_off_thresh.grad.item()

    def test_module_wrapper_matches_function(self):
        module = ArctanSurrogate(alpha=3.0)
        x = torch.tensor([-0.5, 0.0, 0.5])
        s_mod = module(x)
        s_fn = surrogate_heaviside(x, alpha=3.0)
        torch.testing.assert_close(s_mod, s_fn)
