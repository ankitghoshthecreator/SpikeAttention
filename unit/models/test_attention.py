"""Unit tests for TemporalAttention and InterLeadAttention modules."""

import pytest
import torch

from spikeattn.models.snn.attention import InterLeadAttention, TemporalAttention


class TestTemporalAttention:
    def test_output_shape_matches_input(self):
        B, T, C = 4, 30, 32
        attn = TemporalAttention(channels=C, n_heads=4, dropout=0.0)
        x = torch.randn(B, T, C)
        out = attn(x)
        assert out.shape == (B, T, C)

    def test_gradient_flow(self):
        B, T, C = 2, 10, 16
        attn = TemporalAttention(channels=C, n_heads=2)
        x = torch.randn(B, T, C, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_indivisible_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            TemporalAttention(channels=10, n_heads=4)


class TestInterLeadAttention:
    def test_output_shape_and_weights_shape(self):
        B, T, C = 3, 25, 32
        lead_attn = InterLeadAttention(channels=C, dropout=0.0)
        x = torch.randn(B, T, C)
        out, weights = lead_attn(x)

        assert out.shape == (B, T, C)
        assert weights.shape == (B, 2, 2)

    def test_weights_sum_to_one_across_keys(self):
        """Attention weights for each query lead must sum to 1.0 across key leads."""
        B, T, C = 4, 20, 16
        lead_attn = InterLeadAttention(channels=C, dropout=0.0)
        x = torch.randn(B, T, C)
        _, weights = lead_attn(x)

        row_sums = weights.sum(dim=-1)  # [B, 2]
        expected_sums = torch.ones((B, 2), dtype=weights.dtype)
        torch.testing.assert_close(row_sums, expected_sums, atol=1e-5, rtol=1e-5)

    def test_odd_channels_raises(self):
        with pytest.raises(ValueError, match="even"):
            InterLeadAttention(channels=15)
