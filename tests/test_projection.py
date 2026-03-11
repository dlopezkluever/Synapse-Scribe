"""Tests for src/features/projection.py — Pathway B linear projection + positional encoding."""

import pytest
import torch

from src.features.projection import SinusoidalPositionalEncoding, LinearProjection


class TestSinusoidalPositionalEncoding:
    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=4096, dropout=0.0)
        x = torch.randn(2, 100, 512)
        out = pe(x)
        assert out.shape == (2, 100, 512)

    def test_positions_vary(self):
        """Positional encodings should differ across positions."""
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 50, 64)
        out = pe(x)
        # Different positions should have different encodings
        pos0 = out[0, 0]
        pos1 = out[0, 1]
        pos10 = out[0, 10]
        assert not torch.allclose(pos0, pos1)
        assert not torch.allclose(pos0, pos10)

    def test_deterministic_without_dropout(self):
        pe = SinusoidalPositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.zeros(1, 20, 32)
        out1 = pe(x)
        out2 = pe(x)
        torch.testing.assert_close(out1, out2)

    def test_encoding_values_bounded(self):
        """Sin/cos values should be in [-1, 1]."""
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 100, 64)
        out = pe(x)
        # Without input, output is just the PE
        assert out.max() <= 1.0 + 1e-6
        assert out.min() >= -1.0 - 1e-6


class TestLinearProjection:
    def test_output_shape(self):
        proj = LinearProjection(n_channels=192, d_model=512, dropout=0.0)
        x = torch.randn(4, 2000, 192)
        out = proj(x)
        assert out.shape == (4, 2000, 512)

    def test_output_shape_small(self):
        proj = LinearProjection(n_channels=32, d_model=128, dropout=0.0)
        x = torch.randn(2, 100, 32)
        out = proj(x)
        assert out.shape == (2, 100, 128)

    def test_gradient_flow(self):
        proj = LinearProjection(n_channels=32, d_model=64, dropout=0.0)
        x = torch.randn(2, 50, 32)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        for name, p in proj.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_positional_encoding_added(self):
        """Output should differ from pure linear projection due to PE."""
        proj = LinearProjection(n_channels=32, d_model=64, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = proj(x)
        # All-zero input through linear gives bias only,
        # but different positions get different PE
        assert not torch.allclose(out[0, 0], out[0, 1])
