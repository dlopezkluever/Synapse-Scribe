"""Tests for src/features/temporal_conv.py — Pathway A temporal convolution bank."""

import pytest
import torch

from src.features.temporal_conv import TemporalConvBank


@pytest.fixture
def small_input():
    """Small input: [B=2, T=100, C=32]."""
    return torch.randn(2, 100, 32)


@pytest.fixture
def standard_input():
    """Standard input: [B=4, T=2000, C=192]."""
    return torch.randn(4, 2000, 192)


class TestTemporalConvBank:
    def test_output_shape_no_pooling(self, small_input):
        bank = TemporalConvBank(n_channels=32, use_pooling=False)
        out = bank(small_input)
        # 3 kernels × 256 channels = 768
        assert out.shape == (2, 100, 768)

    def test_output_shape_with_pooling(self, small_input):
        bank = TemporalConvBank(n_channels=32, use_pooling=True)
        out = bank(small_input)
        # T halved by pooling
        assert out.shape == (2, 50, 768)

    def test_output_shape_standard(self, standard_input):
        bank = TemporalConvBank(n_channels=192, use_pooling=False)
        out = bank(standard_input)
        assert out.shape == (4, 2000, 768)

    def test_custom_kernel_sizes(self, small_input):
        bank = TemporalConvBank(
            n_channels=32,
            out_channels_per_kernel=128,
            kernel_sizes=(3, 5),
            use_pooling=False,
        )
        out = bank(small_input)
        # 2 kernels × 128 = 256
        assert out.shape == (2, 100, 256)

    def test_gradient_flow(self, small_input):
        bank = TemporalConvBank(n_channels=32)
        out = bank(small_input)
        loss = out.sum()
        loss.backward()
        for name, p in bank.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_output_channels_attribute(self):
        bank = TemporalConvBank(n_channels=32, out_channels_per_kernel=64, kernel_sizes=(3, 7, 15))
        assert bank.output_channels == 64 * 3

    def test_different_input_channels(self):
        for n_ch in [10, 64, 192]:
            bank = TemporalConvBank(n_channels=n_ch)
            x = torch.randn(1, 50, n_ch)
            out = bank(x)
            assert out.shape == (1, 50, 768)

    def test_odd_sequence_length_with_pooling(self):
        bank = TemporalConvBank(n_channels=16, use_pooling=True)
        x = torch.randn(1, 101, 16)
        out = bank(x)
        # MaxPool1d with stride 2: floor(101/2) = 50
        assert out.shape[1] == 50
