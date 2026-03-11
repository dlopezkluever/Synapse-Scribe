"""Tests for src/features/firing_rate.py — Pathway C firing rate binning."""

import numpy as np
import pytest

from src.features.firing_rate import (
    bin_firing_rates,
    sqrt_transform,
    compute_firing_rate_features,
)


class TestBinFiringRates:
    def test_output_shape(self):
        """10 ms bins at 250 Hz → bin_size=2, so 1000 samples → 500 bins."""
        signals = np.random.rand(1000, 192).astype(np.float32)
        binned = bin_firing_rates(signals, bin_width_ms=10.0, fs=250.0)
        # bin_size = int(10/1000 * 250) = 2
        assert binned.shape == (500, 192)

    def test_output_shape_different_bin_width(self):
        """20 ms bins at 250 Hz → bin_size=5, so 1000 samples → 200 bins."""
        signals = np.random.rand(1000, 192).astype(np.float32)
        binned = bin_firing_rates(signals, bin_width_ms=20.0, fs=250.0)
        assert binned.shape == (200, 192)

    def test_binning_correctness(self):
        """Verify that binning sums correctly."""
        signals = np.ones((10, 4), dtype=np.float32)
        binned = bin_firing_rates(signals, bin_width_ms=20.0, fs=250.0)
        # bin_size = 5 → 2 bins, each summing 5 ones = 5.0
        assert binned.shape == (2, 4)
        np.testing.assert_allclose(binned, 5.0)

    def test_handles_non_divisible_length(self):
        """Truncation for non-divisible lengths."""
        signals = np.random.rand(103, 10).astype(np.float32)
        binned = bin_firing_rates(signals, bin_width_ms=10.0, fs=250.0)
        # bin_size = 2, n_bins = 103 // 2 = 51
        assert binned.shape == (51, 10)

    def test_output_dtype(self):
        signals = np.random.rand(100, 10).astype(np.float64)
        binned = bin_firing_rates(signals, bin_width_ms=10.0, fs=250.0)
        assert binned.dtype == np.float32


class TestSqrtTransform:
    def test_non_negative_output(self):
        """sqrt(abs(x)) should always be non-negative."""
        rates = np.random.randn(100, 10).astype(np.float32)  # may have negatives
        transformed = sqrt_transform(rates)
        assert np.all(transformed >= 0)

    def test_correct_values(self):
        rates = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32)
        transformed = sqrt_transform(rates)
        np.testing.assert_allclose(transformed, [[2.0, 3.0], [4.0, 5.0]])

    def test_output_dtype(self):
        rates = np.array([[1.0, 4.0]], dtype=np.float64)
        transformed = sqrt_transform(rates)
        assert transformed.dtype == np.float32


class TestComputeFiringRateFeatures:
    def test_end_to_end_shape(self):
        signals = np.random.rand(2000, 192).astype(np.float32)
        features = compute_firing_rate_features(signals, bin_width_ms=10.0, fs=250.0)
        assert features.shape == (1000, 192)

    def test_end_to_end_non_negative(self):
        signals = np.random.randn(500, 50).astype(np.float32)
        features = compute_firing_rate_features(signals, bin_width_ms=10.0, fs=250.0)
        assert np.all(features >= 0)
