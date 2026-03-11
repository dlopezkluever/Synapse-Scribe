"""Tests for src/preprocessing/normalize.py — z-score, channel rejection."""

import numpy as np
import pytest

from src.preprocessing.normalize import (
    compute_normalization_stats,
    zscore_normalize,
    detect_bad_channels_inline,
    remove_bad_channels,
    get_bad_channels,
    NormalizationStats,
)


@pytest.fixture
def training_signals():
    """Generate a list of training trial signals."""
    np.random.seed(42)
    return [
        np.random.randn(200, 10).astype(np.float32) * 2 + 5,
        np.random.randn(300, 10).astype(np.float32) * 2 + 5,
        np.random.randn(250, 10).astype(np.float32) * 2 + 5,
    ]


class TestNormalizationStats:
    def test_compute_stats_shape(self, training_signals):
        stats = compute_normalization_stats(training_signals)
        assert stats.mean.shape == (10,)
        assert stats.std.shape == (10,)

    def test_stats_values(self, training_signals):
        stats = compute_normalization_stats(training_signals)
        # Mean should be approximately 5, std approximately 2
        np.testing.assert_allclose(stats.mean, 5.0, atol=0.3)
        np.testing.assert_allclose(stats.std, 2.0, atol=0.3)

    def test_zero_variance_handling(self):
        """Channels with zero variance should get std=1 to avoid division by zero."""
        signals = [np.ones((100, 3), dtype=np.float32)]  # constant → zero variance
        stats = compute_normalization_stats(signals)
        assert np.all(stats.std == 1.0)

    def test_save_load(self, training_signals, tmp_path):
        stats = compute_normalization_stats(training_signals)
        save_path = tmp_path / "stats.npz"
        stats.save(save_path)
        loaded = NormalizationStats.load(save_path)
        np.testing.assert_array_equal(stats.mean, loaded.mean)
        np.testing.assert_array_equal(stats.std, loaded.std)


class TestZscoreNormalize:
    def test_output_zero_mean_unit_var(self, training_signals):
        stats = compute_normalization_stats(training_signals)
        all_data = np.concatenate(training_signals, axis=0)
        normalized = zscore_normalize(all_data, stats, clip=100.0)  # high clip to not interfere

        # Should be approximately zero mean, unit variance
        np.testing.assert_allclose(np.mean(normalized, axis=0), 0.0, atol=0.05)
        np.testing.assert_allclose(np.std(normalized, axis=0), 1.0, atol=0.05)

    def test_clipping(self, training_signals):
        stats = compute_normalization_stats(training_signals)
        # Create a signal with extreme values
        extreme = np.ones((10, 10), dtype=np.float32) * 1000
        normalized = zscore_normalize(extreme, stats, clip=5.0)
        assert np.all(normalized <= 5.0)
        assert np.all(normalized >= -5.0)

    def test_output_dtype(self, training_signals):
        stats = compute_normalization_stats(training_signals)
        signal = np.random.randn(50, 10).astype(np.float64)
        normalized = zscore_normalize(signal, stats)
        assert normalized.dtype == np.float32


class TestBadChannelDetection:
    def test_zero_variance_detected(self):
        signal = np.random.randn(100, 5).astype(np.float32)
        signal[:, 2] = 0.0  # zero variance channel
        bad = detect_bad_channels_inline(signal)
        assert 2 in bad

    def test_high_variance_detected(self):
        signal = np.random.randn(100, 5).astype(np.float32)
        signal[:, 3] *= 100  # extreme variance
        bad = detect_bad_channels_inline(signal, var_threshold=10.0)
        assert 3 in bad

    def test_clean_signal_no_bad(self):
        np.random.seed(0)
        signal = np.random.randn(100, 5).astype(np.float32)
        bad = detect_bad_channels_inline(signal)
        assert len(bad) == 0


class TestRemoveBadChannels:
    def test_removes_channels(self):
        signal = np.random.randn(100, 10).astype(np.float32)
        cleaned, kept = remove_bad_channels(signal, [2, 5, 7])
        assert cleaned.shape == (100, 7)
        assert len(kept) == 7
        assert 2 not in kept and 5 not in kept and 7 not in kept

    def test_no_removal(self):
        signal = np.random.randn(100, 10).astype(np.float32)
        cleaned, kept = remove_bad_channels(signal, [])
        assert cleaned.shape == signal.shape
        assert kept == list(range(10))


class TestGetBadChannels:
    def test_uses_diagnostics_when_available(self):
        signal = np.random.randn(100, 5).astype(np.float32)
        bad = get_bad_channels(signal, diagnostics_bad_indices=[1, 3])
        assert bad == [1, 3]

    def test_falls_back_to_inline(self):
        signal = np.random.randn(100, 5).astype(np.float32)
        signal[:, 0] = 0.0
        bad = get_bad_channels(signal, diagnostics_bad_indices=None)
        assert 0 in bad
