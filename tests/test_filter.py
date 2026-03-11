"""Tests for src/preprocessing/filter.py — bandpass, notch, downsample, smoothing."""

import numpy as np
import pytest

from src.preprocessing.filter import (
    bandpass_filter,
    notch_filter,
    artifact_rejection,
    temporal_downsample,
    GaussianTemporalSmoothing,
)


@pytest.fixture
def sample_signal():
    """Create a synthetic signal with known frequency content: 10 Hz + 60 Hz + 200 Hz."""
    fs = 1000.0
    T = 2.0
    t = np.arange(0, T, 1 / fs)
    C = 8
    # 10 Hz (wanted), 60 Hz (line noise), 200 Hz (to be filtered out with low bandpass)
    signal = np.zeros((len(t), C), dtype=np.float64)
    for ch in range(C):
        signal[:, ch] = (
            np.sin(2 * np.pi * 10 * t)
            + 0.5 * np.sin(2 * np.pi * 60 * t)
            + 0.3 * np.sin(2 * np.pi * 200 * t)
        )
    return signal, fs


class TestBandpassFilter:
    def test_shape_preserved(self, sample_signal):
        signal, fs = sample_signal
        filtered = bandpass_filter(signal, fs, low=5.0, high=100.0)
        assert filtered.shape == signal.shape

    def test_attenuates_outside_passband(self, sample_signal):
        """Verify that frequencies outside the passband are attenuated."""
        signal, fs = sample_signal
        # Keep only 5-50 Hz → 200 Hz is well outside the passband
        filtered = bandpass_filter(signal, fs, low=5.0, high=50.0)

        # Check via FFT that 200 Hz is heavily attenuated
        freqs = np.fft.rfftfreq(filtered.shape[0], 1 / fs)
        spectrum_orig = np.abs(np.fft.rfft(signal[:, 0]))
        spectrum_filt = np.abs(np.fft.rfft(filtered[:, 0]))

        idx_200 = np.argmin(np.abs(freqs - 200))
        assert spectrum_filt[idx_200] < spectrum_orig[idx_200] * 0.1

    def test_preserves_passband(self, sample_signal):
        """10 Hz should remain largely intact in a 5-100 Hz bandpass."""
        signal, fs = sample_signal
        filtered = bandpass_filter(signal, fs, low=5.0, high=100.0)

        freqs = np.fft.rfftfreq(filtered.shape[0], 1 / fs)
        spectrum_orig = np.abs(np.fft.rfft(signal[:, 0]))
        spectrum_filt = np.abs(np.fft.rfft(filtered[:, 0]))

        idx_10 = np.argmin(np.abs(freqs - 10))
        # 10 Hz should retain at least 80% of its power
        assert spectrum_filt[idx_10] > spectrum_orig[idx_10] * 0.8


class TestNotchFilter:
    def test_shape_preserved(self, sample_signal):
        signal, fs = sample_signal
        filtered = notch_filter(signal, fs, freqs=[60.0])
        assert filtered.shape == signal.shape

    def test_removes_60hz(self, sample_signal):
        signal, fs = sample_signal
        filtered = notch_filter(signal, fs, freqs=[60.0])

        freqs = np.fft.rfftfreq(filtered.shape[0], 1 / fs)
        spectrum_orig = np.abs(np.fft.rfft(signal[:, 0]))
        spectrum_filt = np.abs(np.fft.rfft(filtered[:, 0]))

        idx_60 = np.argmin(np.abs(freqs - 60))
        assert spectrum_filt[idx_60] < spectrum_orig[idx_60] * 0.3

    def test_skips_high_freq(self):
        """Notch at 200 Hz with fs=300 should still work (under Nyquist)."""
        fs = 300.0
        t = np.arange(0, 1, 1 / fs)
        signal = np.sin(2 * np.pi * 100 * t).reshape(-1, 1)
        filtered = notch_filter(signal, fs, freqs=[100.0])
        assert filtered.shape == signal.shape


class TestArtifactRejection:
    def test_clean_trial_not_rejected(self):
        signal = np.random.randn(500, 10).astype(np.float64)
        _, rejected = artifact_rejection(signal, threshold=3.0)
        assert not rejected

    def test_artifact_trial_rejected(self):
        signal = np.random.randn(500, 10).astype(np.float64)
        # Make one channel have extreme variance
        signal[:, 5] *= 100
        _, rejected = artifact_rejection(signal, threshold=3.0)
        assert rejected


class TestTemporalDownsample:
    def test_downsample_2x(self):
        signal = np.random.randn(1000, 10).astype(np.float64)
        ds = temporal_downsample(signal, current_fs=500.0, target_fs=250.0)
        assert ds.shape[1] == signal.shape[1]
        assert ds.shape[0] == 500

    def test_no_downsample_if_same_rate(self):
        signal = np.random.randn(1000, 10).astype(np.float64)
        ds = temporal_downsample(signal, current_fs=250.0, target_fs=250.0)
        assert ds.shape == signal.shape


class TestGaussianTemporalSmoothing:
    def test_shape_preserved(self):
        smoother = GaussianTemporalSmoothing(sigma_ms=30.0, fs=250.0)
        signal = np.random.randn(500, 10).astype(np.float32)
        smoothed = smoother(signal)
        assert smoothed.shape == signal.shape

    def test_reduces_high_freq_variance(self):
        """Smoothed signal should have less high-frequency content."""
        smoother = GaussianTemporalSmoothing(sigma_ms=30.0, fs=250.0)
        np.random.seed(42)
        signal = np.random.randn(1000, 4).astype(np.float32)
        smoothed = smoother(signal)

        # Check that diff variance is reduced (proxy for high-freq content)
        orig_diff_var = np.var(np.diff(signal, axis=0))
        smooth_diff_var = np.var(np.diff(smoothed, axis=0))
        assert smooth_diff_var < orig_diff_var

    def test_small_sigma_is_near_identity(self):
        """Very small sigma should barely change the signal."""
        smoother = GaussianTemporalSmoothing(sigma_ms=0.1, fs=250.0)
        signal = np.random.randn(100, 4).astype(np.float32)
        smoothed = smoother(signal)
        np.testing.assert_allclose(smoothed, signal, atol=1e-3)
