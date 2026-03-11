"""Tests for src/diagnostics/spectral_analysis.py."""

import numpy as np
import pytest

from src.diagnostics.spectral_analysis import compute_psd, plot_power_spectrum


@pytest.fixture
def simple_signal():
    """Multi-channel signal with known spectral content."""
    rng = np.random.default_rng(42)
    fs = 500.0
    T = 2000
    t = np.arange(T) / fs
    C = 10

    sig = rng.standard_normal((T, C)).astype(np.float64) * 0.1
    # Add a 60 Hz component to all channels
    for ch in range(C):
        sig[:, ch] += 2.0 * np.sin(2 * np.pi * 60 * t)

    return sig, fs


class TestComputePSD:
    def test_returns_correct_type(self, simple_signal):
        sig, fs = simple_signal
        result = compute_psd(sig, fs=fs)
        assert hasattr(result, "freqs")
        assert hasattr(result, "psd")
        assert hasattr(result, "psd_mean")
        assert hasattr(result, "line_noise_detected")

    def test_psd_shape(self, simple_signal):
        sig, fs = simple_signal
        result = compute_psd(sig, fs=fs)
        n_freqs = len(result.freqs)
        assert result.psd.shape == (n_freqs, 10)
        assert result.psd_mean.shape == (n_freqs,)

    def test_frequency_resolution(self, simple_signal):
        sig, fs = simple_signal
        result = compute_psd(sig, fs=fs, nperseg=256)
        # Frequency resolution = fs / nperseg
        expected_resolution = fs / 256
        actual_resolution = result.freqs[1] - result.freqs[0]
        assert abs(actual_resolution - expected_resolution) < 0.01

    def test_max_frequency(self, simple_signal):
        sig, fs = simple_signal
        result = compute_psd(sig, fs=fs)
        # Nyquist frequency
        assert result.freqs[-1] <= fs / 2

    def test_detects_line_noise(self, simple_signal):
        """Signal has strong 60 Hz → line noise should be detected."""
        sig, fs = simple_signal
        result = compute_psd(sig, fs=fs, line_noise_freq=60.0)
        assert result.line_noise_detected is True

    def test_psd_non_negative(self, simple_signal):
        sig, fs = simple_signal
        result = compute_psd(sig, fs=fs)
        assert np.all(result.psd >= 0)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_psd(np.zeros((100,)))


class TestPlotPowerSpectrum:
    def test_returns_figure(self, simple_signal):
        import matplotlib.pyplot as plt
        sig, fs = simple_signal
        result = compute_psd(sig, fs=fs)
        fig = plot_power_spectrum(result, channels=[0, 1])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
