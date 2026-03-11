"""Tests for src/diagnostics/snr_analysis.py."""

import numpy as np
import pytest

from src.diagnostics.snr_analysis import compute_snr, plot_snr_distribution


@pytest.fixture
def synthetic_signal_with_known_snr():
    """Create a 2-channel signal where channel 0 has high-gamma power and channel 1 is noise."""
    rng = np.random.default_rng(42)
    fs = 500.0
    T = 2000
    t = np.arange(T) / fs

    sig = np.zeros((T, 2), dtype=np.float64)

    # Channel 0: strong 100 Hz component (in high gamma band) + weak noise
    sig[:, 0] = 5.0 * np.sin(2 * np.pi * 100 * t) + 0.1 * rng.standard_normal(T)

    # Channel 1: strong 60 Hz component (noise band) + weak signal
    sig[:, 1] = 5.0 * np.sin(2 * np.pi * 60 * t) + 0.1 * rng.standard_normal(T)

    return sig, fs


class TestComputeSNR:
    def test_returns_correct_type(self, synthetic_signal_with_known_snr):
        sig, fs = synthetic_signal_with_known_snr
        result = compute_snr(sig, fs=fs)
        assert hasattr(result, "snr_per_channel")
        assert hasattr(result, "low_quality_indices")
        assert hasattr(result, "signal_power")
        assert hasattr(result, "noise_power")

    def test_snr_shape(self, synthetic_signal_with_known_snr):
        sig, fs = synthetic_signal_with_known_snr
        result = compute_snr(sig, fs=fs)
        assert result.snr_per_channel.shape == (2,)

    def test_channel_0_higher_snr(self, synthetic_signal_with_known_snr):
        """Channel 0 has signal in high gamma band, so it should have higher SNR."""
        sig, fs = synthetic_signal_with_known_snr
        result = compute_snr(sig, fs=fs, signal_band=(70, 150), noise_band=(55, 65))
        assert result.snr_per_channel[0] > result.snr_per_channel[1]

    def test_channel_1_flagged_low_quality(self, synthetic_signal_with_known_snr):
        """Channel 1 should have low SNR (noise band is strong, signal band is weak)."""
        sig, fs = synthetic_signal_with_known_snr
        result = compute_snr(sig, fs=fs, signal_band=(70, 150), noise_band=(55, 65), threshold_db=10.0)
        assert 1 in result.low_quality_indices

    def test_signal_power_positive(self, synthetic_signal_with_known_snr):
        sig, fs = synthetic_signal_with_known_snr
        result = compute_snr(sig, fs=fs)
        assert np.all(result.signal_power >= 0)

    def test_noise_power_positive(self, synthetic_signal_with_known_snr):
        sig, fs = synthetic_signal_with_known_snr
        result = compute_snr(sig, fs=fs)
        assert np.all(result.noise_power >= 0)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_snr(np.zeros((100,)))


class TestPlotSNR:
    def test_returns_figure(self, synthetic_signal_with_known_snr):
        import matplotlib.pyplot as plt
        sig, fs = synthetic_signal_with_known_snr
        result = compute_snr(sig, fs=fs)
        fig = plot_snr_distribution(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
