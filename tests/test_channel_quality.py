"""Tests for src/diagnostics/channel_quality.py."""

import numpy as np
import pytest

from src.diagnostics.channel_quality import detect_bad_channels, plot_channel_variance_heatmap


@pytest.fixture
def normal_signals():
    """192-channel signal with normal variance."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 192)).astype(np.float32)


@pytest.fixture
def signals_with_bad_channels():
    """Signals with known bad channels injected."""
    rng = np.random.default_rng(42)
    sig = rng.standard_normal((500, 192)).astype(np.float32)

    # Channel 0: zero variance (flatline at 0)
    sig[:, 0] = 0.0

    # Channel 1: extremely high variance
    sig[:, 1] = rng.standard_normal(500).astype(np.float32) * 100.0

    # Channel 2: flatline segment (constant value for a stretch)
    sig[100:200, 2] = 5.0

    return sig


class TestDetectBadChannels:
    def test_returns_correct_type(self, normal_signals):
        result = detect_bad_channels(normal_signals)
        assert hasattr(result, "n_total")
        assert hasattr(result, "n_good")
        assert hasattr(result, "n_bad")
        assert hasattr(result, "bad_indices")
        assert hasattr(result, "labels")
        assert hasattr(result, "variances")

    def test_total_channels(self, normal_signals):
        result = detect_bad_channels(normal_signals)
        assert result.n_total == 192

    def test_good_plus_bad_equals_total(self, normal_signals):
        result = detect_bad_channels(normal_signals)
        assert result.n_good + result.n_bad == result.n_total

    def test_detects_zero_variance(self, signals_with_bad_channels):
        result = detect_bad_channels(signals_with_bad_channels)
        assert 0 in result.bad_indices
        assert "zero_var" in result.reasons[0]

    def test_detects_high_variance(self, signals_with_bad_channels):
        result = detect_bad_channels(signals_with_bad_channels)
        assert 1 in result.bad_indices
        assert "high_var" in result.reasons[1]

    def test_labels_length_matches_channels(self, normal_signals):
        result = detect_bad_channels(normal_signals)
        assert len(result.labels) == 192

    def test_variances_shape(self, normal_signals):
        result = detect_bad_channels(normal_signals)
        assert result.variances.shape == (192,)

    def test_mostly_good_on_normal_data(self, normal_signals):
        result = detect_bad_channels(normal_signals)
        # With random normal data most channels should be good
        assert result.n_good >= 180

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="2-D"):
            detect_bad_channels(np.zeros((100,)))

    def test_flatline_detection(self):
        """A channel that is constant for a long stretch should be flagged."""
        sig = np.random.default_rng(0).standard_normal((1000, 10)).astype(np.float32)
        sig[:, 5] = 3.0  # entire channel is constant
        result = detect_bad_channels(sig, fs=250.0, flatline_threshold_s=0.1)
        assert 5 in result.bad_indices


class TestPlotChannelVariance:
    def test_returns_figure(self, normal_signals):
        import matplotlib.pyplot as plt
        result = detect_bad_channels(normal_signals)
        fig = plot_channel_variance_heatmap(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
