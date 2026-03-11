"""Tests for src/diagnostics/correlation_analysis.py."""

import numpy as np
import pytest

from src.diagnostics.correlation_analysis import compute_channel_correlation, plot_correlation_matrix


@pytest.fixture
def independent_signals():
    """Channels with independent random noise (low correlation)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1000, 20)).astype(np.float64)


@pytest.fixture
def correlated_signals():
    """Signals where some channels are highly correlated."""
    rng = np.random.default_rng(42)
    sig = rng.standard_normal((1000, 10)).astype(np.float64)

    # Make channel 1 = channel 0 + tiny noise (correlation > 0.99)
    sig[:, 1] = sig[:, 0] + 0.01 * rng.standard_normal(1000)

    # Make channel 5 = -channel 4 + tiny noise (anti-correlated)
    sig[:, 5] = -sig[:, 4] + 0.01 * rng.standard_normal(1000)

    return sig


class TestComputeChannelCorrelation:
    def test_returns_correct_type(self, independent_signals):
        result = compute_channel_correlation(independent_signals)
        assert hasattr(result, "correlation_matrix")
        assert hasattr(result, "high_corr_pairs")
        assert hasattr(result, "n_high_corr_pairs")
        assert hasattr(result, "threshold")

    def test_matrix_shape(self, independent_signals):
        result = compute_channel_correlation(independent_signals)
        assert result.correlation_matrix.shape == (20, 20)

    def test_matrix_symmetric(self, independent_signals):
        result = compute_channel_correlation(independent_signals)
        np.testing.assert_allclose(
            result.correlation_matrix,
            result.correlation_matrix.T,
            atol=1e-10,
        )

    def test_diagonal_is_one(self, independent_signals):
        result = compute_channel_correlation(independent_signals)
        np.testing.assert_allclose(
            np.diag(result.correlation_matrix), 1.0, atol=1e-10,
        )

    def test_values_in_range(self, independent_signals):
        result = compute_channel_correlation(independent_signals)
        assert np.all(result.correlation_matrix >= -1.0 - 1e-10)
        assert np.all(result.correlation_matrix <= 1.0 + 1e-10)

    def test_detects_high_correlation(self, correlated_signals):
        result = compute_channel_correlation(correlated_signals, threshold=0.95)
        pair_channels = [(p[0], p[1]) for p in result.high_corr_pairs]
        assert (0, 1) in pair_channels

    def test_few_high_corr_on_independent(self, independent_signals):
        result = compute_channel_correlation(independent_signals, threshold=0.95)
        assert result.n_high_corr_pairs == 0

    def test_handles_zero_variance_channel(self):
        sig = np.random.default_rng(0).standard_normal((500, 5)).astype(np.float64)
        sig[:, 2] = 0.0  # zero variance
        result = compute_channel_correlation(sig)
        # Should not crash; diagonal still 1
        assert result.correlation_matrix[2, 2] == 1.0
        # Off-diagonal for zero-var channel should be 0
        assert result.correlation_matrix[2, 0] == 0.0

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_channel_correlation(np.zeros((100,)))


class TestPlotCorrelationMatrix:
    def test_returns_figure(self, independent_signals):
        import matplotlib.pyplot as plt
        result = compute_channel_correlation(independent_signals)
        fig = plot_correlation_matrix(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
