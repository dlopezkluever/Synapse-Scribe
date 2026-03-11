"""Tests for src/diagnostics/trial_quality.py."""

import numpy as np
import pytest

from src.diagnostics.trial_quality import detect_bad_trials, plot_trial_quality_histogram


@pytest.fixture
def normal_trials():
    """List of trials with normal variance."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal((300, 192)).astype(np.float32) for _ in range(20)]


@pytest.fixture
def trials_with_artifacts():
    """Trials with injected high-variance and amplitude-spike artifacts."""
    rng = np.random.default_rng(42)
    trials = [rng.standard_normal((300, 192)).astype(np.float32) for _ in range(20)]

    # Trial 3: high variance artifact
    trials[3] = rng.standard_normal((300, 192)).astype(np.float32) * 50.0

    # Trial 7: amplitude spike
    trials[7] = rng.standard_normal((300, 192)).astype(np.float32)
    trials[7][150, :] = 500.0  # single massive spike

    return trials


class TestDetectBadTrials:
    def test_returns_correct_type(self, normal_trials):
        result = detect_bad_trials(normal_trials)
        assert hasattr(result, "n_total")
        assert hasattr(result, "n_usable")
        assert hasattr(result, "n_rejected")
        assert hasattr(result, "rejected_indices")
        assert hasattr(result, "trial_variances")

    def test_total_count(self, normal_trials):
        result = detect_bad_trials(normal_trials)
        assert result.n_total == 20

    def test_usable_plus_rejected_equals_total(self, normal_trials):
        result = detect_bad_trials(normal_trials)
        assert result.n_usable + result.n_rejected == result.n_total

    def test_detects_high_variance_trial(self, trials_with_artifacts):
        result = detect_bad_trials(trials_with_artifacts)
        assert 3 in result.rejected_indices
        assert "high_variance" in result.reasons[3]

    def test_detects_amplitude_spike(self, trials_with_artifacts):
        result = detect_bad_trials(trials_with_artifacts)
        assert 7 in result.rejected_indices
        assert "amplitude_spike" in result.reasons[7]

    def test_trial_variances_shape(self, normal_trials):
        result = detect_bad_trials(normal_trials)
        assert result.trial_variances.shape == (20,)

    def test_trial_max_amplitudes_shape(self, normal_trials):
        result = detect_bad_trials(normal_trials)
        assert result.trial_max_amplitudes.shape == (20,)

    def test_mostly_usable_on_normal_data(self, normal_trials):
        result = detect_bad_trials(normal_trials)
        assert result.n_usable >= 18

    def test_empty_trials(self):
        result = detect_bad_trials([])
        assert result.n_total == 0
        assert result.n_usable == 0
        assert result.n_rejected == 0


class TestPlotTrialQuality:
    def test_returns_figure(self, normal_trials):
        import matplotlib.pyplot as plt
        result = detect_bad_trials(normal_trials)
        fig = plot_trial_quality_histogram(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_trials_returns_figure(self):
        import matplotlib.pyplot as plt
        result = detect_bad_trials([])
        fig = plot_trial_quality_histogram(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
