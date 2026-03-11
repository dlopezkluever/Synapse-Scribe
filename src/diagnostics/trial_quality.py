"""Trial quality detection for neural recordings.

Flags trials with excessive variance or amplitude spikes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class TrialQualityResult:
    """Result of trial quality analysis."""

    n_total: int
    n_usable: int
    n_rejected: int
    rejected_indices: list[int]
    trial_variances: np.ndarray  # [n_trials]
    trial_max_amplitudes: np.ndarray  # [n_trials]
    reasons: dict[int, list[str]]  # trial_idx → rejection reasons


def detect_bad_trials(
    trials: list[np.ndarray],
    var_threshold: float = 5.0,
    amplitude_threshold: float = 10.0,
) -> TrialQualityResult:
    """Flag trials with abnormal variance or amplitude spikes.

    Args:
        trials: List of arrays, each [T, C] for one trial.
        var_threshold: Trials with variance > this × median trial variance are flagged.
        amplitude_threshold: Trials with max |amplitude| > this × median max amplitude are flagged.

    Returns:
        TrialQualityResult with per-trial quality info.
    """
    n_trials = len(trials)
    if n_trials == 0:
        return TrialQualityResult(
            n_total=0, n_usable=0, n_rejected=0,
            rejected_indices=[], trial_variances=np.array([]),
            trial_max_amplitudes=np.array([]), reasons={},
        )

    trial_vars = np.array([np.var(t) for t in trials])
    trial_max_amps = np.array([np.max(np.abs(t)) for t in trials])

    median_var = np.median(trial_vars)
    median_amp = np.median(trial_max_amps)

    reasons: dict[int, list[str]] = {}
    rejected = []

    for i in range(n_trials):
        trial_reasons = []

        # Variance check
        if median_var > 0 and trial_vars[i] > var_threshold * median_var:
            trial_reasons.append("high_variance")

        # Amplitude spike check
        if median_amp > 0 and trial_max_amps[i] > amplitude_threshold * median_amp:
            trial_reasons.append("amplitude_spike")

        if trial_reasons:
            reasons[i] = trial_reasons
            rejected.append(i)

    n_rejected = len(rejected)
    logger.info(
        "Trial quality: %d/%d usable, %d rejected (high_var=%d, amp_spike=%d)",
        n_trials - n_rejected, n_trials, n_rejected,
        sum(1 for r in reasons.values() if "high_variance" in r),
        sum(1 for r in reasons.values() if "amplitude_spike" in r),
    )

    return TrialQualityResult(
        n_total=n_trials,
        n_usable=n_trials - n_rejected,
        n_rejected=n_rejected,
        rejected_indices=rejected,
        trial_variances=trial_vars,
        trial_max_amplitudes=trial_max_amps,
        reasons=reasons,
    )


def plot_trial_quality_histogram(
    result: TrialQualityResult,
    figsize: tuple = (12, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot histogram of per-trial variance with rejected trials highlighted.

    Args:
        result: Output from detect_bad_trials().
        figsize: Figure size.
        ax: Optional pre-existing Axes.

    Returns:
        The matplotlib Figure.
    """
    if result.n_total == 0:
        fig, ax_new = plt.subplots(figsize=figsize)
        ax_new.text(0.5, 0.5, "No trials", ha="center", va="center")
        return fig

    show_new = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    good_mask = np.ones(result.n_total, dtype=bool)
    good_mask[result.rejected_indices] = False

    ax.hist(result.trial_variances[good_mask], bins=30, alpha=0.7,
            label=f"Usable ({result.n_usable})", color="steelblue", edgecolor="black")
    if result.n_rejected > 0:
        ax.hist(result.trial_variances[~good_mask], bins=15, alpha=0.7,
                label=f"Rejected ({result.n_rejected})", color="red", edgecolor="black")

    ax.set_xlabel("Trial Variance")
    ax.set_ylabel("Count")
    ax.set_title("Trial Quality Distribution")
    ax.legend()

    if show_new:
        fig.tight_layout()
    return fig
