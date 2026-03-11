"""Firing rate binning (Pathway C) for Willett spike count data.

Bins spike counts into non-overlapping windows and applies a square-root
transform to stabilize variance. This is a data-level preprocessing step
applied before any model.
"""

from __future__ import annotations

import numpy as np


def bin_firing_rates(
    signals: np.ndarray,
    bin_width_ms: float = 10.0,
    fs: float = 250.0,
) -> np.ndarray:
    """Bin spike counts into non-overlapping time windows.

    Args:
        signals: Array [T, C] of raw spike counts.
        bin_width_ms: Bin width in milliseconds.
        fs: Sampling rate in Hz.

    Returns:
        Binned array [n_bins, C] where n_bins = T // bin_size.
    """
    bin_size = max(1, int(bin_width_ms / 1000.0 * fs))
    T, C = signals.shape
    n_bins = T // bin_size

    if n_bins == 0:
        return signals[:1]  # return at least 1 bin

    # Truncate to an exact multiple of bin_size, then reshape and sum
    truncated = signals[: n_bins * bin_size]
    reshaped = truncated.reshape(n_bins, bin_size, C)
    binned = reshaped.sum(axis=1).astype(np.float32)
    return binned


def sqrt_transform(firing_rates: np.ndarray) -> np.ndarray:
    """Apply square-root transform to stabilize variance.

    Args:
        firing_rates: Array [n_bins, C] of binned firing rates.

    Returns:
        Transformed array with same shape, all values >= 0.
    """
    return np.sqrt(np.abs(firing_rates)).astype(np.float32)


def compute_firing_rate_features(
    signals: np.ndarray,
    bin_width_ms: float = 10.0,
    fs: float = 250.0,
) -> np.ndarray:
    """Full Pathway C: bin firing rates then apply sqrt transform.

    Args:
        signals: Array [T, C] of spike counts.
        bin_width_ms: Bin width in milliseconds.
        fs: Sampling rate in Hz.

    Returns:
        Array [n_bins, C] of sqrt-transformed binned firing rates.
    """
    binned = bin_firing_rates(signals, bin_width_ms, fs)
    return sqrt_transform(binned)
