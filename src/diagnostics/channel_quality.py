"""Channel quality detection for neural recordings.

Flags channels with zero variance, excessive variance, line noise, or flatline segments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class ChannelQualityResult:
    """Result of channel quality analysis."""

    n_total: int
    n_good: int
    n_bad: int
    bad_indices: list[int]
    labels: np.ndarray  # per-channel: "good", "zero_var", "high_var", "line_noise", "flatline"
    variances: np.ndarray
    reasons: dict[int, list[str]]  # channel_idx → list of rejection reasons


def detect_bad_channels(
    signals: np.ndarray,
    fs: float = 250.0,
    var_threshold: float = 10.0,
    flatline_threshold_s: float = 0.1,
    line_noise_freq: float = 60.0,
    line_noise_ratio: float = 5.0,
) -> ChannelQualityResult:
    """Flag bad channels in a neural recording.

    Args:
        signals: Array of shape [T, C] (timesteps × channels).
        fs: Sampling rate in Hz.
        var_threshold: Channels with variance > this × median variance are flagged.
        flatline_threshold_s: Minimum flatline duration (seconds) to flag a channel.
        line_noise_freq: Line noise frequency to check (Hz).
        line_noise_ratio: If power at line_noise_freq > this × median spectral power,
            the channel is flagged for excessive line noise.

    Returns:
        ChannelQualityResult with per-channel labels and summary.
    """
    if signals.ndim != 2:
        raise ValueError(f"Expected 2-D array [T, C], got shape {signals.shape}")

    T, C = signals.shape
    variances = np.var(signals, axis=0)
    median_var = np.median(variances)

    labels = np.array(["good"] * C, dtype=object)
    reasons: dict[int, list[str]] = {}

    def _flag(ch: int, reason: str) -> None:
        if labels[ch] == "good":
            labels[ch] = reason
        if ch not in reasons:
            reasons[ch] = []
        reasons[ch].append(reason)

    # 1. Zero variance
    for ch in range(C):
        if variances[ch] == 0:
            _flag(ch, "zero_var")

    # 2. Excessive variance (> var_threshold × median)
    if median_var > 0:
        for ch in range(C):
            if variances[ch] > var_threshold * median_var:
                _flag(ch, "high_var")

    # 3. Flatline detection: consecutive identical values
    flatline_samples = int(flatline_threshold_s * fs)
    if flatline_samples >= 2:
        for ch in range(C):
            if labels[ch] == "zero_var":
                continue
            diffs = np.diff(signals[:, ch])
            run_length = 1
            for d in diffs:
                if d == 0:
                    run_length += 1
                    if run_length >= flatline_samples:
                        _flag(ch, "flatline")
                        break
                else:
                    run_length = 1

    # 4. Line noise detection via FFT
    if T > 1 and fs > 2 * line_noise_freq:
        freqs = np.fft.rfftfreq(T, d=1.0 / fs)
        # Find the bin closest to line_noise_freq
        line_idx = np.argmin(np.abs(freqs - line_noise_freq))
        if line_idx > 0:
            for ch in range(C):
                if labels[ch] == "zero_var":
                    continue
                spectrum = np.abs(np.fft.rfft(signals[:, ch])) ** 2
                median_power = np.median(spectrum)
                if median_power > 0 and spectrum[line_idx] > line_noise_ratio * median_power:
                    _flag(ch, "line_noise")

    bad_indices = [i for i in range(C) if labels[i] != "good"]
    n_bad = len(bad_indices)

    logger.info(
        "Channel quality: %d/%d good, %d bad (zero_var=%d, high_var=%d, flatline=%d, line_noise=%d)",
        C - n_bad, C, n_bad,
        sum(1 for r in reasons.values() if "zero_var" in r),
        sum(1 for r in reasons.values() if "high_var" in r),
        sum(1 for r in reasons.values() if "flatline" in r),
        sum(1 for r in reasons.values() if "line_noise" in r),
    )

    return ChannelQualityResult(
        n_total=C,
        n_good=C - n_bad,
        n_bad=n_bad,
        bad_indices=bad_indices,
        labels=labels,
        variances=variances,
        reasons=reasons,
    )


def plot_channel_variance_heatmap(
    result: ChannelQualityResult,
    grid_shape: tuple[int, int] = (16, 12),
    figsize: tuple = (10, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a 2-D grid heatmap of per-channel variance (e.g., 16×12 for 192 channels).

    Args:
        result: Output from detect_bad_channels().
        grid_shape: (rows, cols) for arranging channels into a 2-D grid.
        figsize: Figure size.
        ax: Optional pre-existing Axes.

    Returns:
        The matplotlib Figure.
    """
    rows, cols = grid_shape
    n_channels = result.n_total

    # Pad variances to fill grid
    padded = np.full(rows * cols, np.nan)
    padded[:n_channels] = result.variances
    grid = padded.reshape(rows, cols)

    show_new = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(grid, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title("Channel Variance Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="Variance")

    # Mark bad channels with an X
    for idx in result.bad_indices:
        if idx < rows * cols:
            r, c = divmod(idx, cols)
            ax.plot(c, r, "cx", markersize=8, markeredgewidth=2)

    if show_new:
        fig.tight_layout()
    return fig
