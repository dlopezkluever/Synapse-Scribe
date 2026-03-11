"""Power spectrum analysis for neural recordings.

Computes PSD via Welch's method and detects spectral contamination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class SpectralResult:
    """Result of power spectrum analysis."""

    freqs: np.ndarray  # [n_freqs]
    psd: np.ndarray  # [n_freqs, C] power spectral density
    psd_mean: np.ndarray  # [n_freqs] session-average PSD
    line_noise_detected: bool
    line_noise_power: float  # power at 60 Hz (or configured freq)
    high_gamma_present: bool
    high_gamma_power: float


def compute_psd(
    signals: np.ndarray,
    fs: float = 250.0,
    nperseg: Optional[int] = None,
    line_noise_freq: float = 60.0,
    line_noise_threshold: float = 5.0,
    high_gamma_band: tuple[float, float] = (70.0, 150.0),
) -> SpectralResult:
    """Compute power spectral density per channel using Welch's method.

    Args:
        signals: Array of shape [T, C].
        fs: Sampling rate in Hz.
        nperseg: Segment length for Welch. None → min(256, T).
        line_noise_freq: Frequency to check for line noise.
        line_noise_threshold: Ratio of line noise power to median power to flag contamination.
        high_gamma_band: (low, high) Hz for high gamma detection.

    Returns:
        SpectralResult with PSD data and contamination flags.
    """
    if signals.ndim != 2:
        raise ValueError(f"Expected 2-D array [T, C], got shape {signals.shape}")

    T, C = signals.shape
    if nperseg is None:
        nperseg = min(256, T)

    freqs, psd = welch(signals, fs=fs, nperseg=nperseg, axis=0)  # [n_freqs, C]
    psd_mean = np.mean(psd, axis=1)  # [n_freqs]

    # Line noise detection
    line_idx = np.argmin(np.abs(freqs - line_noise_freq))
    line_power = psd_mean[line_idx]
    median_power = np.median(psd_mean)
    line_noise_detected = bool(median_power > 0 and line_power > line_noise_threshold * median_power)

    # High gamma detection
    hg_mask = (freqs >= high_gamma_band[0]) & (freqs <= high_gamma_band[1])
    hg_power = float(np.mean(psd_mean[hg_mask])) if hg_mask.any() else 0.0
    # Consider high gamma "present" if its power exceeds baseline
    baseline_mask = (freqs >= 1.0) & (freqs <= 30.0)
    baseline_power = float(np.mean(psd_mean[baseline_mask])) if baseline_mask.any() else 0.0
    high_gamma_present = bool(hg_power > 0.1 * baseline_power) if baseline_power > 0 else False

    logger.info(
        "Spectral analysis: line_noise=%s (%.2e at %.0f Hz), high_gamma=%s (%.2e)",
        line_noise_detected, line_power, line_noise_freq,
        high_gamma_present, hg_power,
    )

    return SpectralResult(
        freqs=freqs,
        psd=psd,
        psd_mean=psd_mean,
        line_noise_detected=line_noise_detected,
        line_noise_power=float(line_power),
        high_gamma_present=high_gamma_present,
        high_gamma_power=hg_power,
    )


def plot_power_spectrum(
    result: SpectralResult,
    channels: Optional[list[int]] = None,
    figsize: tuple = (12, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot PSD for individual channels and session-average.

    Args:
        result: Output from compute_psd().
        channels: Specific channel indices to plot. None → show only session average.
        figsize: Figure size.
        ax: Optional pre-existing Axes.

    Returns:
        The matplotlib Figure.
    """
    show_new = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if channels is not None:
        for ch in channels:
            if ch < result.psd.shape[1]:
                ax.semilogy(result.freqs, result.psd[:, ch], alpha=0.4, linewidth=0.5)

    ax.semilogy(result.freqs, result.psd_mean, color="black", linewidth=2, label="Session Average")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Power Spectrum Analysis")
    ax.legend()

    if show_new:
        fig.tight_layout()
    return fig
