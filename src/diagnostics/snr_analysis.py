"""Signal-to-Noise Ratio (SNR) analysis for neural recordings.

Estimates SNR as power(signal_band) / power(noise_band) per channel.
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
class SNRResult:
    """Result of SNR analysis."""

    snr_per_channel: np.ndarray  # [C] SNR values in dB
    low_quality_indices: list[int]
    signal_power: np.ndarray  # [C] power in signal band
    noise_power: np.ndarray  # [C] power in noise band
    threshold_db: float


def compute_snr(
    signals: np.ndarray,
    fs: float = 250.0,
    signal_band: tuple[float, float] = (70.0, 150.0),
    noise_band: tuple[float, float] = (55.0, 65.0),
    threshold_db: float = 3.0,
    nperseg: Optional[int] = None,
) -> SNRResult:
    """Compute per-channel SNR from power in signal vs. noise frequency bands.

    Args:
        signals: Array of shape [T, C].
        fs: Sampling rate in Hz.
        signal_band: (low, high) Hz for the signal band (default: high gamma 70-150 Hz).
        noise_band: (low, high) Hz for the noise band (default: 55-65 Hz line noise).
        threshold_db: Channels with SNR below this (dB) are flagged as low quality.
        nperseg: Segment length for Welch PSD. None → min(256, T).

    Returns:
        SNRResult with per-channel SNR and flagged low-quality channels.
    """
    if signals.ndim != 2:
        raise ValueError(f"Expected 2-D array [T, C], got shape {signals.shape}")

    T, C = signals.shape
    if nperseg is None:
        nperseg = min(256, T)

    freqs, psd = welch(signals, fs=fs, nperseg=nperseg, axis=0)  # psd: [n_freqs, C]

    # Find frequency indices for each band
    sig_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])

    # Compute mean power in each band per channel
    signal_power = np.mean(psd[sig_mask, :], axis=0) if sig_mask.any() else np.zeros(C)
    noise_power = np.mean(psd[noise_mask, :], axis=0) if noise_mask.any() else np.zeros(C)

    # SNR in dB: 10 * log10(signal / noise)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db = 10.0 * np.log10(signal_power / np.where(noise_power > 0, noise_power, 1e-30))
    snr_db = np.nan_to_num(snr_db, nan=0.0, posinf=100.0, neginf=-100.0)

    low_quality = [i for i in range(C) if snr_db[i] < threshold_db]

    logger.info(
        "SNR analysis: %d/%d channels above %.1f dB threshold. Mean SNR=%.1f dB",
        C - len(low_quality), C, threshold_db, np.mean(snr_db),
    )

    return SNRResult(
        snr_per_channel=snr_db,
        low_quality_indices=low_quality,
        signal_power=signal_power,
        noise_power=noise_power,
        threshold_db=threshold_db,
    )


def plot_snr_distribution(
    result: SNRResult,
    figsize: tuple = (10, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot histogram of per-channel SNR values.

    Args:
        result: Output from compute_snr().
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

    ax.hist(result.snr_per_channel, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(result.threshold_db, color="red", linestyle="--", label=f"Threshold ({result.threshold_db} dB)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Number of Channels")
    ax.set_title("SNR Distribution Across Channels")
    ax.legend()

    if show_new:
        fig.tight_layout()
    return fig
