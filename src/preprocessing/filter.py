"""Signal filtering: bandpass, notch, downsampling, and temporal smoothing.

All functions operate on arrays of shape [T, C] (timesteps x channels).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, decimate


def bandpass_filter(
    signals: np.ndarray,
    fs: float,
    low: float = 1.0,
    high: float = 200.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter.

    Args:
        signals: Array [T, C].
        fs: Sampling rate in Hz.
        low: Low cutoff frequency.
        high: High cutoff frequency.
        order: Filter order.

    Returns:
        Filtered array with same shape as input.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signals, axis=0).astype(signals.dtype)


def notch_filter(
    signals: np.ndarray,
    fs: float,
    freqs: list[float] | None = None,
    quality: float = 30.0,
) -> np.ndarray:
    """Remove line noise at specified frequencies (default 60 Hz + 120 Hz harmonic).

    Args:
        signals: Array [T, C].
        fs: Sampling rate in Hz.
        freqs: Frequencies to notch out.
        quality: Quality factor for the notch filter.

    Returns:
        Filtered array with same shape as input.
    """
    if freqs is None:
        freqs = [60.0, 120.0]

    out = signals.copy()
    nyq = fs / 2.0
    for f in freqs:
        if f >= nyq:
            continue
        b, a = iirnotch(f, quality, fs)
        out = filtfilt(b, a, out, axis=0)
    return out.astype(signals.dtype)


def artifact_rejection(
    signals: np.ndarray,
    threshold: float = 3.0,
) -> tuple[np.ndarray, bool]:
    """Flag a trial for gross artifact rejection.

    A trial is rejected if any channel's variance exceeds
    `threshold` times the median channel variance.

    Args:
        signals: Array [T, C] — single trial.
        threshold: Variance multiplier for rejection.

    Returns:
        (signals, is_rejected) tuple.
    """
    variances = np.var(signals, axis=0)
    median_var = np.median(variances)
    if median_var > 0 and np.any(variances > threshold * median_var):
        return signals, True
    return signals, False


def temporal_downsample(
    signals: np.ndarray,
    current_fs: float,
    target_fs: float = 250.0,
) -> np.ndarray:
    """Downsample signals along the time axis using scipy.signal.decimate.

    Args:
        signals: Array [T, C].
        current_fs: Current sampling rate.
        target_fs: Target sampling rate.

    Returns:
        Downsampled array [T', C] where T' = T * target_fs / current_fs.
    """
    ratio = current_fs / target_fs
    if ratio <= 1.0:
        return signals

    factor = int(round(ratio))
    if factor <= 1:
        return signals

    # decimate operates along axis=0
    return decimate(signals, factor, axis=0).astype(signals.dtype)


class GaussianTemporalSmoothing:
    """Apply Gaussian kernel smoothing across the time axis.

    Converts noisy spike counts into smooth firing rate estimates.
    """

    def __init__(self, sigma_ms: float = 30.0, fs: float = 250.0):
        """
        Args:
            sigma_ms: Standard deviation of the Gaussian kernel in milliseconds.
            fs: Sampling rate in Hz.
        """
        self.sigma_ms = sigma_ms
        self.fs = fs

    def __call__(self, signals: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing.

        Args:
            signals: Array [T, C].

        Returns:
            Smoothed array with same shape.
        """
        sigma_samples = self.sigma_ms / 1000.0 * self.fs
        if sigma_samples < 0.5:
            return signals

        # Build 1-D Gaussian kernel
        radius = int(3 * sigma_samples)
        t = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (t / sigma_samples) ** 2)
        kernel /= kernel.sum()

        # Convolve each channel independently
        from scipy.ndimage import convolve1d
        return convolve1d(signals, kernel, axis=0, mode="nearest").astype(signals.dtype)
