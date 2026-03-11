"""Trial segmentation: extract per-trial windows and pad/truncate to fixed length."""

from __future__ import annotations

import numpy as np


def segment_trials(
    continuous_signal: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    fs: float = 250.0,
    pre_pad_ms: float = 100.0,
    post_pad_ms: float = 200.0,
) -> list[np.ndarray]:
    """Extract per-trial windows from a continuous recording.

    Args:
        continuous_signal: Array [T_total, C].
        onsets: Trial onset sample indices.
        offsets: Trial offset sample indices.
        fs: Sampling rate in Hz.
        pre_pad_ms: Padding before onset in ms.
        post_pad_ms: Padding after offset in ms.

    Returns:
        List of trial arrays, each [T_trial, C].
    """
    pre_samples = int(pre_pad_ms / 1000.0 * fs)
    post_samples = int(post_pad_ms / 1000.0 * fs)
    T_total = continuous_signal.shape[0]

    trials = []
    for onset, offset in zip(onsets, offsets):
        start = max(0, int(onset) - pre_samples)
        end = min(T_total, int(offset) + post_samples)
        trials.append(continuous_signal[start:end])

    return trials


def pad_or_truncate(
    signal: np.ndarray,
    t_max: int = 2000,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad (right) or truncate a signal to exactly t_max timesteps.

    Args:
        signal: Array [T, C].
        t_max: Target length in timesteps.
        pad_value: Value used for padding.

    Returns:
        Array [t_max, C].
    """
    T, C = signal.shape
    if T >= t_max:
        return signal[:t_max]

    padded = np.full((t_max, C), pad_value, dtype=signal.dtype)
    padded[:T] = signal
    return padded


def pad_or_truncate_batch(
    signals: list[np.ndarray],
    t_max: int = 2000,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate a list of variable-length signals to fixed length.

    Args:
        signals: List of arrays each [T_i, C].
        t_max: Target length.
        pad_value: Value used for padding.

    Returns:
        (padded_batch [N, t_max, C], actual_lengths [N]).
    """
    lengths = np.array([min(s.shape[0], t_max) for s in signals])
    C = signals[0].shape[1]
    batch = np.full((len(signals), t_max, C), pad_value, dtype=np.float32)
    for i, s in enumerate(signals):
        L = lengths[i]
        batch[i, :L] = s[:L]
    return batch, lengths
