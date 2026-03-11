"""Signal normalization and bad channel rejection.

Provides per-channel z-score normalization using training-set statistics,
and channel rejection using diagnostics output or inline detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Per-channel mean and std from training data."""

    mean: np.ndarray  # [C]
    std: np.ndarray   # [C]

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        data = np.load(Path(path))
        return cls(mean=data["mean"], std=data["std"])


def compute_normalization_stats(
    signals_list: list[np.ndarray],
) -> NormalizationStats:
    """Compute per-channel mean and std from a list of trial signals.

    Args:
        signals_list: List of arrays each shaped [T_i, C].

    Returns:
        NormalizationStats with mean and std of shape [C].
    """
    all_data = np.concatenate(signals_list, axis=0)  # [sum(T_i), C]
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    return NormalizationStats(mean=mean, std=std)


def zscore_normalize(
    signals: np.ndarray,
    stats: NormalizationStats,
    clip: float = 5.0,
) -> np.ndarray:
    """Per-channel z-score normalization, clipped to [-clip, clip].

    Args:
        signals: Array [T, C].
        stats: Training-set normalization statistics.
        clip: Clip value for normalized output.

    Returns:
        Normalized and clipped array [T, C].
    """
    normalized = (signals - stats.mean) / stats.std
    return np.clip(normalized, -clip, clip).astype(np.float32)


def detect_bad_channels_inline(
    signals: np.ndarray,
    var_threshold: float = 10.0,
) -> list[int]:
    """Inline fallback for bad channel detection (when diagnostics haven't been run).

    Flags channels with zero variance or variance > var_threshold × session median.

    Args:
        signals: Array [T, C].
        var_threshold: Multiplier above median variance to flag.

    Returns:
        List of bad channel indices.
    """
    variances = np.var(signals, axis=0)
    median_var = np.median(variances)
    bad = []
    for ch in range(signals.shape[1]):
        if variances[ch] == 0:
            bad.append(ch)
        elif median_var > 0 and variances[ch] > var_threshold * median_var:
            bad.append(ch)
    return bad


def remove_bad_channels(
    signals: np.ndarray,
    bad_indices: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Remove bad channels from signals.

    Args:
        signals: Array [T, C].
        bad_indices: Indices of channels to remove.

    Returns:
        (cleaned_signals [T, C'], kept_indices) where C' = C - len(bad_indices).
    """
    if not bad_indices:
        return signals, list(range(signals.shape[1]))

    all_channels = set(range(signals.shape[1]))
    kept = sorted(all_channels - set(bad_indices))
    logger.info(
        "Removing %d bad channels (%d → %d kept)",
        len(bad_indices), signals.shape[1], len(kept),
    )
    return signals[:, kept], kept


def get_bad_channels(
    signals: np.ndarray,
    diagnostics_bad_indices: Optional[list[int]] = None,
    var_threshold: float = 10.0,
) -> list[int]:
    """Get bad channel indices, preferring diagnostics output over inline detection.

    Args:
        signals: Array [T, C].
        diagnostics_bad_indices: Pre-computed bad indices from Phase 0.6 diagnostics.
        var_threshold: Threshold for inline fallback.

    Returns:
        List of bad channel indices.
    """
    if diagnostics_bad_indices is not None:
        return diagnostics_bad_indices
    return detect_bad_channels_inline(signals, var_threshold)
