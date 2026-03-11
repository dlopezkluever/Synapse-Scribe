"""Channel correlation analysis for neural recordings.

Computes pairwise correlation matrix and flags highly correlated channel pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of channel correlation analysis."""

    correlation_matrix: np.ndarray  # [C, C]
    high_corr_pairs: list[tuple[int, int, float]]  # (ch_i, ch_j, corr_value)
    n_high_corr_pairs: int
    threshold: float


def compute_channel_correlation(
    signals: np.ndarray,
    threshold: float = 0.95,
) -> CorrelationResult:
    """Compute pairwise Pearson correlation across all channels.

    Args:
        signals: Array of shape [T, C].
        threshold: Correlation threshold above which pairs are flagged.

    Returns:
        CorrelationResult with correlation matrix and flagged pairs.
    """
    if signals.ndim != 2:
        raise ValueError(f"Expected 2-D array [T, C], got shape {signals.shape}")

    T, C = signals.shape

    # Handle channels with zero variance (would produce NaN correlations)
    stds = np.std(signals, axis=0)
    zero_var_mask = stds == 0

    # Compute correlation matrix
    if C <= 1:
        corr = np.ones((C, C))
    else:
        corr = np.corrcoef(signals.T)  # [C, C]

    # Replace NaN from zero-variance channels with 0
    corr = np.nan_to_num(corr, nan=0.0)

    # Set diagonal back to 1.0
    np.fill_diagonal(corr, 1.0)

    # Zero-variance channels: set their rows/cols to 0 (except self)
    for ch in np.where(zero_var_mask)[0]:
        corr[ch, :] = 0.0
        corr[:, ch] = 0.0
        corr[ch, ch] = 1.0

    # Find highly correlated pairs (upper triangle only)
    high_pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            if abs(corr[i, j]) > threshold:
                high_pairs.append((i, j, float(corr[i, j])))

    logger.info(
        "Correlation analysis: %d channel pairs above threshold %.2f",
        len(high_pairs), threshold,
    )

    return CorrelationResult(
        correlation_matrix=corr,
        high_corr_pairs=high_pairs,
        n_high_corr_pairs=len(high_pairs),
        threshold=threshold,
    )


def plot_correlation_matrix(
    result: CorrelationResult,
    figsize: tuple = (10, 9),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the channel-channel correlation matrix as a heatmap.

    Args:
        result: Output from compute_channel_correlation().
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

    im = ax.imshow(
        result.correlation_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title(f"Channel Correlation Matrix ({result.n_high_corr_pairs} pairs > {result.threshold})")
    fig.colorbar(im, ax=ax, label="Pearson Correlation")

    if show_new:
        fig.tight_layout()
    return fig
