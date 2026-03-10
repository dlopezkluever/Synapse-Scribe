"""Raw neural signal visualization utilities.

Functions for plotting multi-channel time series and channel heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_multichannel_timeseries(
    signals: np.ndarray,
    fs: float = 250.0,
    channels: Optional[list[int]] = None,
    title: str = "Multi-Channel Neural Time Series",
    offset_scale: float = 3.0,
    figsize: tuple = (14, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot raw multi-channel time series for a single trial.

    Args:
        signals: Array of shape [T, C] (timesteps × channels).
        fs: Sampling rate in Hz, used to build the time axis.
        channels: Subset of channel indices to plot. None → first 20.
        title: Plot title.
        offset_scale: Vertical offset multiplier between channels.
        figsize: Figure size (width, height) in inches.
        ax: Optional pre-existing Axes to draw on.

    Returns:
        The matplotlib Figure.
    """
    if signals.ndim != 2:
        raise ValueError(f"Expected 2-D array [T, C], got shape {signals.shape}")

    T, C = signals.shape
    time = np.arange(T) / fs

    if channels is None:
        channels = list(range(min(C, 20)))

    show_new = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    std = np.std(signals[:, channels])
    std = std if std > 0 else 1.0

    for idx, ch in enumerate(channels):
        offset = idx * offset_scale * std
        ax.plot(time, signals[:, ch] + offset, linewidth=0.5, label=f"Ch {ch}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (offset)")
    ax.set_title(title)
    ax.set_yticks([i * offset_scale * std for i in range(len(channels))])
    ax.set_yticklabels([f"Ch {ch}" for ch in channels])

    if show_new:
        fig.tight_layout()
    return fig


def plot_channel_heatmap(
    signals: np.ndarray,
    fs: float = 250.0,
    title: str = "Channel Amplitude Heatmap",
    figsize: tuple = (14, 6),
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a [time × channels] heatmap color-coded by amplitude.

    Args:
        signals: Array of shape [T, C].
        fs: Sampling rate in Hz.
        title: Plot title.
        figsize: Figure size.
        cmap: Matplotlib colormap name.
        ax: Optional pre-existing Axes.

    Returns:
        The matplotlib Figure.
    """
    if signals.ndim != 2:
        raise ValueError(f"Expected 2-D array [T, C], got shape {signals.shape}")

    T, C = signals.shape
    time_extent = T / fs  # seconds

    show_new = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        signals.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_extent, 0, C],
        cmap=cmap,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude")

    if show_new:
        fig.tight_layout()
    return fig


def plot_trial_overview(
    signals: np.ndarray,
    label: str = "",
    fs: float = 250.0,
    channels: Optional[list[int]] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Combined view: time series on top, heatmap on bottom, with label."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    plot_multichannel_timeseries(signals, fs=fs, channels=channels, ax=ax1,
                                  title=f"Time Series — label: '{label}'")
    plot_channel_heatmap(signals, fs=fs, ax=ax2, title="Channel Heatmap")

    fig.tight_layout()
    return fig
