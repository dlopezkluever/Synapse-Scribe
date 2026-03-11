"""Neural state trajectory visualization.

Projects hidden states at each timestep to 2D (via PCA) and plots
time-colored paths showing neural state evolution during character production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn


def extract_temporal_embeddings(
    model: nn.Module,
    features: torch.Tensor | np.ndarray,
    layer_name: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Extract per-timestep hidden states for a single trial.

    Args:
        model: Trained decoder model.
        features: [T, C] or [1, T, C] features for one trial.
        layer_name: Module to hook (auto-detect if None).
        device: Torch device.

    Returns:
        Array [T, D] of hidden state vectors at each timestep.
    """
    model.eval()
    model.to(device)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if features.ndim == 2:
        features = features.unsqueeze(0)
    features = features.to(device)

    from src.analysis.embeddings import _find_layer

    target_layer = _find_layer(model, layer_name)

    captured = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        captured.append(output.detach().cpu())

    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(features)

    handle.remove()

    if not captured:
        return np.empty((0, 0))

    hidden = captured[0][0]  # [T, D] (first sample in batch)
    return hidden.numpy()


def plot_neural_trajectory(
    hidden_states: np.ndarray,
    label: str = "",
    title: str = "Neural State Trajectory",
    figsize: tuple = (10, 8),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot 2D neural trajectory with time-colored path.

    Args:
        hidden_states: Array [T, D] of per-timestep hidden states.
        label: Ground truth label string (shown in title).
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    T, D = hidden_states.shape

    # PCA to 2D
    coords = _pca_2d(hidden_states)

    fig, ax = plt.subplots(figsize=figsize)

    # Color by time
    colors = cm.viridis(np.linspace(0, 1, T))

    # Plot trajectory as connected line segments
    for i in range(T - 1):
        ax.plot(
            coords[i:i + 2, 0], coords[i:i + 2, 1],
            color=colors[i], linewidth=1.0, alpha=0.7,
        )

    # Mark start and end
    ax.scatter(coords[0, 0], coords[0, 1], c="green", s=100, zorder=5,
               marker="o", edgecolors="black", label="Start")
    ax.scatter(coords[-1, 0], coords[-1, 1], c="red", s=100, zorder=5,
               marker="s", edgecolors="black", label="End")

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, T))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Timestep")

    full_title = title
    if label:
        full_title += f"  (label: '{label}')"
    ax.set_title(full_title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multi_trial_trajectories(
    trajectories: list[np.ndarray],
    labels: list[str],
    title: str = "Multi-Trial Neural Trajectories",
    figsize: tuple = (12, 10),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot multiple trial trajectories in shared PCA space.

    Args:
        trajectories: List of [T_i, D] arrays.
        labels: Label per trial.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    # Concatenate all for shared PCA
    all_states = np.concatenate(trajectories, axis=0)
    coords_all = _pca_2d(all_states)

    fig, ax = plt.subplots(figsize=figsize)

    # Assign colors per unique label
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
    label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    offset = 0
    for traj, label in zip(trajectories, labels):
        T = traj.shape[0]
        coords = coords_all[offset:offset + T]
        offset += T

        color = label_to_color[label]
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=0.8, alpha=0.6)
        ax.scatter(coords[0, 0], coords[0, 1], c=[color], s=40, marker="o",
                   edgecolors="black", linewidth=0.5, zorder=5)

    # Legend
    for lbl in unique_labels:
        display = "space" if lbl == " " else lbl
        ax.plot([], [], color=label_to_color[lbl], label=display, linewidth=2)
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _pca_2d(data: np.ndarray) -> np.ndarray:
    """Project data to 2D using PCA."""
    centered = data - data.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ Vt[:2].T
    except np.linalg.LinAlgError:
        if centered.shape[1] >= 2:
            return centered[:, :2]
        return np.column_stack([centered[:, 0], np.zeros(len(centered))])
