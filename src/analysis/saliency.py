"""Gradient-based electrode importance maps.

Implements input×gradient and integrated gradients methods to determine
which electrodes contribute most to decoding each character.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def input_x_gradient(
    model: nn.Module,
    features: torch.Tensor | np.ndarray,
    target_class: int | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Compute input × gradient attribution for a single trial.

    Args:
        model: Trained decoder model.
        features: [T, C] or [1, T, C] input features.
        target_class: If given, compute gradient w.r.t. this class logit sum.
            If None, uses the argmax class at each timestep.
        device: Torch device.

    Returns:
        Attribution map [T, C] — magnitude indicates electrode importance.
    """
    model.eval()
    model.to(device)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if features.ndim == 2:
        features = features.unsqueeze(0)

    features = features.to(device).requires_grad_(True)

    logits = model(features)  # [1, T', n_classes]

    if target_class is not None:
        score = logits[0, :, target_class].sum()
    else:
        # Sum of max-class logits across time
        score = logits[0].max(dim=-1).values.sum()

    score.backward()

    grad = features.grad[0].detach().cpu().numpy()  # [T, C]
    inp = features[0].detach().cpu().numpy()          # [T, C]

    attribution = inp * grad  # [T, C]
    return attribution


def integrated_gradients(
    model: nn.Module,
    features: torch.Tensor | np.ndarray,
    target_class: int | None = None,
    n_steps: int = 50,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Compute integrated gradients attribution for a single trial.

    Args:
        model: Trained decoder model.
        features: [T, C] or [1, T, C] input features.
        target_class: Target class index (uses argmax if None).
        n_steps: Number of interpolation steps.
        device: Torch device.

    Returns:
        Attribution map [T, C].
    """
    model.eval()
    model.to(device)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if features.ndim == 2:
        features = features.unsqueeze(0)
    features = features.to(device)

    baseline = torch.zeros_like(features)
    delta = features - baseline

    grads_sum = torch.zeros_like(features)

    for step in range(n_steps):
        alpha = step / n_steps
        interp = baseline + alpha * delta
        interp = interp.detach().requires_grad_(True)

        logits = model(interp)

        if target_class is not None:
            score = logits[0, :, target_class].sum()
        else:
            score = logits[0].max(dim=-1).values.sum()

        score.backward()
        grads_sum += interp.grad.detach()

    avg_grad = grads_sum / n_steps
    attribution = (delta * avg_grad)[0].cpu().numpy()  # [T, C]

    return attribution


def electrode_importance(
    attribution: np.ndarray,
    aggregate: str = "mean_abs",
) -> np.ndarray:
    """Aggregate temporal attribution into per-electrode importance.

    Args:
        attribution: [T, C] attribution map.
        aggregate: 'mean_abs' (default) or 'sum_abs'.

    Returns:
        Array [C] of per-electrode importance scores.
    """
    if aggregate == "sum_abs":
        return np.abs(attribution).sum(axis=0)
    return np.abs(attribution).mean(axis=0)


def plot_electrode_importance(
    importance: np.ndarray,
    title: str = "Electrode Importance",
    n_channels: int | None = None,
    figsize: tuple = (14, 5),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart of per-electrode importance scores.

    Args:
        importance: Array [C] of importance values.
        title: Plot title.
        n_channels: Show only top-N channels. None = all.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    C = len(importance)

    if n_channels is not None and n_channels < C:
        top_idx = np.argsort(importance)[-n_channels:][::-1]
        importance = importance[top_idx]
        labels = [f"Ch {i}" for i in top_idx]
    else:
        top_idx = np.arange(C)
        labels = [f"{i}" for i in range(C)]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.hot(importance / (importance.max() + 1e-10))
    ax.bar(range(len(importance)), importance, color=colors, edgecolor="gray",
           linewidth=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("Importance Score")
    ax.set_title(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_electrode_heatmap(
    attribution: np.ndarray,
    title: str = "Electrode Attribution Heatmap",
    figsize: tuple = (14, 6),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Heatmap of [time × electrodes] attribution values.

    Args:
        attribution: [T, C] attribution map.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    abs_attr = np.abs(attribution)
    im = ax.imshow(abs_attr.T, aspect="auto", origin="lower", cmap="hot")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Electrode")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="|Attribution|", shrink=0.8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
