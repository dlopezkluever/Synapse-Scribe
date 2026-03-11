"""CTC-specific visualization: probability heatmaps, per-character error bars,
and training curve comparison plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.data.dataset import IDX_TO_CHAR, VOCAB_SIZE


def plot_ctc_heatmap(
    logits: np.ndarray,
    reference: str = "",
    title: str = "CTC Probability Heatmap",
    figsize: tuple = (16, 6),
    max_timesteps: int = 500,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot CTC output probabilities as [time x characters] heatmap.

    Args:
        logits: Array of shape [T, n_classes] (raw logits, softmax applied here).
        reference: Optional ground truth string shown in title.
        title: Plot title.
        figsize: Figure size.
        max_timesteps: Truncate to this many timesteps for readability.
        ax: Optional pre-existing Axes.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    if logits.ndim == 3:
        logits = logits[0]

    T, C = logits.shape

    # Apply softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

    # Truncate for readability
    if T > max_timesteps:
        probs = probs[:max_timesteps]
        T = max_timesteps

    show_new = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        probs.T,
        aspect="auto",
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=1,
    )

    # Y-axis labels: character names
    char_labels = []
    for i in range(C):
        ch = IDX_TO_CHAR.get(i, "?")
        if i == 0:
            char_labels.append("blank")
        elif ch == " ":
            char_labels.append("space")
        else:
            char_labels.append(ch)

    ax.set_yticks(range(C))
    ax.set_yticklabels(char_labels, fontsize=7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Character")

    full_title = title
    if reference:
        full_title += f"  (ref: '{reference}')"
    ax.set_title(full_title)

    fig.colorbar(im, ax=ax, label="Probability", shrink=0.8)

    if show_new:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_character_errors(
    char_error_rates: dict[str, float],
    title: str = "Per-Character Error Rate",
    figsize: tuple = (14, 5),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart of per-character error rates.

    Args:
        char_error_rates: Dict mapping character -> error rate.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    chars = sorted(char_error_rates.keys())
    rates = [char_error_rates[c] for c in chars]
    display_labels = ["space" if c == " " else c for c in chars]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn_r(np.array(rates))
    ax.bar(range(len(chars)), rates, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel("Error Rate")
    ax.set_title(title)
    ax.set_ylim(0, min(max(rates) * 1.2, 1.0) if rates else 1.0)
    ax.axhline(y=np.mean(rates) if rates else 0, color="blue", linestyle="--",
               linewidth=1, label=f"Mean: {np.mean(rates):.3f}" if rates else "")
    ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    histories: dict[str, dict],
    title: str = "Training Curves Comparison",
    figsize: tuple = (14, 10),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot training/val loss and CER for multiple models on one figure.

    Args:
        histories: Dict of model_name -> dict with keys
            'train_losses', 'val_losses', 'val_cers', 'learning_rates'.
        title: Overall title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_train_loss, ax_val_loss = axes[0]
    ax_val_cer, ax_lr = axes[1]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 1)))

    for idx, (name, hist) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        epochs = range(1, len(hist.get("train_losses", [])) + 1)

        if "train_losses" in hist:
            ax_train_loss.plot(epochs, hist["train_losses"], label=name, color=color)
        if "val_losses" in hist:
            ax_val_loss.plot(epochs, hist["val_losses"], label=name, color=color)
        if "val_cers" in hist:
            ax_val_cer.plot(epochs, hist["val_cers"], label=name, color=color)
        if "learning_rates" in hist:
            ax_lr.plot(epochs, hist["learning_rates"], label=name, color=color)

    ax_train_loss.set_title("Training Loss")
    ax_train_loss.set_xlabel("Epoch")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.legend(fontsize=8)
    ax_train_loss.grid(True, alpha=0.3)

    ax_val_loss.set_title("Validation Loss")
    ax_val_loss.set_xlabel("Epoch")
    ax_val_loss.set_ylabel("Loss")
    ax_val_loss.legend(fontsize=8)
    ax_val_loss.grid(True, alpha=0.3)

    ax_val_cer.set_title("Validation CER")
    ax_val_cer.set_xlabel("Epoch")
    ax_val_cer.set_ylabel("CER")
    ax_val_cer.legend(fontsize=8)
    ax_val_cer.grid(True, alpha=0.3)

    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.legend(fontsize=8)
    ax_lr.set_yscale("log")
    ax_lr.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    confusion: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Character Confusion Matrix",
    figsize: tuple = (12, 10),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        confusion: Array of shape [n_chars, n_chars].
        labels: Character labels for axes.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    if labels is None:
        labels = [chr(ord("a") + i) for i in range(26)] + ["space"]

    n = confusion.shape[0]
    labels = labels[:n]

    # Normalize rows for display
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Normalized Frequency", shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
