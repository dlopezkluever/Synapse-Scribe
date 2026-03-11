"""Trial similarity analysis via pairwise cosine similarity.

Computes and visualizes similarity matrices between trial embeddings,
grouped by character class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Array [N, D] of embedding vectors.

    Returns:
        Similarity matrix [N, N] with values in [-1, 1].
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T


def compute_class_similarity(
    embeddings: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute mean pairwise similarity between character classes.

    Args:
        embeddings: Array [N, D].
        labels: Character label per sample.

    Returns:
        (similarity_matrix, class_labels) where similarity_matrix is
        [n_classes, n_classes] of mean within/between-class similarities.
    """
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    # Compute full similarity
    sim = compute_cosine_similarity(embeddings)

    # Average per class pair
    class_sim = np.zeros((n_classes, n_classes))
    class_counts = np.zeros((n_classes, n_classes))

    for i in range(len(labels)):
        for j in range(len(labels)):
            ci = label_to_idx[labels[i]]
            cj = label_to_idx[labels[j]]
            class_sim[ci, cj] += sim[i, j]
            class_counts[ci, cj] += 1

    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_sim /= class_counts

    display_labels = ["space" if lbl == " " else lbl for lbl in unique_labels]
    return class_sim, display_labels


def plot_similarity_matrix(
    similarity: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Trial Similarity Matrix",
    figsize: tuple = (10, 8),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a similarity matrix heatmap.

    Args:
        similarity: Array [N, N] or [n_classes, n_classes].
        labels: Axis labels.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)

    if labels is not None and len(labels) <= 40:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_class_similarity(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Character-Grouped Similarity",
    figsize: tuple = (12, 10),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Compute and plot character-class similarity heatmap.

    Args:
        embeddings: Array [N, D].
        labels: Character label per sample.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    class_sim, class_labels = compute_class_similarity(embeddings, labels)
    return plot_similarity_matrix(
        class_sim, labels=class_labels, title=title,
        figsize=figsize, save_path=save_path,
    )
