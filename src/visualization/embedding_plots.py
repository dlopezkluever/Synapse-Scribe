"""Embedding visualization: t-SNE and UMAP plots of learned representations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    method: str = "tsne",
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    title: str = "Learned Representations",
    figsize: tuple = (12, 10),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """2D scatter plot of embeddings colored by character class.

    Args:
        embeddings: Array [N, D] of embedding vectors.
        labels: Character labels per sample (length N).
        method: 'tsne' or 'pca'. Falls back to PCA if sklearn unavailable.
        perplexity: t-SNE perplexity.
        n_neighbors: UMAP n_neighbors (unused for t-SNE/PCA).
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        The matplotlib Figure.
    """
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    # Dimensionality reduction
    coords_2d = _reduce_dims(embeddings, method, perplexity)

    # Assign colors per unique label
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))
    label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=figsize)

    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        pts = coords_2d[mask]
        display = "space" if lbl == " " else lbl
        ax.scatter(pts[:, 0], pts[:, 1], c=[label_to_color[lbl]],
                   label=display, s=20, alpha=0.7)

    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Place legend outside
    if len(unique_labels) <= 30:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
                  markerscale=1.5, ncol=1)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _reduce_dims(
    embeddings: np.ndarray,
    method: str = "tsne",
    perplexity: float = 30.0,
) -> np.ndarray:
    """Reduce embeddings to 2D."""
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1),
                           random_state=42)
            return reducer.fit_transform(embeddings)
        except ImportError:
            pass

    # Fallback: PCA
    centered = embeddings - embeddings.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n_components = min(2, Vt.shape[0])
        projected = centered @ Vt[:n_components].T
        if projected.shape[1] < 2:
            projected = np.column_stack([projected, np.zeros(len(projected))])
        return projected
    except np.linalg.LinAlgError:
        if centered.shape[1] >= 2:
            return centered[:, :2]
        return np.column_stack([centered[:, 0], np.zeros(len(centered))])
