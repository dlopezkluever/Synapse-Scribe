"""Neural manifold analysis for population-level neural embeddings.

Provides PCA- and UMAP-based dimensionality reduction, per-character cluster
metrics (centroids, inter-class distances, silhouette scores), neural dynamics
velocity fields, and 2D/3D trajectory visualizations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def fit_pca(
    embeddings: np.ndarray,
    n_components: int = 3,
) -> tuple[np.ndarray, PCA]:
    """Fit PCA to embeddings and return projected data.

    Args:
        embeddings: [N, D] array of high-dimensional embeddings.
        n_components: Number of PCA components (2 or 3).

    Returns:
        (projected, pca_model) — [N, n_components] projected data and the fitted PCA.
    """
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(
        "PCA: %d components explain %.1f%% of variance", n_components, explained
    )
    return projected, pca


def fit_umap(
    embeddings: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, object]:
    """Fit UMAP to embeddings and return projected data.

    Falls back to PCA if ``umap-learn`` is not installed.

    Args:
        embeddings: [N, D] array.
        n_components: Target dimensionality.
        n_neighbors: UMAP locality parameter.
        min_dist: Minimum distance in embedding space.
        random_state: Reproducibility seed.

    Returns:
        (projected, reducer) — [N, n_components] projected data and the fitted
        UMAP (or PCA) model.
    """
    try:
        from umap import UMAP  # type: ignore[import-untyped]

        reducer = UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
            min_dist=min_dist,
            random_state=random_state,
        )
        projected = reducer.fit_transform(embeddings)
        logger.info("UMAP: projected %d samples to %dD", len(embeddings), n_components)
        return projected, reducer
    except ImportError:
        logger.warning("umap-learn not installed; falling back to PCA")
        return fit_pca(embeddings, n_components)


# ---------------------------------------------------------------------------
# Cluster metrics
# ---------------------------------------------------------------------------

def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: list[str],
) -> dict:
    """Compute per-character cluster statistics in embedding space.

    Args:
        embeddings: [N, D] array.
        labels: [N] list of character/class labels.

    Returns:
        Dict with keys:
            centroids: dict[label, np.ndarray[D]]
            inter_class_distances: np.ndarray [K, K] pairwise centroid distances
            class_labels: list[str] unique sorted labels
            silhouette: float (or None if < 2 classes or < 2 samples per class)
            intra_class_variance: dict[label, float] mean squared distance to centroid
    """
    unique_labels = sorted(set(labels))
    label_arr = np.array(labels)

    # Centroids
    centroids: dict[str, np.ndarray] = {}
    intra_var: dict[str, float] = {}
    for lbl in unique_labels:
        mask = label_arr == lbl
        cluster = embeddings[mask]
        centroid = cluster.mean(axis=0)
        centroids[lbl] = centroid
        intra_var[lbl] = float(np.mean(np.sum((cluster - centroid) ** 2, axis=1)))

    # Inter-class distance matrix
    k = len(unique_labels)
    dist_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            dist_matrix[i, j] = np.linalg.norm(
                centroids[unique_labels[i]] - centroids[unique_labels[j]]
            )

    # Silhouette score
    sil = None
    if k >= 2 and embeddings.shape[0] > k:
        le = LabelEncoder()
        encoded = le.fit_transform(labels)
        # Need at least 2 samples in some cluster
        _, counts = np.unique(encoded, return_counts=True)
        if np.all(counts >= 1) and len(np.unique(encoded)) >= 2:
            try:
                sil = float(silhouette_score(embeddings, encoded))
            except ValueError:
                sil = None

    return {
        "centroids": centroids,
        "inter_class_distances": dist_matrix,
        "class_labels": unique_labels,
        "silhouette": sil,
        "intra_class_variance": intra_var,
    }


# ---------------------------------------------------------------------------
# Neural dynamics / velocity
# ---------------------------------------------------------------------------

def compute_velocity_field(
    temporal_embeddings: np.ndarray,
) -> dict:
    """Compute velocity in latent space from per-timestep hidden states.

    Args:
        temporal_embeddings: [T, D] array of hidden states over time.

    Returns:
        Dict with keys:
            velocities: [T-1, D] instantaneous velocity vectors
            speeds: [T-1] magnitude of velocity at each timestep
            mean_speed: float
            acceleration: [T-2, D] second derivative
    """
    velocities = np.diff(temporal_embeddings, axis=0)  # [T-1, D]
    speeds = np.linalg.norm(velocities, axis=1)  # [T-1]
    acceleration = np.diff(velocities, axis=0)  # [T-2, D]

    return {
        "velocities": velocities,
        "speeds": speeds,
        "mean_speed": float(speeds.mean()),
        "acceleration": acceleration,
    }


def compute_multi_trial_dynamics(
    trial_embeddings: list[np.ndarray],
    labels: list[str],
) -> dict:
    """Compute velocity statistics across multiple trials.

    Args:
        trial_embeddings: List of [T_i, D] arrays, one per trial.
        labels: [N] trial labels.

    Returns:
        Dict with per-trial speeds, mean speeds grouped by label, etc.
    """
    per_trial = []
    for emb in trial_embeddings:
        if emb.shape[0] < 2:
            per_trial.append({"mean_speed": 0.0, "max_speed": 0.0})
            continue
        vf = compute_velocity_field(emb)
        per_trial.append({
            "mean_speed": vf["mean_speed"],
            "max_speed": float(vf["speeds"].max()),
        })

    # Group by label
    label_speeds: dict[str, list[float]] = {}
    for lbl, info in zip(labels, per_trial):
        label_speeds.setdefault(lbl, []).append(info["mean_speed"])

    label_mean_speeds = {
        lbl: float(np.mean(speeds)) for lbl, speeds in label_speeds.items()
    }

    return {
        "per_trial": per_trial,
        "label_mean_speeds": label_mean_speeds,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_manifold_2d(
    embeddings: np.ndarray,
    labels: list[str],
    method: str = "PCA",
    title: str = "",
    figsize: tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
):
    """Plot 2D manifold embedding colored by character class.

    Args:
        embeddings: [N, 2] already-projected embeddings (or [N, D] to auto-project).
        labels: [N] labels.
        method: Label for the method used (e.g. "PCA", "UMAP").
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if embeddings.shape[1] > 2:
        embeddings, _ = fit_pca(embeddings, n_components=2)

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = sorted(set(labels))
    label_arr = np.array(labels)
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = label_arr == lbl
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[cmap(i)],
            label=lbl,
            alpha=0.7,
            s=30,
            edgecolors="none",
        )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(title or f"Neural Manifold ({method})")
    if len(unique_labels) <= 30:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved manifold plot to %s", save_path)

    return fig


def plot_manifold_3d(
    embeddings: np.ndarray,
    labels: list[str],
    method: str = "PCA",
    title: str = "",
    figsize: tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
):
    """Plot 3D manifold embedding colored by character class.

    Args:
        embeddings: [N, 3+] — first 3 dims used for 3D projection.
        labels: [N] labels.
        method: Method label.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if embeddings.shape[1] > 3:
        embeddings, _ = fit_pca(embeddings, n_components=3)
    elif embeddings.shape[1] < 3:
        pad = np.zeros((embeddings.shape[0], 3 - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, pad])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = sorted(set(labels))
    label_arr = np.array(labels)
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = label_arr == lbl
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            embeddings[mask, 2],
            c=[cmap(i)],
            label=lbl,
            alpha=0.7,
            s=20,
        )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_zlabel(f"{method} 3")
    ax.set_title(title or f"Neural Manifold 3D ({method})")
    if len(unique_labels) <= 30:
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_neural_dynamics_3d(
    temporal_embeddings: np.ndarray,
    label: str = "",
    title: str = "",
    figsize: tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
):
    """Plot 3D neural trajectory with time-colored path.

    Args:
        temporal_embeddings: [T, D] per-timestep hidden states for one trial.
        label: Trial label string.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Project to 3D
    if temporal_embeddings.shape[1] > 3:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(temporal_embeddings)
    elif temporal_embeddings.shape[1] < 3:
        pad = np.zeros((temporal_embeddings.shape[0], 3 - temporal_embeddings.shape[1]))
        coords = np.hstack([temporal_embeddings, pad])
    else:
        coords = temporal_embeddings

    T = coords.shape[0]
    t_norm = np.linspace(0, 1, T)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create line segments
    points = coords.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = plt.cm.viridis
    colors = cmap(t_norm[:-1])

    lc = Line3DCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    # Start and end markers
    ax.scatter(*coords[0], color="green", s=100, zorder=5, label="Start")
    ax.scatter(*coords[-1], color="red", s=100, marker="s", zorder=5, label="End")

    # Set limits
    for dim in range(3):
        margin = (coords[:, dim].max() - coords[:, dim].min()) * 0.1
        if dim == 0:
            ax.set_xlim(coords[:, dim].min() - margin, coords[:, dim].max() + margin)
        elif dim == 1:
            ax.set_ylim(coords[:, dim].min() - margin, coords[:, dim].max() + margin)
        else:
            ax.set_zlim(coords[:, dim].min() - margin, coords[:, dim].max() + margin)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(title or f"Neural Trajectory: '{label}'")
    ax.legend()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Timestep", shrink=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_velocity_field(
    temporal_embeddings: np.ndarray,
    label: str = "",
    title: str = "",
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """Plot speed profile and acceleration over time.

    Args:
        temporal_embeddings: [T, D] per-timestep hidden states.
        label: Trial label.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    dynamics = compute_velocity_field(temporal_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Speed profile
    axes[0].plot(dynamics["speeds"], color="steelblue", linewidth=1.5)
    axes[0].axhline(
        dynamics["mean_speed"], color="red", linestyle="--", alpha=0.7,
        label=f"Mean speed: {dynamics['mean_speed']:.3f}",
    )
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Speed (L2 norm)")
    axes[0].set_title(f"Speed Profile: '{label}'")
    axes[0].legend()

    # Acceleration magnitude
    accel_mag = np.linalg.norm(dynamics["acceleration"], axis=1)
    axes[1].plot(accel_mag, color="coral", linewidth=1.5)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Acceleration (L2 norm)")
    axes[1].set_title(f"Acceleration Profile: '{label}'")

    fig.suptitle(title or "Neural Dynamics")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cluster_distances(
    metrics: dict,
    title: str = "",
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """Plot inter-class distance heatmap from cluster metrics.

    Args:
        metrics: Output of compute_cluster_metrics().
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    dist_matrix = metrics["inter_class_distances"]
    class_labels = metrics["class_labels"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")

    if len(class_labels) <= 30:
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(class_labels)

    ax.set_title(title or "Inter-Class Centroid Distances")
    fig.colorbar(im, ax=ax, label="Euclidean Distance")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
