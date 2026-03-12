"""Tests for neural manifold analysis (src/analysis/manifold.py).

Verifies dimensionality reduction, cluster metrics, velocity computation,
and visualization functions using synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.manifold import (
    compute_cluster_metrics,
    compute_multi_trial_dynamics,
    compute_velocity_field,
    fit_pca,
    fit_umap,
    plot_cluster_distances,
    plot_manifold_2d,
    plot_manifold_3d,
    plot_neural_dynamics_3d,
    plot_velocity_field,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embeddings_and_labels():
    """Create synthetic embeddings with 3 clear clusters."""
    rng = np.random.RandomState(42)
    n_per_class = 20
    dim = 16

    # Three well-separated clusters
    cluster_a = rng.randn(n_per_class, dim) + np.array([5.0] * dim)
    cluster_b = rng.randn(n_per_class, dim) + np.array([-5.0] * dim)
    cluster_c = rng.randn(n_per_class, dim)

    embeddings = np.vstack([cluster_a, cluster_b, cluster_c])
    labels = ["a"] * n_per_class + ["b"] * n_per_class + ["c"] * n_per_class
    return embeddings, labels


@pytest.fixture
def temporal_embeddings():
    """Create synthetic per-timestep hidden states (smooth trajectory)."""
    rng = np.random.RandomState(42)
    T, D = 100, 32
    t = np.linspace(0, 2 * np.pi, T).reshape(-1, 1)
    # Smooth spiral-like trajectory
    base = np.hstack([np.sin(t), np.cos(t), t / (2 * np.pi)])
    noise = rng.randn(T, D) * 0.05
    noise[:, :3] = 0  # Keep first 3 dims clean
    return np.hstack([base, rng.randn(T, D - 3) * 0.1]) + noise[:, :D - 0]


# ---------------------------------------------------------------------------
# PCA / UMAP
# ---------------------------------------------------------------------------

class TestFitPCA:
    def test_output_shape(self, embeddings_and_labels):
        embeddings, _ = embeddings_and_labels
        projected, pca = fit_pca(embeddings, n_components=3)
        assert projected.shape == (60, 3)

    def test_2d_projection(self, embeddings_and_labels):
        embeddings, _ = embeddings_and_labels
        projected, pca = fit_pca(embeddings, n_components=2)
        assert projected.shape == (60, 2)

    def test_explained_variance(self, embeddings_and_labels):
        embeddings, _ = embeddings_and_labels
        _, pca = fit_pca(embeddings, n_components=3)
        assert pca.explained_variance_ratio_.sum() > 0

    def test_clamps_components(self):
        """n_components should be clamped to min(N, D)."""
        small = np.random.randn(5, 3)
        projected, _ = fit_pca(small, n_components=10)
        assert projected.shape[1] <= 3


class TestFitUMAP:
    def test_falls_back_to_pca(self, embeddings_and_labels):
        """If UMAP is not installed, should fall back to PCA."""
        embeddings, _ = embeddings_and_labels
        projected, reducer = fit_umap(embeddings, n_components=2)
        assert projected.shape == (60, 2)

    def test_output_shape(self, embeddings_and_labels):
        embeddings, _ = embeddings_and_labels
        projected, _ = fit_umap(embeddings, n_components=3)
        assert projected.shape[0] == 60
        assert projected.shape[1] in (2, 3)  # 3 if UMAP available, 2/3 if PCA fallback


# ---------------------------------------------------------------------------
# Cluster metrics
# ---------------------------------------------------------------------------

class TestClusterMetrics:
    def test_centroids_computed(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        assert "a" in metrics["centroids"]
        assert "b" in metrics["centroids"]
        assert "c" in metrics["centroids"]

    def test_centroid_shape(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        for lbl, centroid in metrics["centroids"].items():
            assert centroid.shape == (16,)

    def test_inter_class_distances_shape(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        assert metrics["inter_class_distances"].shape == (3, 3)

    def test_diagonal_is_zero(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        diag = np.diag(metrics["inter_class_distances"])
        np.testing.assert_allclose(diag, 0.0, atol=1e-10)

    def test_symmetric(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        dm = metrics["inter_class_distances"]
        np.testing.assert_allclose(dm, dm.T)

    def test_silhouette_computed(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        assert metrics["silhouette"] is not None
        # Well-separated clusters should have high silhouette
        assert metrics["silhouette"] > 0.5

    def test_intra_class_variance(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        for lbl, var in metrics["intra_class_variance"].items():
            assert var >= 0.0

    def test_class_labels_sorted(self, embeddings_and_labels):
        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        assert metrics["class_labels"] == ["a", "b", "c"]

    def test_single_class_no_silhouette(self):
        """Silhouette should be None with only one class."""
        embeddings = np.random.randn(10, 8)
        labels = ["x"] * 10
        metrics = compute_cluster_metrics(embeddings, labels)
        assert metrics["silhouette"] is None


# ---------------------------------------------------------------------------
# Velocity / dynamics
# ---------------------------------------------------------------------------

class TestVelocityField:
    def test_output_shapes(self, temporal_embeddings):
        vf = compute_velocity_field(temporal_embeddings)
        T = temporal_embeddings.shape[0]
        D = temporal_embeddings.shape[1]
        assert vf["velocities"].shape == (T - 1, D)
        assert vf["speeds"].shape == (T - 1,)
        assert vf["acceleration"].shape == (T - 2, D)

    def test_speeds_non_negative(self, temporal_embeddings):
        vf = compute_velocity_field(temporal_embeddings)
        assert np.all(vf["speeds"] >= 0)

    def test_mean_speed_positive(self, temporal_embeddings):
        vf = compute_velocity_field(temporal_embeddings)
        assert vf["mean_speed"] > 0

    def test_constant_input_zero_velocity(self):
        """Constant embeddings should have zero velocity."""
        constant = np.ones((50, 10))
        vf = compute_velocity_field(constant)
        np.testing.assert_allclose(vf["speeds"], 0.0, atol=1e-10)


class TestMultiTrialDynamics:
    def test_per_trial_speeds(self, temporal_embeddings):
        trials = [temporal_embeddings, temporal_embeddings[:50]]
        labels = ["hello", "world"]
        result = compute_multi_trial_dynamics(trials, labels)
        assert len(result["per_trial"]) == 2
        assert "mean_speed" in result["per_trial"][0]

    def test_label_mean_speeds(self, temporal_embeddings):
        trials = [temporal_embeddings, temporal_embeddings]
        labels = ["a", "a"]
        result = compute_multi_trial_dynamics(trials, labels)
        assert "a" in result["label_mean_speeds"]

    def test_short_trial_handling(self):
        """Trials with < 2 timesteps should return zero speed."""
        trials = [np.ones((1, 10))]
        labels = ["x"]
        result = compute_multi_trial_dynamics(trials, labels)
        assert result["per_trial"][0]["mean_speed"] == 0.0


# ---------------------------------------------------------------------------
# Visualization (smoke tests — verify figure creation, not pixel content)
# ---------------------------------------------------------------------------

class TestPlotManifold2D:
    def test_creates_figure(self, embeddings_and_labels):
        import matplotlib
        matplotlib.use("Agg")

        embeddings, labels = embeddings_and_labels
        projected, _ = fit_pca(embeddings, n_components=2)
        fig = plot_manifold_2d(projected, labels, method="PCA")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_auto_projects_high_dim(self, embeddings_and_labels):
        """Should auto-project if embeddings have > 2 dims."""
        import matplotlib
        matplotlib.use("Agg")

        embeddings, labels = embeddings_and_labels
        fig = plot_manifold_2d(embeddings, labels)  # 16D input
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotManifold3D:
    def test_creates_figure(self, embeddings_and_labels):
        import matplotlib
        matplotlib.use("Agg")

        embeddings, labels = embeddings_and_labels
        projected, _ = fit_pca(embeddings, n_components=3)
        fig = plot_manifold_3d(projected, labels)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotNeuralDynamics3D:
    def test_creates_figure(self, temporal_embeddings):
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_neural_dynamics_3d(temporal_embeddings, label="test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotVelocityField:
    def test_creates_figure(self, temporal_embeddings):
        import matplotlib
        matplotlib.use("Agg")

        fig = plot_velocity_field(temporal_embeddings, label="test")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotClusterDistances:
    def test_creates_figure(self, embeddings_and_labels):
        import matplotlib
        matplotlib.use("Agg")

        embeddings, labels = embeddings_and_labels
        metrics = compute_cluster_metrics(embeddings, labels)
        fig = plot_cluster_distances(metrics)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
