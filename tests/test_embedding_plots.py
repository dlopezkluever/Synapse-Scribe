"""Tests for src/visualization/embedding_plots.py — t-SNE/PCA scatter plots."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.embedding_plots import plot_embedding_scatter, _reduce_dims


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


class TestEmbeddingScatter:
    def test_basic_pca(self):
        embeddings = np.random.randn(50, 64)
        labels = [chr(ord("a") + (i % 5)) for i in range(50)]
        fig = plot_embedding_scatter(embeddings, labels, method="pca")
        assert isinstance(fig, plt.Figure)

    def test_with_numpy_labels(self):
        embeddings = np.random.randn(30, 32)
        labels = np.array(["a", "b", "c"] * 10)
        fig = plot_embedding_scatter(embeddings, labels, method="pca")
        assert isinstance(fig, plt.Figure)

    def test_save_path(self, tmp_path):
        embeddings = np.random.randn(20, 16)
        labels = ["x"] * 20
        path = tmp_path / "scatter.png"
        fig = plot_embedding_scatter(embeddings, labels, method="pca",
                                     save_path=str(path))
        assert path.exists()

    def test_many_classes(self):
        embeddings = np.random.randn(100, 32)
        labels = [chr(ord("a") + (i % 26)) for i in range(100)]
        fig = plot_embedding_scatter(embeddings, labels, method="pca")
        assert isinstance(fig, plt.Figure)


class TestReduceDims:
    def test_pca_output_shape(self):
        data = np.random.randn(50, 64)
        coords = _reduce_dims(data, method="pca")
        assert coords.shape == (50, 2)

    def test_pca_with_small_dims(self):
        data = np.random.randn(10, 2)
        coords = _reduce_dims(data, method="pca")
        assert coords.shape == (10, 2)

    def test_pca_single_dim(self):
        data = np.random.randn(10, 1)
        coords = _reduce_dims(data, method="pca")
        assert coords.shape == (10, 2)
