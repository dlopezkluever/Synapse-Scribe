"""Tests for src/analysis/similarity_matrix.py — cosine similarity analysis."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.similarity_matrix import (
    compute_cosine_similarity,
    compute_class_similarity,
    plot_similarity_matrix,
    plot_class_similarity,
)


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


class TestCosineSimilarity:
    def test_identity(self):
        embeddings = np.eye(5)
        sim = compute_cosine_similarity(embeddings)
        np.testing.assert_array_almost_equal(sim, np.eye(5))

    def test_self_similarity_is_one(self):
        embeddings = np.random.randn(10, 32)
        sim = compute_cosine_similarity(embeddings)
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(10), decimal=5)

    def test_symmetric(self):
        embeddings = np.random.randn(20, 16)
        sim = compute_cosine_similarity(embeddings)
        np.testing.assert_array_almost_equal(sim, sim.T, decimal=5)

    def test_shape(self):
        embeddings = np.random.randn(15, 64)
        sim = compute_cosine_similarity(embeddings)
        assert sim.shape == (15, 15)

    def test_values_bounded(self):
        embeddings = np.random.randn(10, 8)
        sim = compute_cosine_similarity(embeddings)
        assert sim.min() >= -1.0 - 1e-6
        assert sim.max() <= 1.0 + 1e-6

    def test_zero_vectors(self):
        embeddings = np.zeros((3, 4))
        sim = compute_cosine_similarity(embeddings)
        # Zero vectors normalized to zero, so 0·0 = 0
        assert sim.shape == (3, 3)


class TestClassSimilarity:
    def test_basic(self):
        embeddings = np.random.randn(20, 16)
        labels = ["a"] * 10 + ["b"] * 10
        class_sim, class_labels = compute_class_similarity(embeddings, labels)
        assert class_sim.shape == (2, 2)
        assert len(class_labels) == 2
        assert "a" in class_labels
        assert "b" in class_labels

    def test_diagonal_higher(self):
        # Same class should be more similar (with enough separation)
        a = np.random.randn(20, 16) + np.array([5.0] * 16)
        b = np.random.randn(20, 16) + np.array([-5.0] * 16)
        embeddings = np.concatenate([a, b], axis=0)
        labels = ["a"] * 20 + ["b"] * 20
        class_sim, _ = compute_class_similarity(embeddings, labels)
        # Within-class should be > between-class
        assert class_sim[0, 0] > class_sim[0, 1]

    def test_space_label_display(self):
        embeddings = np.random.randn(10, 8)
        labels = [" "] * 5 + ["a"] * 5
        _, class_labels = compute_class_similarity(embeddings, labels)
        assert "space" in class_labels


class TestPlotSimilarityMatrix:
    def test_basic(self):
        sim = np.random.randn(10, 10)
        fig = plot_similarity_matrix(sim)
        assert isinstance(fig, plt.Figure)

    def test_with_labels(self):
        sim = np.random.randn(5, 5)
        fig = plot_similarity_matrix(sim, labels=["a", "b", "c", "d", "e"])
        assert isinstance(fig, plt.Figure)

    def test_save(self, tmp_path):
        sim = np.eye(5)
        path = tmp_path / "sim.png"
        fig = plot_similarity_matrix(sim, save_path=str(path))
        assert path.exists()


class TestPlotClassSimilarity:
    def test_basic(self):
        embeddings = np.random.randn(20, 16)
        labels = ["a"] * 10 + ["b"] * 10
        fig = plot_class_similarity(embeddings, labels)
        assert isinstance(fig, plt.Figure)
