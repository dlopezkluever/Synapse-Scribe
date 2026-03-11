"""Tests for src/analysis/trajectory_plots.py — neural state trajectories."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.trajectory_plots import (
    extract_temporal_embeddings,
    plot_neural_trajectory,
    plot_multi_trial_trajectories,
    _pca_2d,
)


class _TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.output_proj = nn.Linear(64, 28)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.output_proj(h)


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


class TestExtractTemporalEmbeddings:
    def test_numpy_input(self):
        model = _TestModel()
        features = np.random.randn(100, 32).astype(np.float32)
        hidden = extract_temporal_embeddings(model, features, device="cpu")
        assert hidden.shape == (100, 64)

    def test_torch_input(self):
        model = _TestModel()
        features = torch.randn(1, 50, 32)
        hidden = extract_temporal_embeddings(model, features, device="cpu")
        assert hidden.shape == (50, 64)

    def test_2d_input(self):
        model = _TestModel()
        features = torch.randn(80, 32)
        hidden = extract_temporal_embeddings(model, features, device="cpu")
        assert hidden.shape == (80, 64)


class TestPlotNeuralTrajectory:
    def test_basic(self):
        hidden = np.random.randn(100, 64)
        fig = plot_neural_trajectory(hidden)
        assert isinstance(fig, plt.Figure)

    def test_with_label(self):
        hidden = np.random.randn(50, 32)
        fig = plot_neural_trajectory(hidden, label="hello")
        assert isinstance(fig, plt.Figure)

    def test_save(self, tmp_path):
        hidden = np.random.randn(50, 16)
        path = tmp_path / "traj.png"
        fig = plot_neural_trajectory(hidden, save_path=str(path))
        assert path.exists()


class TestMultiTrialTrajectories:
    def test_basic(self):
        trajs = [np.random.randn(50, 32) for _ in range(5)]
        labels = ["a", "b", "c", "d", "e"]
        fig = plot_multi_trial_trajectories(trajs, labels)
        assert isinstance(fig, plt.Figure)

    def test_duplicate_labels(self):
        trajs = [np.random.randn(30, 16) for _ in range(6)]
        labels = ["a", "a", "b", "b", "c", "c"]
        fig = plot_multi_trial_trajectories(trajs, labels)
        assert isinstance(fig, plt.Figure)


class TestPCA2D:
    def test_output_shape(self):
        data = np.random.randn(50, 64)
        coords = _pca_2d(data)
        assert coords.shape == (50, 2)

    def test_small_dim(self):
        data = np.random.randn(10, 2)
        coords = _pca_2d(data)
        assert coords.shape == (10, 2)
