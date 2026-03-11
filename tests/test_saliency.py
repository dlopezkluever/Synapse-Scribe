"""Tests for src/analysis/saliency.py — gradient-based electrode importance."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.saliency import (
    input_x_gradient,
    integrated_gradients,
    electrode_importance,
    plot_electrode_importance,
    plot_electrode_heatmap,
)


class _TestModel(nn.Module):
    def __init__(self, n_channels=32, n_classes=28):
        super().__init__()
        self.linear = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


class TestInputXGradient:
    def test_basic(self):
        model = _TestModel()
        features = np.random.randn(100, 32).astype(np.float32)
        attr = input_x_gradient(model, features, device="cpu")
        assert attr.shape == (100, 32)

    def test_specific_class(self):
        model = _TestModel()
        features = torch.randn(50, 32)
        attr = input_x_gradient(model, features, target_class=5, device="cpu")
        assert attr.shape == (50, 32)

    def test_3d_input(self):
        model = _TestModel()
        features = torch.randn(1, 80, 32)
        attr = input_x_gradient(model, features, device="cpu")
        assert attr.shape == (80, 32)

    def test_nonzero_attribution(self):
        model = _TestModel()
        features = torch.randn(50, 32)
        attr = input_x_gradient(model, features, device="cpu")
        # Attribution should be non-trivial
        assert np.abs(attr).sum() > 0


class TestIntegratedGradients:
    def test_basic(self):
        model = _TestModel()
        features = np.random.randn(50, 32).astype(np.float32)
        attr = integrated_gradients(model, features, n_steps=10, device="cpu")
        assert attr.shape == (50, 32)

    def test_specific_class(self):
        model = _TestModel()
        features = torch.randn(30, 32)
        attr = integrated_gradients(model, features, target_class=3,
                                     n_steps=5, device="cpu")
        assert attr.shape == (30, 32)


class TestElectrodeImportance:
    def test_mean_abs(self):
        attr = np.random.randn(100, 32)
        imp = electrode_importance(attr, aggregate="mean_abs")
        assert imp.shape == (32,)
        assert (imp >= 0).all()

    def test_sum_abs(self):
        attr = np.random.randn(100, 32)
        imp = electrode_importance(attr, aggregate="sum_abs")
        assert imp.shape == (32,)
        assert (imp >= 0).all()


class TestPlotElectrodeImportance:
    def test_basic(self):
        imp = np.random.rand(32)
        fig = plot_electrode_importance(imp)
        assert isinstance(fig, plt.Figure)

    def test_top_n(self):
        imp = np.random.rand(192)
        fig = plot_electrode_importance(imp, n_channels=20)
        assert isinstance(fig, plt.Figure)

    def test_save(self, tmp_path):
        imp = np.random.rand(64)
        path = tmp_path / "imp.png"
        fig = plot_electrode_importance(imp, save_path=str(path))
        assert path.exists()


class TestPlotElectrodeHeatmap:
    def test_basic(self):
        attr = np.random.randn(100, 32)
        fig = plot_electrode_heatmap(attr)
        assert isinstance(fig, plt.Figure)

    def test_save(self, tmp_path):
        attr = np.random.randn(50, 16)
        path = tmp_path / "heatmap.png"
        fig = plot_electrode_heatmap(attr, save_path=str(path))
        assert path.exists()
