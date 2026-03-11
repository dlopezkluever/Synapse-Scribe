"""Tests for src/visualization/ctc_plots.py — CTC probability heatmaps,
per-character error charts, training curves, and confusion matrix plots.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.ctc_plots import (
    plot_ctc_heatmap,
    plot_per_character_errors,
    plot_training_curves,
    plot_confusion_matrix,
)


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


class TestCTCHeatmap:
    def test_basic_heatmap(self):
        logits = np.random.randn(200, 28)
        fig = plot_ctc_heatmap(logits)
        assert isinstance(fig, plt.Figure)

    def test_with_reference(self):
        logits = np.random.randn(100, 28)
        fig = plot_ctc_heatmap(logits, reference="hello")
        assert isinstance(fig, plt.Figure)

    def test_3d_input(self):
        logits = np.random.randn(1, 100, 28)
        fig = plot_ctc_heatmap(logits)
        assert isinstance(fig, plt.Figure)

    def test_truncation(self):
        logits = np.random.randn(1000, 28)
        fig = plot_ctc_heatmap(logits, max_timesteps=200)
        assert isinstance(fig, plt.Figure)

    def test_save_path(self, tmp_path):
        logits = np.random.randn(100, 28)
        save_path = tmp_path / "heatmap.png"
        fig = plot_ctc_heatmap(logits, save_path=str(save_path))
        assert save_path.exists()


class TestPerCharacterErrors:
    def test_basic_plot(self):
        errors = {"a": 0.1, "b": 0.2, "c": 0.05, "d": 0.3}
        fig = plot_per_character_errors(errors)
        assert isinstance(fig, plt.Figure)

    def test_empty_errors(self):
        fig = plot_per_character_errors({})
        assert isinstance(fig, plt.Figure)

    def test_full_alphabet(self):
        errors = {chr(ord("a") + i): np.random.random() for i in range(26)}
        errors[" "] = 0.15
        fig = plot_per_character_errors(errors)
        assert isinstance(fig, plt.Figure)

    def test_save(self, tmp_path):
        errors = {"a": 0.1, "b": 0.2}
        path = tmp_path / "errors.png"
        fig = plot_per_character_errors(errors, save_path=str(path))
        assert path.exists()


class TestTrainingCurves:
    def test_single_model(self):
        histories = {
            "GRU": {
                "train_losses": [1.0, 0.8, 0.6],
                "val_losses": [1.1, 0.9, 0.7],
                "val_cers": [0.9, 0.7, 0.5],
                "learning_rates": [1e-3, 8e-4, 5e-4],
            }
        }
        fig = plot_training_curves(histories)
        assert isinstance(fig, plt.Figure)

    def test_multiple_models(self):
        histories = {
            "GRU": {
                "train_losses": [1.0, 0.8],
                "val_losses": [1.1, 0.9],
                "val_cers": [0.9, 0.7],
                "learning_rates": [1e-3, 8e-4],
            },
            "CNN-LSTM": {
                "train_losses": [0.9, 0.7],
                "val_losses": [1.0, 0.8],
                "val_cers": [0.8, 0.6],
                "learning_rates": [1e-3, 8e-4],
            },
        }
        fig = plot_training_curves(histories)
        assert isinstance(fig, plt.Figure)

    def test_empty_histories(self):
        fig = plot_training_curves({})
        assert isinstance(fig, plt.Figure)


class TestConfusionMatrixPlot:
    def test_basic_plot(self):
        cm = np.random.randint(0, 10, (27, 27))
        fig = plot_confusion_matrix(cm)
        assert isinstance(fig, plt.Figure)

    def test_with_labels(self):
        cm = np.random.randint(0, 10, (5, 5))
        fig = plot_confusion_matrix(cm, labels=["a", "b", "c", "d", "e"])
        assert isinstance(fig, plt.Figure)

    def test_save(self, tmp_path):
        cm = np.eye(27, dtype=np.int64)
        path = tmp_path / "cm.png"
        fig = plot_confusion_matrix(cm, save_path=str(path))
        assert path.exists()
