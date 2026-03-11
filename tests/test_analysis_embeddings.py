"""Tests for src/analysis/embeddings.py — hidden-layer embedding extraction."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.analysis.embeddings import (
    extract_embeddings,
    save_embeddings,
    load_embeddings,
    _find_layer,
)


# Minimal model for testing
class _TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.output_proj = nn.Linear(64, 28)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.output_proj(h)


class _SimpleLoader:
    """Minimal dataloader for testing."""
    def __init__(self, n_batches=2, batch_size=4, T=50, C=32):
        self.batches = []
        for _ in range(n_batches):
            self.batches.append({
                "features": torch.randn(batch_size, T, C),
                "label_texts": ["test"] * batch_size,
            })

    def __iter__(self):
        return iter(self.batches)


class TestExtractEmbeddings:
    def test_basic_extraction(self):
        model = _TestModel()
        loader = _SimpleLoader()
        result = extract_embeddings(model, loader, device="cpu")
        assert "embeddings" in result
        assert "labels" in result
        assert "layer_name" in result
        assert result["embeddings"].shape[0] == 8  # 2 batches × 4
        assert result["embeddings"].shape[1] == 64  # GRU hidden dim

    def test_specific_layer(self):
        model = _TestModel()
        loader = _SimpleLoader()
        result = extract_embeddings(model, loader, layer_name="gru", device="cpu")
        assert result["layer_name"] == "gru"
        assert result["embeddings"].shape[0] == 8

    def test_labels_preserved(self):
        model = _TestModel()
        loader = _SimpleLoader(n_batches=1, batch_size=3)
        result = extract_embeddings(model, loader, device="cpu")
        assert len(result["labels"]) == 3
        assert all(l == "test" for l in result["labels"])

    def test_invalid_layer_name(self):
        model = _TestModel()
        loader = _SimpleLoader()
        with pytest.raises(ValueError, match="not found"):
            extract_embeddings(model, loader, layer_name="nonexistent", device="cpu")


class TestSaveLoadEmbeddings:
    def test_save_and_load(self, tmp_path):
        data = {
            "embeddings": np.random.randn(10, 64),
            "labels": ["a", "b", "c", "d", "e"] * 2,
            "layer_name": "gru",
        }
        path = save_embeddings(data, output_dir=tmp_path, prefix="test")
        assert path.exists()

        loaded = load_embeddings(path)
        assert loaded["embeddings"].shape == (10, 64)
        assert len(loaded["labels"]) == 10
        assert loaded["layer_name"] == "gru"
        np.testing.assert_array_almost_equal(loaded["embeddings"], data["embeddings"])


class TestFindLayer:
    def test_auto_detect_gru(self):
        model = _TestModel()
        layer = _find_layer(model, None)
        assert isinstance(layer, nn.GRU)

    def test_by_name(self):
        model = _TestModel()
        layer = _find_layer(model, "output_proj")
        assert isinstance(layer, nn.Linear)
