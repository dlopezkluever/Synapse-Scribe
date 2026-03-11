"""Tests for src/data/dataset.py — NeuralTrialDataset, collate, splits."""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.data.dataset import (
    text_to_indices,
    indices_to_text,
    NeuralTrialDataset,
    ctc_collate_fn,
    split_trial_index,
    VOCAB_SIZE,
    BLANK_IDX,
    SPACE_IDX,
)


class TestVocabulary:
    def test_vocab_size(self):
        assert VOCAB_SIZE == 28

    def test_blank_is_zero(self):
        assert BLANK_IDX == 0

    def test_space_is_27(self):
        assert SPACE_IDX == 27

    def test_text_to_indices(self):
        indices = text_to_indices("abc")
        assert indices == [1, 2, 3]

    def test_text_to_indices_space(self):
        indices = text_to_indices("a b")
        assert indices == [1, 27, 2]

    def test_text_to_indices_case_insensitive(self):
        assert text_to_indices("ABC") == text_to_indices("abc")

    def test_indices_to_text(self):
        text = indices_to_text([1, 2, 3])
        assert text == "abc"

    def test_roundtrip(self):
        original = "hello world"
        indices = text_to_indices(original)
        recovered = indices_to_text(indices)
        assert recovered == original

    def test_unknown_chars_skipped(self):
        indices = text_to_indices("a1b!c")
        assert indices == [1, 2, 3]  # only a, b, c


class TestNeuralTrialDataset:
    @pytest.fixture
    def trial_data(self, tmp_path):
        """Create synthetic trial data on disk."""
        neural_dir = tmp_path / "neural"
        label_dir = tmp_path / "labels"
        neural_dir.mkdir()
        label_dir.mkdir()

        rows = []
        for i in range(10):
            T = np.random.randint(100, 300)
            sig = np.random.randn(T, 32).astype(np.float32)
            sig_path = neural_dir / f"trial_{i:04d}_signals.npy"
            np.save(sig_path, sig)

            label = "hello" if i % 2 == 0 else "world"
            lbl_path = label_dir / f"trial_{i:04d}_transcript.txt"
            lbl_path.write_text(label)

            rows.append({
                "subject": 1,
                "session": 1,
                "trial_id": i,
                "signal_path": str(sig_path),
                "label_path": str(lbl_path),
                "n_timesteps": T,
                "n_channels": 32,
            })
        return pd.DataFrame(rows)

    def test_dataset_length(self, trial_data):
        ds = NeuralTrialDataset(trial_data, t_max=200)
        assert len(ds) == 10

    def test_getitem_keys(self, trial_data):
        ds = NeuralTrialDataset(trial_data, t_max=200)
        item = ds[0]
        assert "features" in item
        assert "target" in item
        assert "input_length" in item
        assert "target_length" in item
        assert "label_text" in item

    def test_features_shape(self, trial_data):
        ds = NeuralTrialDataset(trial_data, t_max=200)
        item = ds[0]
        assert item["features"].shape == (200, 32)

    def test_target_is_tensor(self, trial_data):
        ds = NeuralTrialDataset(trial_data, t_max=200)
        item = ds[0]
        assert isinstance(item["target"], torch.Tensor)
        assert item["target"].dtype == torch.long


class TestCollate:
    @pytest.fixture
    def batch(self, tmp_path):
        neural_dir = tmp_path / "neural"
        label_dir = tmp_path / "labels"
        neural_dir.mkdir()
        label_dir.mkdir()

        rows = []
        for i in range(4):
            sig = np.random.randn(150, 16).astype(np.float32)
            sig_path = neural_dir / f"trial_{i}_signals.npy"
            np.save(sig_path, sig)
            lbl_path = label_dir / f"trial_{i}_transcript.txt"
            lbl_path.write_text("cat")
            rows.append({
                "subject": 1, "session": 1, "trial_id": i,
                "signal_path": str(sig_path), "label_path": str(lbl_path),
                "n_timesteps": 150, "n_channels": 16,
            })

        ds = NeuralTrialDataset(pd.DataFrame(rows), t_max=200)
        return [ds[i] for i in range(4)]

    def test_collate_keys(self, batch):
        collated = ctc_collate_fn(batch)
        assert "features" in collated
        assert "targets" in collated
        assert "input_lengths" in collated
        assert "target_lengths" in collated

    def test_collate_shapes(self, batch):
        collated = ctc_collate_fn(batch)
        assert collated["features"].shape == (4, 200, 16)
        assert collated["input_lengths"].shape == (4,)
        assert collated["target_lengths"].shape == (4,)
        # "cat" = 3 chars × 4 samples = 12 targets
        assert collated["targets"].shape == (12,)


class TestSplitTrialIndex:
    def test_split_sizes(self):
        df = pd.DataFrame({
            "subject": [1] * 100,
            "session": [1] * 100,
            "trial_id": range(100),
        })
        train, val, test = split_trial_index(df, [0.8, 0.1, 0.1])
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        df = pd.DataFrame({
            "subject": [1] * 50,
            "session": [1] * 50,
            "trial_id": range(50),
        })
        train, val, test = split_trial_index(df, [0.6, 0.2, 0.2])
        all_ids = set(train["trial_id"]) | set(val["trial_id"]) | set(test["trial_id"])
        assert len(all_ids) == 50  # no duplicates
