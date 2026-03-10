"""Tests for src/data/loader.py — data loading, parsing, and trial-index building."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import convert_to_standard_format, build_trial_index_from_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trials():
    """Create a list of fake trial dicts matching the loader's internal format."""
    trials = []
    for i in range(10):
        T = np.random.randint(200, 800)
        C = 192
        trials.append({
            "neural": np.random.randn(T, C).astype(np.float32),
            "label": f"trial {chr(ord('a') + i)}",
            "metadata": {"source_file": "fake.mat", "trial_idx": i},
        })
    return trials


@pytest.fixture
def standard_dataset(tmp_path, sample_trials):
    """Convert sample trials to the standardized directory format."""
    dataset_dir = tmp_path / "willett_handwriting"
    df = convert_to_standard_format(
        sample_trials, dataset_dir, subject_id=1, session_id=1
    )
    return dataset_dir, df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrialIndex:
    EXPECTED_COLUMNS = [
        "subject", "session", "trial_id",
        "signal_path", "label_path",
        "n_timesteps", "n_channels",
    ]

    def test_columns_present(self, standard_dataset):
        _, df = standard_dataset
        for col in self.EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_non_zero_rows(self, standard_dataset):
        _, df = standard_dataset
        assert len(df) > 0

    def test_row_count_matches_input(self, sample_trials, standard_dataset):
        _, df = standard_dataset
        assert len(df) == len(sample_trials)

    def test_signal_paths_exist(self, standard_dataset):
        _, df = standard_dataset
        from pathlib import Path
        for path in df["signal_path"]:
            assert Path(path).exists(), f"Signal file missing: {path}"

    def test_label_paths_exist(self, standard_dataset):
        _, df = standard_dataset
        from pathlib import Path
        for path in df["label_path"]:
            assert Path(path).exists(), f"Label file missing: {path}"

    def test_n_timesteps_positive(self, standard_dataset):
        _, df = standard_dataset
        assert (df["n_timesteps"] > 0).all()

    def test_n_channels_correct(self, standard_dataset):
        _, df = standard_dataset
        assert (df["n_channels"] == 192).all()


class TestStandardFormat:
    def test_signal_roundtrip(self, sample_trials, standard_dataset):
        _, df = standard_dataset
        row = df.iloc[0]
        loaded = np.load(row["signal_path"])
        original = sample_trials[0]["neural"]
        np.testing.assert_array_equal(loaded, original)

    def test_label_roundtrip(self, sample_trials, standard_dataset):
        _, df = standard_dataset
        row = df.iloc[0]
        loaded = open(row["label_path"]).read().strip()
        assert loaded == sample_trials[0]["label"]


class TestBuildTrialIndexFromDir:
    def test_builds_index(self, standard_dataset):
        dataset_dir, original_df = standard_dataset
        rebuilt = build_trial_index_from_dir(dataset_dir)
        assert len(rebuilt) == len(original_df)
        assert set(rebuilt.columns) == set(original_df.columns)
