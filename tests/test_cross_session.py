"""Tests for cross-session generalization (src/training/cross_session.py).

Verifies session-based splitting, per-session normalization, and
cross-session evaluation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.gru_decoder import GRUDecoder
from src.training.cross_session import (
    SessionNormalizedDataset,
    SessionNormalizer,
    cross_session_report,
    evaluate_cross_session,
    get_available_sessions,
    split_by_session,
)
from src.data.dataset import ctc_collate_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    return GRUDecoder(
        n_channels=4, n_classes=28, proj_dim=8,
        hidden_size=16, n_layers=1, dropout=0.0,
    )


@pytest.fixture
def multi_session_trial_index(tmp_path):
    """Trial index with 2 sessions, each with 6 trials."""
    rows = []
    for sess in ["ses1", "ses2"]:
        for trial in range(6):
            sig_dir = tmp_path / "sub01" / sess / "neural"
            sig_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir = tmp_path / "sub01" / sess / "labels"
            lbl_dir.mkdir(parents=True, exist_ok=True)

            sig_path = sig_dir / f"trial_{trial:04d}_signals.npy"
            lbl_path = lbl_dir / f"trial_{trial:04d}_transcript.txt"

            # Different distributions per session to test normalization
            if sess == "ses1":
                data = np.random.randn(20, 4).astype(np.float32) * 2.0 + 1.0
            else:
                data = np.random.randn(20, 4).astype(np.float32) * 0.5 - 1.0

            np.save(sig_path, data)
            lbl_path.write_text("ab", encoding="utf-8")

            rows.append({
                "subject": "sub01",
                "session": sess,
                "trial_id": trial,
                "signal_path": str(sig_path),
                "label_path": str(lbl_path),
                "n_timesteps": 20,
                "n_channels": 4,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Session splitting
# ---------------------------------------------------------------------------

class TestSplitBySession:
    def test_splits_correctly(self, multi_session_trial_index):
        train_df, eval_df = split_by_session(
            multi_session_trial_index,
            train_sessions=["ses1"],
            eval_sessions=["ses2"],
        )
        assert len(train_df) == 6
        assert len(eval_df) == 6
        assert all(train_df["session"].astype(str) == "ses1")
        assert all(eval_df["session"].astype(str) == "ses2")

    def test_reverse_split(self, multi_session_trial_index):
        train_df, eval_df = split_by_session(
            multi_session_trial_index,
            train_sessions=["ses2"],
            eval_sessions=["ses1"],
        )
        assert len(train_df) == 6
        assert len(eval_df) == 6

    def test_empty_result_for_unknown_session(self, multi_session_trial_index):
        train_df, eval_df = split_by_session(
            multi_session_trial_index,
            train_sessions=["ses1"],
            eval_sessions=["ses999"],
        )
        assert len(eval_df) == 0


class TestGetAvailableSessions:
    def test_returns_sorted(self, multi_session_trial_index):
        sessions = get_available_sessions(multi_session_trial_index)
        assert sessions == ["ses1", "ses2"]


# ---------------------------------------------------------------------------
# Session normalization
# ---------------------------------------------------------------------------

class TestSessionNormalizer:
    def test_computes_stats(self, multi_session_trial_index):
        normalizer = SessionNormalizer(multi_session_trial_index)
        assert "ses1" in normalizer.stats
        assert "ses2" in normalizer.stats

    def test_stats_have_mean_and_std(self, multi_session_trial_index):
        normalizer = SessionNormalizer(multi_session_trial_index)
        for sess, stats in normalizer.stats.items():
            assert "mean" in stats
            assert "std" in stats
            assert stats["mean"].shape == (4,)
            assert stats["std"].shape == (4,)

    def test_normalize_output_shape(self, multi_session_trial_index):
        normalizer = SessionNormalizer(multi_session_trial_index)
        features = np.random.randn(20, 4).astype(np.float32)
        out = normalizer.normalize(features, "ses1")
        assert out.shape == (20, 4)

    def test_different_sessions_different_stats(self, multi_session_trial_index):
        normalizer = SessionNormalizer(multi_session_trial_index)
        # Sessions have different distributions (see fixture)
        mean_1 = normalizer.stats["ses1"]["mean"]
        mean_2 = normalizer.stats["ses2"]["mean"]
        assert not np.allclose(mean_1, mean_2, atol=0.1)

    def test_normalized_close_to_zero_mean(self, multi_session_trial_index):
        """After normalization, data should be roughly zero-mean."""
        normalizer = SessionNormalizer(multi_session_trial_index)
        # Load all ses1 trials and normalize
        ses1 = multi_session_trial_index[
            multi_session_trial_index["session"] == "ses1"
        ]
        all_normed = []
        for _, row in ses1.iterrows():
            data = np.load(row["signal_path"]).astype(np.float32)
            normed = normalizer.normalize(data, "ses1")
            all_normed.append(normed)
        combined = np.concatenate(all_normed, axis=0)
        assert abs(combined.mean()) < 0.3  # should be approximately zero

    def test_unknown_session_uses_fallback(self, multi_session_trial_index):
        normalizer = SessionNormalizer(multi_session_trial_index)
        features = np.random.randn(10, 4).astype(np.float32)
        out = normalizer.normalize(features, "unknown_session")
        assert out.shape == (10, 4)


# ---------------------------------------------------------------------------
# SessionNormalizedDataset
# ---------------------------------------------------------------------------

class TestSessionNormalizedDataset:
    def test_returns_session_field(self, multi_session_trial_index):
        normalizer = SessionNormalizer(multi_session_trial_index)
        ds = SessionNormalizedDataset(
            multi_session_trial_index, normalizer, t_max=20,
        )
        item = ds[0]
        assert "session" in item
        assert "features" in item

    def test_dataset_length(self, multi_session_trial_index):
        normalizer = SessionNormalizer(multi_session_trial_index)
        ds = SessionNormalizedDataset(
            multi_session_trial_index, normalizer, t_max=20,
        )
        assert len(ds) == 12


# ---------------------------------------------------------------------------
# Cross-session evaluation
# ---------------------------------------------------------------------------

class TestEvaluateCrossSession:
    def test_returns_cer(self, multi_session_trial_index, tiny_model):
        normalizer = SessionNormalizer(multi_session_trial_index)
        _, eval_df = split_by_session(
            multi_session_trial_index,
            train_sessions=["ses1"],
            eval_sessions=["ses2"],
        )
        ds = SessionNormalizedDataset(eval_df, normalizer, t_max=20)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=ctc_collate_fn,
        )
        result = evaluate_cross_session(tiny_model, loader, device="cpu")
        assert "cer" in result
        assert "n_trials" in result
        assert result["n_trials"] == 6

    def test_returns_predictions_and_references(self, multi_session_trial_index, tiny_model):
        normalizer = SessionNormalizer(multi_session_trial_index)
        _, eval_df = split_by_session(
            multi_session_trial_index,
            train_sessions=["ses1"],
            eval_sessions=["ses2"],
        )
        ds = SessionNormalizedDataset(eval_df, normalizer, t_max=20)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=ctc_collate_fn,
        )
        result = evaluate_cross_session(tiny_model, loader, device="cpu")
        assert len(result["predictions"]) == 6
        assert len(result["references"]) == 6


class TestCrossSessionReport:
    def test_generates_report(self, multi_session_trial_index, tiny_model):
        results = cross_session_report(
            tiny_model,
            multi_session_trial_index,
            t_max=20,
            batch_size=4,
            device="cpu",
            use_session_norm=True,
        )
        assert "ses1" in results
        assert "ses2" in results
        for sess, res in results.items():
            assert "cer" in res
            assert "n_trials" in res

    def test_report_without_normalization(self, multi_session_trial_index, tiny_model):
        results = cross_session_report(
            tiny_model,
            multi_session_trial_index,
            t_max=20,
            batch_size=4,
            device="cpu",
            use_session_norm=False,
        )
        assert len(results) == 2
