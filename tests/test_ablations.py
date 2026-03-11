"""Tests for src/evaluation/ablations.py — ablation runner, confusion matrix,
per-character errors, per-subject metrics, and bootstrap significance testing.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.ablations import (
    compute_confusion_matrix,
    get_char_labels,
    per_character_cer,
    per_subject_metrics,
    paired_bootstrap_test,
    per_sample_cer,
    run_single_evaluation,
    AblationResult,
    _align_strings,
)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_perfect_predictions(self):
        refs = ["abc", "def"]
        preds = ["abc", "def"]
        cm = compute_confusion_matrix(preds, refs)
        assert cm.shape == (27, 27)
        # Diagonal should have counts
        assert cm[0, 0] > 0  # 'a' correct
        assert cm.trace() > 0

    def test_all_wrong(self):
        refs = ["aaa"]
        preds = ["bbb"]
        cm = compute_confusion_matrix(preds, refs)
        # a->b should have count
        assert cm[0, 1] > 0  # a confused with b

    def test_empty_inputs(self):
        cm = compute_confusion_matrix([], [])
        assert cm.shape == (27, 27)
        assert cm.sum() == 0

    def test_shape_always_27x27(self):
        cm = compute_confusion_matrix(["hello"], ["world"])
        assert cm.shape == (27, 27)


class TestCharLabels:
    def test_labels_count(self):
        labels = get_char_labels()
        assert len(labels) == 27
        assert labels[0] == "a"
        assert labels[25] == "z"
        assert labels[26] == "space"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_identical(self):
        p, r = _align_strings("abc", "abc")
        assert p == "abc"
        assert r == "abc"

    def test_insertion(self):
        p, r = _align_strings("abxc", "abc")
        assert len(p) == len(r)

    def test_deletion(self):
        p, r = _align_strings("ac", "abc")
        assert len(p) == len(r)

    def test_empty_strings(self):
        p, r = _align_strings("", "")
        assert p == ""
        assert r == ""

    def test_one_empty(self):
        p, r = _align_strings("", "abc")
        assert len(p) == len(r)


# ---------------------------------------------------------------------------
# Per-character CER
# ---------------------------------------------------------------------------

class TestPerCharacterCER:
    def test_perfect(self):
        rates = per_character_cer(["abc"], ["abc"])
        for ch, rate in rates.items():
            assert rate == 0.0

    def test_all_wrong(self):
        rates = per_character_cer(["bcd"], ["abc"])
        # a should have error
        assert rates.get("a", 0.0) > 0.0

    def test_returns_dict(self):
        rates = per_character_cer(["hello world"], ["hello world"])
        assert isinstance(rates, dict)

    def test_empty(self):
        rates = per_character_cer([], [])
        assert rates == {}


# ---------------------------------------------------------------------------
# Per-subject metrics
# ---------------------------------------------------------------------------

class TestPerSubjectMetrics:
    def test_single_subject(self):
        preds = ["abc", "def"]
        refs = ["abc", "def"]
        subjects = [1, 1]
        result = per_subject_metrics(preds, refs, subjects)
        assert 1 in result
        assert result[1]["cer"] == 0.0
        assert result[1]["exact_match"] == 1.0
        assert result[1]["n_samples"] == 2

    def test_multiple_subjects(self):
        preds = ["abc", "xyz"]
        refs = ["abc", "xyz"]
        subjects = [1, 2]
        result = per_subject_metrics(preds, refs, subjects)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result


# ---------------------------------------------------------------------------
# Bootstrap test
# ---------------------------------------------------------------------------

class TestBootstrapTest:
    def test_identical_scores(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = paired_bootstrap_test(scores, scores, n_resamples=100)
        assert abs(result.mean_diff) < 1e-10
        assert result.p_value > 0.05

    def test_clearly_different(self):
        scores_a = [0.0] * 50
        scores_b = [1.0] * 50
        result = paired_bootstrap_test(scores_a, scores_b, n_resamples=500)
        assert result.mean_diff < 0
        assert result.significant

    def test_returns_bootstrap_result(self):
        result = paired_bootstrap_test([0.1, 0.2], [0.3, 0.4], n_resamples=100)
        assert hasattr(result, "mean_diff")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "p_value")
        assert result.n_resamples == 100


# ---------------------------------------------------------------------------
# Per-sample CER
# ---------------------------------------------------------------------------

class TestPerSampleCER:
    def test_perfect(self):
        scores = per_sample_cer(["abc"], ["abc"])
        assert scores == [0.0]

    def test_imperfect(self):
        scores = per_sample_cer(["abd"], ["abc"])
        assert len(scores) == 1
        assert scores[0] > 0

    def test_empty_ref(self):
        scores = per_sample_cer([""], [""])
        assert scores == [0.0]


# ---------------------------------------------------------------------------
# AblationResult
# ---------------------------------------------------------------------------

class TestAblationResult:
    def test_to_dict(self):
        r = AblationResult(
            model_name="gru",
            decoding_method="greedy",
            augmentation="none",
            feature_pathway="default",
            cer=0.3,
            wer=0.5,
            exact_match=0.1,
            n_samples=100,
            inference_time_s=1.5,
        )
        d = r.to_dict()
        assert "model_name" in d
        assert "predictions" not in d
        assert "references" not in d
        assert d["cer"] == 0.3


# ---------------------------------------------------------------------------
# Run single evaluation (with dummy model)
# ---------------------------------------------------------------------------

class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(192, 28)

    def forward(self, x):
        return self.linear(x)


class TestRunSingleEvaluation:
    def _make_dataloader(self):
        """Create a minimal dataloader with synthetic data."""
        from torch.utils.data import DataLoader, TensorDataset

        B, T, C = 4, 100, 192
        features = torch.randn(B, T, C)
        # Build batch dicts manually
        class _DS:
            def __init__(self):
                self.data = [
                    {
                        "features": features[i],
                        "label_texts": "hello",
                    }
                    for i in range(B)
                ]

            def __iter__(self):
                batch = {
                    "features": features,
                    "label_texts": ["hello"] * B,
                }
                yield batch

        return _DS()

    def test_greedy_evaluation(self):
        model = _DummyModel()
        dl = self._make_dataloader()
        preds, refs, elapsed = run_single_evaluation(
            model, dl, decoding_method="greedy", device="cpu",
        )
        assert len(preds) == 4
        assert len(refs) == 4
        assert elapsed >= 0

    def test_beam_evaluation(self):
        model = _DummyModel()
        dl = self._make_dataloader()
        preds, refs, elapsed = run_single_evaluation(
            model, dl, decoding_method="beam", beam_width=5, device="cpu",
        )
        assert len(preds) == 4
