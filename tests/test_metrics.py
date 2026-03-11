"""Tests for src/evaluation/metrics.py — CER, WER, exact match."""

import pytest

from src.evaluation.metrics import (
    compute_cer,
    compute_wer,
    exact_match_accuracy,
    _edit_distance,
)


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_insertion(self):
        assert _edit_distance("helo", "hello") == 1

    def test_deletion(self):
        assert _edit_distance("hello", "helo") == 1

    def test_substitution(self):
        assert _edit_distance("hello", "hallo") == 1

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0
        assert _edit_distance("abc", "") == 3
        assert _edit_distance("", "abc") == 3


class TestCER:
    def test_perfect_prediction(self):
        cer = compute_cer(["hello"], ["hello"])
        assert cer == 0.0

    def test_completely_wrong(self):
        cer = compute_cer(["xyz"], ["abc"])
        assert cer > 0.0

    def test_empty_prediction(self):
        cer = compute_cer([""], ["hello"])
        assert cer > 0.0

    def test_multiple_samples(self):
        preds = ["hello", "world"]
        refs = ["hello", "world"]
        cer = compute_cer(preds, refs)
        assert cer == 0.0


class TestWER:
    def test_perfect_prediction(self):
        wer = compute_wer(["hello world"], ["hello world"])
        assert wer == 0.0

    def test_wrong_word(self):
        wer = compute_wer(["hello earth"], ["hello world"])
        assert wer > 0.0


class TestExactMatchAccuracy:
    def test_all_correct(self):
        acc = exact_match_accuracy(["a", "b", "c"], ["a", "b", "c"])
        assert acc == 1.0

    def test_none_correct(self):
        acc = exact_match_accuracy(["x", "y", "z"], ["a", "b", "c"])
        assert acc == 0.0

    def test_partial(self):
        acc = exact_match_accuracy(["a", "x"], ["a", "b"])
        assert acc == 0.5

    def test_empty_list(self):
        acc = exact_match_accuracy([], [])
        assert acc == 0.0
