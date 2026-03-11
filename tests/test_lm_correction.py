"""Tests for src/decoding/lm_correction.py — language model correction."""

import pytest

from src.decoding.beam_search import Hypothesis
from src.decoding.lm_correction import (
    DummyLMScorer,
    rescore_hypotheses,
    load_lm_scorer,
)


@pytest.fixture
def sample_hypotheses():
    """Sample hypotheses to test rescoring."""
    return [
        Hypothesis(text="hello", score=-5.0),
        Hypothesis(text="helo", score=-3.0),
        Hypothesis(text="hlelo", score=-7.0),
    ]


class TestDummyLMScorer:
    def test_always_returns_zero(self):
        scorer = DummyLMScorer()
        assert scorer.score("hello") == 0.0
        assert scorer.score("") == 0.0
        assert scorer.score("any text") == 0.0


class TestRescoreHypotheses:
    def test_dummy_scorer_preserves_ctc_ranking(self, sample_hypotheses):
        """With DummyLMScorer (score=0) and alpha=0, ranking stays the same."""
        scorer = DummyLMScorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.0, beta=0.0)
        assert rescored[0].text == "helo"  # highest CTC score (-3.0)
        assert rescored[1].text == "hello"
        assert rescored[2].text == "hlelo"

    def test_length_bonus(self, sample_hypotheses):
        """Beta > 0 should favor longer hypotheses."""
        scorer = DummyLMScorer()
        # With large enough beta, longer texts get boosted
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.0, beta=1.0)
        # "hlelo" (5 chars, score -7+5=-2) > "hello" (5 chars, score -5+5=0)
        # Actually: helo=-3+4=1, hello=-5+5=0, hlelo=-7+5=-2
        assert rescored[0].text == "helo"
        assert rescored[1].text == "hello"

    def test_returns_hypothesis_objects(self, sample_hypotheses):
        scorer = DummyLMScorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.0)
        for h in rescored:
            assert isinstance(h, Hypothesis)

    def test_sorted_by_score(self, sample_hypotheses):
        scorer = DummyLMScorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.0)
        scores = [h.score for h in rescored]
        assert scores == sorted(scores, reverse=True)


class TestLoadLMScorer:
    def test_none_path_returns_dummy(self):
        scorer = load_lm_scorer(None)
        assert isinstance(scorer, DummyLMScorer)

    def test_nonexistent_path_returns_dummy(self, tmp_path):
        scorer = load_lm_scorer(tmp_path / "nonexistent.arpa")
        assert isinstance(scorer, DummyLMScorer)
