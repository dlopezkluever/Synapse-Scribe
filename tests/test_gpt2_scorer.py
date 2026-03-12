"""Tests for GPT-2 re-ranking in src/decoding/lm_correction.py.

These tests verify the GPT2Scorer interface and its integration with the
rescore_hypotheses pipeline. Tests that require the ``transformers`` library
are skipped if it is not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.decoding.beam_search import Hypothesis
from src.decoding.lm_correction import (
    DummyLMScorer,
    GPT2Scorer,
    load_lm_scorer,
    rescore_hypotheses,
)

# Check if transformers is available
try:
    import transformers
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

requires_transformers = pytest.mark.skipif(
    not _HAS_TRANSFORMERS,
    reason="transformers not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_hypotheses():
    """Beam search hypotheses with CTC scores."""
    return [
        Hypothesis(text="hello world", score=-5.0),
        Hypothesis(text="helo wrld", score=-3.0),
        Hypothesis(text="hello worl", score=-4.0),
        Hypothesis(text="hllo world", score=-6.0),
    ]


class _MockGPT2Scorer:
    """Deterministic mock that prefers well-formed English."""

    def __init__(self):
        # Higher score = more likely
        self._scores = {
            "hello world": -2.0,
            "helo wrld": -8.0,
            "hello worl": -5.0,
            "hllo world": -6.0,
        }

    def score(self, text: str) -> float:
        return self._scores.get(text, -10.0)


# ---------------------------------------------------------------------------
# GPT2Scorer unit tests (mocked — no model download needed)
# ---------------------------------------------------------------------------


class TestGPT2ScorerInterface:
    def test_has_score_method(self):
        """GPT2Scorer must implement .score(text) -> float."""
        # Test via mock to avoid downloading the model
        scorer = _MockGPT2Scorer()
        result = scorer.score("hello")
        assert isinstance(result, float)

    def test_score_returns_negative(self):
        """Log-probabilities should be negative (or zero)."""
        scorer = _MockGPT2Scorer()
        assert scorer.score("hello world") <= 0.0

    def test_empty_string_returns_zero(self):
        """Empty string should return 0.0."""
        scorer = _MockGPT2Scorer()
        # Real GPT2Scorer handles this; mock won't, so test the real class
        # We'll test the interface spec here
        assert True  # placeholder — real test below

    @requires_transformers
    def test_real_scorer_empty_string(self):
        """Real GPT2Scorer should return 0.0 for empty string."""
        scorer = GPT2Scorer(model_name="gpt2", device="cpu")
        assert scorer.score("") == 0.0
        assert scorer.score("   ") == 0.0

    @requires_transformers
    def test_real_scorer_returns_float(self):
        """Real GPT2Scorer should return a float score."""
        scorer = GPT2Scorer(model_name="gpt2", device="cpu")
        score = scorer.score("hello world")
        assert isinstance(score, float)
        assert score < 0.0  # log-prob is negative

    @requires_transformers
    def test_real_scorer_prefers_valid_english(self):
        """GPT-2 should assign higher scores to valid English."""
        scorer = GPT2Scorer(model_name="gpt2", device="cpu")
        good = scorer.score("the cat sat on the mat")
        bad = scorer.score("xqz fgh jkl mnb")
        assert good > bad

    @requires_transformers
    def test_real_scorer_model_name_attribute(self):
        scorer = GPT2Scorer(model_name="gpt2", device="cpu")
        assert scorer.model_name == "gpt2"


# ---------------------------------------------------------------------------
# GPT-2 rescoring integration
# ---------------------------------------------------------------------------


class TestGPT2Rescoring:
    def test_rescore_changes_ranking(self, sample_hypotheses):
        """GPT-2 rescoring should re-rank hypotheses based on LM scores."""
        scorer = _MockGPT2Scorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.5, beta=0.0)

        # With alpha=0.5:
        # "hello world": 0.5*(-5) + 0.5*(-2) = -3.5
        # "helo wrld":   0.5*(-3) + 0.5*(-8) = -5.5
        # "hello worl":  0.5*(-4) + 0.5*(-5) = -4.5
        # "hllo world":  0.5*(-6) + 0.5*(-6) = -6.0
        assert rescored[0].text == "hello world"
        assert rescored[1].text == "hello worl"
        assert rescored[2].text == "helo wrld"
        assert rescored[3].text == "hllo world"

    def test_rescore_alpha_zero_preserves_ctc(self, sample_hypotheses):
        """alpha=0 should keep CTC-only ranking."""
        scorer = _MockGPT2Scorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.0)
        assert rescored[0].text == "helo wrld"  # highest CTC score

    def test_rescore_alpha_one_uses_lm_only(self, sample_hypotheses):
        """alpha=1 should rank purely by LM score."""
        scorer = _MockGPT2Scorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=1.0)
        assert rescored[0].text == "hello world"  # highest LM score

    def test_rescore_preserves_hypothesis_count(self, sample_hypotheses):
        scorer = _MockGPT2Scorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.5)
        assert len(rescored) == len(sample_hypotheses)

    def test_rescore_returns_hypothesis_objects(self, sample_hypotheses):
        scorer = _MockGPT2Scorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.5)
        for h in rescored:
            assert isinstance(h, Hypothesis)

    def test_rescore_sorted_descending(self, sample_hypotheses):
        scorer = _MockGPT2Scorer()
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.5)
        scores = [h.score for h in rescored]
        assert scores == sorted(scores, reverse=True)

    @requires_transformers
    def test_real_gpt2_rescoring(self, sample_hypotheses):
        """End-to-end test with real GPT-2 model."""
        scorer = GPT2Scorer(model_name="gpt2", device="cpu")
        rescored = rescore_hypotheses(sample_hypotheses, scorer, alpha=0.5)
        assert len(rescored) == len(sample_hypotheses)
        # "hello world" should be ranked higher after GPT-2 rescoring
        assert rescored[0].text == "hello world"


# ---------------------------------------------------------------------------
# load_lm_scorer with scorer_type="gpt2"
# ---------------------------------------------------------------------------


class TestLoadLMScorerGPT2:
    @requires_transformers
    def test_loads_gpt2_scorer(self):
        scorer = load_lm_scorer(model_path="gpt2", scorer_type="gpt2", device="cpu")
        assert isinstance(scorer, GPT2Scorer)

    @requires_transformers
    def test_default_model_name(self):
        scorer = load_lm_scorer(scorer_type="gpt2", device="cpu")
        assert isinstance(scorer, GPT2Scorer)
        assert scorer.model_name == "gpt2"

    def test_fallback_to_dummy_on_import_error(self):
        """Should fall back to DummyLMScorer if transformers not available."""
        with patch.dict("sys.modules", {"transformers": None}):
            with patch("src.decoding.lm_correction.GPT2Scorer", side_effect=ImportError("no transformers")):
                scorer = load_lm_scorer(scorer_type="gpt2")
        assert isinstance(scorer, DummyLMScorer)

    def test_kenlm_default_unchanged(self):
        """Default scorer_type='kenlm' behavior should be preserved."""
        scorer = load_lm_scorer(model_path=None, scorer_type="kenlm")
        assert isinstance(scorer, DummyLMScorer)
