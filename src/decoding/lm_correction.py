"""Language model correction via shallow fusion.

Supports KenLM character-level n-gram models for re-scoring beam search
hypotheses. Falls back gracefully when KenLM is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.decoding.beam_search import Hypothesis

logger = logging.getLogger(__name__)


class KenLMScorer:
    """Wrapper around a KenLM model for character-level scoring.

    Args:
        model_path: Path to a trained KenLM binary (.arpa or .binary).
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        try:
            import kenlm
            self.model = kenlm.Model(str(self.model_path))
            logger.info("Loaded KenLM model from %s", self.model_path)
        except ImportError:
            raise ImportError(
                "kenlm is not installed. Install via: pip install kenlm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load KenLM model: {e}")

    def score(self, text: str) -> float:
        """Score a text string using the language model.

        KenLM scores are log10 probabilities. We convert to natural log.

        Args:
            text: Input text to score.

        Returns:
            Log-probability score (natural log, higher is better).
        """
        import math
        # KenLM expects space-separated tokens; for character-level,
        # each character is a token
        char_tokens = " ".join(list(text)) if text else ""
        log10_score = self.model.score(char_tokens, bos=True, eos=True)
        return log10_score * math.log(10)  # convert to ln


class DummyLMScorer:
    """No-op LM scorer that returns 0.0 for all inputs.

    Used as a placeholder when no language model is available.
    """

    def score(self, text: str) -> float:
        return 0.0


def rescore_hypotheses(
    hypotheses: list[Hypothesis],
    lm_scorer: KenLMScorer | DummyLMScorer,
    alpha: float = 0.5,
    beta: float = 0.0,
) -> list[Hypothesis]:
    """Re-score beam search hypotheses with shallow fusion.

    score = (1 - alpha) * CTC_score + alpha * LM_score + beta * len(text)

    Args:
        hypotheses: List of Hypothesis from beam search.
        lm_scorer: Language model scorer (KenLM or dummy).
        alpha: LM interpolation weight (0 = CTC only, 1 = LM only).
        beta: Length bonus per character.

    Returns:
        Re-scored and re-sorted list of Hypothesis.
    """
    rescored = []
    for h in hypotheses:
        lm_score = lm_scorer.score(h.text)
        new_score = (1 - alpha) * h.score + alpha * lm_score + beta * len(h.text)
        rescored.append(Hypothesis(
            text=h.text,
            score=new_score,
            char_probs=h.char_probs,
        ))

    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored


def load_lm_scorer(model_path: Optional[str | Path] = None) -> KenLMScorer | DummyLMScorer:
    """Load a language model scorer, falling back to DummyLMScorer.

    Args:
        model_path: Path to KenLM model. If None, returns DummyLMScorer.

    Returns:
        A scorer object with a .score(text) method.
    """
    if model_path is None:
        logger.info("No LM model path provided; using DummyLMScorer")
        return DummyLMScorer()

    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning("LM model not found at %s; using DummyLMScorer", model_path)
        return DummyLMScorer()

    try:
        return KenLMScorer(model_path)
    except (ImportError, RuntimeError) as e:
        logger.warning("Could not load KenLM: %s; using DummyLMScorer", e)
        return DummyLMScorer()
