"""Language model correction via shallow fusion.

Supports KenLM character-level n-gram models, a pure-Python character
n-gram scorer, and GPT-2 neural LM for re-scoring beam search hypotheses.
Falls back gracefully when optional dependencies are not installed.
"""

from __future__ import annotations

import json
import logging
import math
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


class CharNgramScorer:
    """Pure-Python character n-gram LM scorer.

    Loads a JSON model built by ``scripts/build_char_lm.py`` and scores
    text using smoothed n-gram probabilities with backoff.

    Args:
        model_path: Path to the JSON language model file.
    """

    # BOS/EOS must match build_char_lm.py
    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        with open(self.model_path, "r", encoding="utf-8") as f:
            lm = json.load(f)

        self.order: int = lm["order"]
        self.smoothing: float = lm["smoothing"]
        self.vocab: list[str] = lm["vocab"]
        self.vocab_size: int = len(self.vocab)
        self.counts: dict[str, dict[str, int]] = lm["counts"]
        self.totals: dict[str, int] = lm["totals"]
        logger.info(
            "Loaded CharNgramScorer: order=%d, vocab=%d, contexts=%d",
            self.order, self.vocab_size, len(self.counts),
        )

    def _log_prob(self, char: str, context: tuple[str, ...]) -> float:
        """Compute log P(char | context) with backoff."""
        for n in range(len(context), -1, -1):
            ctx = context[len(context) - n :]
            ctx_key = "|".join(ctx)
            if ctx_key in self.counts:
                count = self.counts[ctx_key].get(char, 0)
                total = self.totals[ctx_key]
                # Lidstone smoothing
                prob = (count + self.smoothing) / (
                    total + self.smoothing * self.vocab_size
                )
                return math.log(prob)
        # Uniform fallback
        return math.log(1.0 / self.vocab_size)

    def score(self, text: str) -> float:
        """Score a text string. Returns log-probability (higher is better)."""
        if not text:
            return 0.0

        text = text.lower()
        padded = [self.BOS] * (self.order - 1) + list(text) + [self.EOS]

        log_prob = 0.0
        for i in range(self.order - 1, len(padded)):
            context = tuple(padded[i - self.order + 1 : i])
            char = padded[i]
            log_prob += self._log_prob(char, context)

        return log_prob


class GPT2Scorer:
    """Score text using a pre-trained GPT-2 model from HuggingFace.

    Computes the mean per-token log-probability of the input text, which
    provides a length-normalized language model score suitable for
    re-ranking CTC beam search hypotheses.

    Args:
        model_name: HuggingFace model identifier (default: ``"gpt2"``).
        device: PyTorch device string (default: auto-detect).
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        except ImportError:
            raise ImportError(
                "transformers is not installed. Install via: "
                "pip install transformers"
            )

        self.model_name = model_name
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info("Loading GPT-2 model '%s' on %s ...", model_name, self._device)
        self._tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self._model = GPT2LMHeadModel.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        logger.info("GPT-2 model loaded (%d parameters)",
                     sum(p.numel() for p in self._model.parameters()))

    def score(self, text: str) -> float:
        """Compute mean per-token log-probability of *text*.

        Returns 0.0 for empty strings. The score is a negative number
        (higher / closer to zero is better), consistent with the
        KenLMScorer interface.
        """
        if not text or not text.strip():
            return 0.0

        import torch

        encodings = self._tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self._device)

        if input_ids.shape[1] == 0:
            return 0.0

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            # outputs.loss is the mean cross-entropy over all tokens
            # Negate because CE loss = -log_prob
            neg_log_likelihood = outputs.loss.item()

        return -neg_log_likelihood  # return log-prob (higher is better)


def rescore_hypotheses(
    hypotheses: list[Hypothesis],
    lm_scorer: KenLMScorer | DummyLMScorer | GPT2Scorer,
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


def load_lm_scorer(
    model_path: Optional[str | Path] = None,
    scorer_type: str = "kenlm",
    device: Optional[str] = None,
) -> KenLMScorer | DummyLMScorer | GPT2Scorer | CharNgramScorer:
    """Load a language model scorer, falling back to DummyLMScorer.

    Args:
        model_path: Path to KenLM model, char n-gram JSON, or HuggingFace
            model name for GPT-2. If None, returns DummyLMScorer.
        scorer_type: ``"kenlm"``, ``"char_ngram"``, or ``"gpt2"``.
        device: Device for GPT-2 (ignored for other scorers).

    Returns:
        A scorer object with a ``.score(text)`` method.
    """
    if scorer_type == "gpt2":
        model_name = str(model_path) if model_path else "gpt2"
        try:
            return GPT2Scorer(model_name=model_name, device=device)
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning("Could not load GPT-2: %s; using DummyLMScorer", e)
            return DummyLMScorer()

    if scorer_type == "char_ngram":
        if model_path is None:
            logger.warning("No LM path for char_ngram; using DummyLMScorer")
            return DummyLMScorer()
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning("Char n-gram LM not found at %s; using DummyLMScorer", model_path)
            return DummyLMScorer()
        try:
            return CharNgramScorer(model_path)
        except Exception as e:
            logger.warning("Could not load char n-gram LM: %s; using DummyLMScorer", e)
            return DummyLMScorer()

    # Default: KenLM
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
