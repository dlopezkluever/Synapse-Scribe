"""Beam search CTC decoding.

Implements prefix beam search for CTC-trained models. Returns top-k
hypotheses ranked by log-probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.data.dataset import BLANK_IDX, IDX_TO_CHAR, VOCAB_SIZE


@dataclass
class Hypothesis:
    """A single beam search hypothesis."""
    text: str
    score: float
    char_probs: list[float] = field(default_factory=list)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along last axis."""
    max_val = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


def beam_search_decode(
    logits: torch.Tensor | np.ndarray,
    beam_width: int = 100,
    top_k: int = 5,
    lm_scorer: Optional[object] = None,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> list[Hypothesis]:
    """CTC prefix beam search decoding.

    Args:
        logits: [T, C] or [B, T, C] logits. If 3-D, decodes the first sample.
        beam_width: Number of beams to maintain at each timestep.
        top_k: Number of top hypotheses to return.
        lm_scorer: Optional language model scorer with a `score(text) -> float`
            method. If provided, scores are combined via shallow fusion.
        alpha: LM weight for shallow fusion.
        beta: Word insertion bonus (per character).

    Returns:
        List of top-k Hypothesis objects sorted by score (highest first).
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    if logits.ndim == 3:
        logits = logits[0]

    T, C = logits.shape
    log_probs = _log_softmax(logits)  # [T, C]

    # Each beam entry: (prefix_tuple, (prob_blank, prob_non_blank))
    # prefix_tuple stores character indices (not including blanks)
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, float("-inf")),  # (log_prob_blank, log_prob_non_blank)
    }

    for t in range(T):
        new_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        # Prune to beam_width before extending
        scored = []
        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            scored.append((prefix, pb, pnb, total))
        scored.sort(key=lambda x: x[3], reverse=True)
        scored = scored[:beam_width]

        for prefix, pb, pnb, _ in scored:
            # Total prob of this prefix ending at t-1
            p_prefix = np.logaddexp(pb, pnb)

            # --- Extend with blank ---
            blank_lp = log_probs[t, BLANK_IDX]
            new_pb = p_prefix + blank_lp

            if prefix in new_beams:
                old_pb, old_pnb = new_beams[prefix]
                new_beams[prefix] = (np.logaddexp(old_pb, new_pb), old_pnb)
            else:
                new_beams[prefix] = (new_pb, float("-inf"))

            # --- Extend with each non-blank character ---
            for c in range(1, C):
                c_lp = log_probs[t, c]
                new_prefix = prefix + (c,)

                if prefix and prefix[-1] == c:
                    # Same char as end of prefix: only extend from blank ending
                    new_pnb = pb + c_lp
                    # Also allow the prefix to continue via non-blank
                    if prefix in new_beams:
                        old_pb2, old_pnb2 = new_beams[prefix]
                        new_beams[prefix] = (
                            old_pb2,
                            np.logaddexp(old_pnb2, pnb + c_lp),
                        )
                    else:
                        new_beams[prefix] = (float("-inf"), pnb + c_lp)
                else:
                    new_pnb = p_prefix + c_lp

                if new_prefix in new_beams:
                    old_pb3, old_pnb3 = new_beams[new_prefix]
                    new_beams[new_prefix] = (old_pb3, np.logaddexp(old_pnb3, new_pnb))
                else:
                    new_beams[new_prefix] = (float("-inf"), new_pnb)

        beams = new_beams

    # Final scoring
    results = []
    for prefix, (pb, pnb) in beams.items():
        ctc_score = np.logaddexp(pb, pnb)

        text = "".join(IDX_TO_CHAR.get(idx, "") for idx in prefix)

        total_score = ctc_score

        # Apply LM scoring if available
        if lm_scorer is not None and alpha > 0:
            lm_score = lm_scorer.score(text)
            total_score = (1 - alpha) * ctc_score + alpha * lm_score

        # Word insertion bonus
        if beta != 0:
            total_score += beta * len(text)

        results.append(Hypothesis(
            text=text,
            score=float(total_score),
            char_probs=[],
        ))

    results.sort(key=lambda h: h.score, reverse=True)
    return results[:top_k]


def beam_search_decode_batch(
    logits: torch.Tensor | np.ndarray,
    beam_width: int = 100,
    top_k: int = 5,
    lm_scorer: Optional[object] = None,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> list[list[Hypothesis]]:
    """Beam search decode an entire batch.

    Args:
        logits: [B, T, C] logits tensor.
        beam_width: Number of beams.
        top_k: Number of top hypotheses per sample.
        lm_scorer: Optional LM scorer.
        alpha: LM weight.
        beta: Word insertion bonus.

    Returns:
        List of lists of Hypothesis, one list per batch element.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    results = []
    for i in range(logits.shape[0]):
        hypotheses = beam_search_decode(
            logits[i], beam_width=beam_width, top_k=top_k,
            lm_scorer=lm_scorer, alpha=alpha, beta=beta,
        )
        results.append(hypotheses)
    return results
