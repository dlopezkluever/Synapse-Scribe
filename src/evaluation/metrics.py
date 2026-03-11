"""Evaluation metrics: CER, WER, and exact match accuracy.

Uses the jiwer library for CER/WER computation.
"""

from __future__ import annotations

from typing import Optional


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Compute Character Error Rate using jiwer.

    Args:
        predictions: List of predicted strings.
        references: List of ground truth strings.

    Returns:
        CER as a float between 0 and 1+ (can exceed 1 with many insertions).
    """
    try:
        import jiwer
        # jiwer.cer expects non-empty strings; handle edge cases
        preds = [p if p else " " for p in predictions]
        refs = [r if r else " " for r in references]
        return jiwer.cer(refs, preds)
    except ImportError:
        # Fallback: simple edit distance based CER
        return _manual_cer(predictions, references)


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Compute Word Error Rate using jiwer.

    Args:
        predictions: List of predicted strings.
        references: List of ground truth strings.

    Returns:
        WER as a float.
    """
    try:
        import jiwer
        preds = [p if p else " " for p in predictions]
        refs = [r if r else " " for r in references]
        return jiwer.wer(refs, preds)
    except ImportError:
        return _manual_wer(predictions, references)


def exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    """Fraction of trials with CER = 0 (exact match).

    Args:
        predictions: List of predicted strings.
        references: List of ground truth strings.

    Returns:
        Fraction in [0, 1].
    """
    if not predictions:
        return 0.0
    exact = sum(1 for p, r in zip(predictions, references) if p == r)
    return exact / len(predictions)


# ---------------------------------------------------------------------------
# Fallback implementations (no jiwer dependency)
# ---------------------------------------------------------------------------

def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def _manual_cer(predictions: list[str], references: list[str]) -> float:
    """Manual CER: sum of char edit distances / sum of reference lengths."""
    total_edits = 0
    total_len = 0
    for p, r in zip(predictions, references):
        total_edits += _edit_distance(p, r)
        total_len += max(len(r), 1)
    return total_edits / max(total_len, 1)


def _manual_wer(predictions: list[str], references: list[str]) -> float:
    """Manual WER: sum of word edit distances / sum of reference word counts."""
    total_edits = 0
    total_words = 0
    for p, r in zip(predictions, references):
        p_words = p.split()
        r_words = r.split()
        total_edits += _edit_distance(" ".join(p_words), " ".join(r_words))
        total_words += max(len(r_words), 1)
    return total_edits / max(total_words, 1)
