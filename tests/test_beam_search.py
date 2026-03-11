"""Tests for src/decoding/beam_search.py — beam search CTC decoding."""

import numpy as np
import pytest
import torch

from src.decoding.beam_search import beam_search_decode, beam_search_decode_batch, Hypothesis
from src.data.dataset import BLANK_IDX, CHAR_TO_IDX


def _make_logits_for_sequence(char_sequence: list[int | str], T: int = 20, C: int = 28) -> np.ndarray:
    """Create logits that produce a specific character sequence via argmax."""
    logits = np.full((T, C), -10.0, dtype=np.float32)
    for t, ch in enumerate(char_sequence):
        if t >= T:
            break
        if isinstance(ch, str):
            if ch == "blank" or ch == "_":
                idx = BLANK_IDX
            else:
                idx = CHAR_TO_IDX.get(ch, BLANK_IDX)
        else:
            idx = ch
        logits[t, idx] = 10.0
    for t in range(len(char_sequence), T):
        logits[t, BLANK_IDX] = 10.0
    return logits


class TestBeamSearchDecode:
    def test_simple_hello(self):
        """Beam search should decode clear logits correctly."""
        seq = ["h", "_", "e", "_", "l", "_", "o"]
        logits = _make_logits_for_sequence(seq, T=15)
        results = beam_search_decode(logits, beam_width=10, top_k=3)
        assert len(results) > 0
        assert results[0].text == "helo"

    def test_repeated_chars_with_blank(self):
        """h e l _ l o → 'hello'."""
        seq = ["h", "e", "l", "_", "l", "o"]
        logits = _make_logits_for_sequence(seq, T=15)
        results = beam_search_decode(logits, beam_width=10, top_k=3)
        assert results[0].text == "hello"

    def test_all_blanks(self):
        """All-blank output should produce empty string as top hypothesis."""
        logits = np.full((20, 28), -10.0, dtype=np.float32)
        logits[:, BLANK_IDX] = 10.0
        results = beam_search_decode(logits, beam_width=10, top_k=3)
        assert results[0].text == ""

    def test_returns_hypothesis_objects(self):
        seq = ["a", "_", "b"]
        logits = _make_logits_for_sequence(seq, T=10)
        results = beam_search_decode(logits, beam_width=10, top_k=5)
        for h in results:
            assert isinstance(h, Hypothesis)
            assert isinstance(h.text, str)
            assert isinstance(h.score, float)

    def test_top_k_limit(self):
        seq = ["a"]
        logits = _make_logits_for_sequence(seq, T=5)
        results = beam_search_decode(logits, beam_width=10, top_k=3)
        assert len(results) <= 3

    def test_results_sorted_by_score(self):
        seq = ["h", "i"]
        logits = _make_logits_for_sequence(seq, T=10)
        results = beam_search_decode(logits, beam_width=20, top_k=5)
        scores = [h.score for h in results]
        assert scores == sorted(scores, reverse=True)

    def test_beam_search_at_least_as_good_as_greedy(self):
        """Beam search should produce equal or better results than greedy on clear logits."""
        from src.decoding.greedy import greedy_decode
        seq = ["c", "_", "a", "_", "t"]
        logits = _make_logits_for_sequence(seq, T=15)
        greedy_result = greedy_decode(logits)
        beam_results = beam_search_decode(logits, beam_width=20, top_k=1)
        # Top beam result should match greedy on clear logits
        assert beam_results[0].text == greedy_result

    def test_accepts_torch_tensor(self):
        seq = ["a", "_", "b"]
        logits = torch.from_numpy(_make_logits_for_sequence(seq, T=10))
        results = beam_search_decode(logits, beam_width=10, top_k=3)
        assert results[0].text == "ab"

    def test_3d_input(self):
        seq = ["x", "_", "y"]
        logits = _make_logits_for_sequence(seq, T=10)
        logits_3d = logits[np.newaxis, :, :]
        results = beam_search_decode(logits_3d, beam_width=10, top_k=3)
        assert results[0].text == "xy"

    def test_with_space(self):
        seq = ["h", "i", "_", " ", "_", "t"]
        logits = _make_logits_for_sequence(seq, T=15)
        results = beam_search_decode(logits, beam_width=10, top_k=3)
        assert results[0].text == "hi t"


class TestBeamSearchBatch:
    def test_batch_decoding(self):
        seq1 = ["h", "i"]
        seq2 = ["b", "y", "e"]
        logits1 = _make_logits_for_sequence(seq1, T=10)
        logits2 = _make_logits_for_sequence(seq2, T=10)
        batch = np.stack([logits1, logits2])
        results = beam_search_decode_batch(batch, beam_width=10, top_k=3)
        assert len(results) == 2
        assert results[0][0].text == "hi"
        assert results[1][0].text == "bye"
