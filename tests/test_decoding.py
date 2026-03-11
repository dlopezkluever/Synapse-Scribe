"""Tests for src/decoding/greedy.py — greedy CTC decoding."""

import numpy as np
import pytest
import torch

from src.decoding.greedy import greedy_decode, greedy_decode_batch
from src.data.dataset import BLANK_IDX, CHAR_TO_IDX


def _make_logits_for_sequence(char_sequence: list[int | str], T: int = 20, C: int = 28) -> np.ndarray:
    """Create logits that produce a specific character sequence via argmax.

    Args:
        char_sequence: List of class indices or chars like 'h', 'blank', etc.
        T: Total number of timesteps.
        C: Number of classes.

    Returns:
        Logits array [T, C].
    """
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
    # Fill remaining timesteps with blank
    for t in range(len(char_sequence), T):
        logits[t, BLANK_IDX] = 10.0
    return logits


class TestGreedyDecode:
    def test_simple_hello(self):
        """h h _ e l l _ o → 'helo' (CTC collapse: hh→h, ll→l, remove blanks)."""
        seq = ["h", "h", "_", "e", "l", "l", "_", "o"]
        logits = _make_logits_for_sequence(seq, T=20)
        result = greedy_decode(logits)
        assert result == "helo"

    def test_repeated_chars_with_blank(self):
        """h e l _ l o → 'hello' (blank separates the two l's)."""
        seq = ["h", "e", "l", "_", "l", "o"]
        logits = _make_logits_for_sequence(seq, T=20)
        result = greedy_decode(logits)
        assert result == "hello"

    def test_all_blanks(self):
        """All-blank output should produce empty string."""
        logits = np.full((20, 28), -10.0, dtype=np.float32)
        logits[:, BLANK_IDX] = 10.0
        result = greedy_decode(logits)
        assert result == ""

    def test_with_space(self):
        """h i _ space _ t → 'hi t'."""
        seq = ["h", "i", "_", " ", "_", "t"]
        logits = _make_logits_for_sequence(seq, T=20)
        result = greedy_decode(logits)
        assert result == "hi t"

    def test_single_char(self):
        """Single character repeated should decode to that char."""
        seq = ["a", "a", "a"]
        logits = _make_logits_for_sequence(seq, T=10)
        result = greedy_decode(logits)
        assert result == "a"

    def test_accepts_torch_tensor(self):
        seq = ["a", "_", "b"]
        logits = _make_logits_for_sequence(seq, T=10)
        logits_tensor = torch.from_numpy(logits)
        result = greedy_decode(logits_tensor)
        assert result == "ab"

    def test_3d_input(self):
        """3-D input should decode the first sample."""
        seq = ["c", "_", "a", "t"]
        logits = _make_logits_for_sequence(seq, T=10)
        logits_3d = logits[np.newaxis, :, :]  # [1, T, C]
        result = greedy_decode(logits_3d)
        assert result == "cat"


class TestGreedyDecodeBatch:
    def test_batch_decoding(self):
        seq1 = ["h", "i"]
        seq2 = ["b", "y", "e"]
        logits1 = _make_logits_for_sequence(seq1, T=10)
        logits2 = _make_logits_for_sequence(seq2, T=10)
        batch = np.stack([logits1, logits2])  # [2, 10, 28]
        results = greedy_decode_batch(batch)
        assert results == ["hi", "bye"]

    def test_batch_with_torch(self):
        seq1 = ["a"]
        seq2 = ["z"]
        logits1 = _make_logits_for_sequence(seq1, T=5)
        logits2 = _make_logits_for_sequence(seq2, T=5)
        batch = torch.from_numpy(np.stack([logits1, logits2]))
        results = greedy_decode_batch(batch)
        assert results == ["a", "z"]
