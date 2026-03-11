"""Greedy CTC decoding.

Implements argmax -> collapse repeats -> remove blanks.
"""

from __future__ import annotations

import torch
import numpy as np

from src.data.dataset import BLANK_IDX, IDX_TO_CHAR


def greedy_decode(logits: torch.Tensor | np.ndarray) -> str:
    """Greedy CTC decoding: argmax per timestep, collapse repeats, remove blanks.

    Args:
        logits: [T, C] or [B, T, C] logits. If 3-D, decodes the first sample.

    Returns:
        Decoded string.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    if logits.ndim == 3:
        logits = logits[0]  # take first sample

    # Argmax per timestep
    best = np.argmax(logits, axis=-1)  # [T]

    # Collapse repeats and remove blanks
    chars = []
    prev = -1
    for idx in best:
        if idx != prev:
            if idx != BLANK_IDX:
                chars.append(IDX_TO_CHAR.get(int(idx), ""))
        prev = idx

    return "".join(chars)


def greedy_decode_batch(logits: torch.Tensor | np.ndarray) -> list[str]:
    """Greedy decode an entire batch.

    Args:
        logits: [B, T, C] logits tensor.

    Returns:
        List of decoded strings, one per batch element.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    results = []
    for i in range(logits.shape[0]):
        results.append(greedy_decode(logits[i]))
    return results
