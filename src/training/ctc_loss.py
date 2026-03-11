"""CTC loss wrapper for neural decoding.

Handles log-softmax and time-first transpose required by PyTorch's CTCLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """Wrapper around torch.nn.CTCLoss with log-softmax and transpose handling.

    PyTorch's CTCLoss expects:
        - log_probs: [T, B, C] (time-first)
        - targets: [sum(target_lengths)] (concatenated)
        - input_lengths: [B]
        - target_lengths: [B]
    """

    def __init__(self, blank: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=zero_infinity)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            logits: [B, T, C] raw logits from model.
            targets: [sum(L_i)] concatenated target indices.
            input_lengths: [B] actual input sequence lengths.
            target_lengths: [B] target sequence lengths.

        Returns:
            Scalar CTC loss.
        """
        # Log-softmax over class dimension
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose to [T, B, C] for CTCLoss
        log_probs = log_probs.transpose(0, 1)

        # Clamp input_lengths to not exceed T
        T = log_probs.size(0)
        input_lengths = input_lengths.clamp(max=T)

        return self.ctc(log_probs, targets, input_lengths, target_lengths)
