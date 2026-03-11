"""Abstract base class for all decoder models.

All models follow the contract: forward(x: [B, T, C]) -> logits: [B, T, n_classes].
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDecoder(ABC, nn.Module):
    """Abstract base class for neural signal decoders."""

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, C] where
                B = batch size, T = timesteps, C = input channels.

        Returns:
            Logits tensor [B, T', n_classes] where T' may differ from T
            for models with temporal downsampling.
        """
        ...

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
