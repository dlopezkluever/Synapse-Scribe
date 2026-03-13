"""Willett-style GRU Decoder (Model A).

Architecture: Linear(192->256) + ReLU + Dropout -> 3-layer unidirectional GRU(512) -> Linear -> logits.
This replicates the architecture from the Willett handwriting BCI paper.

Optionally includes a CNN temporal downsampler (4x) before the GRU to reduce
blank-heavy CTC alignments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseDecoder


class GRUDecoder(BaseDecoder):
    """Willett-style GRU decoder for neural handwriting BCI.

    Args:
        n_channels: Number of input channels (default 192).
        n_classes: Number of output classes (default 28).
        proj_dim: Projection dimension (default 256).
        hidden_size: GRU hidden size (default 512).
        n_layers: Number of GRU layers (default 3).
        dropout: Dropout probability (default 0.3).
        use_downsample: If True, add a 4x temporal downsampler before the GRU.
    """

    def __init__(
        self,
        n_channels: int = 192,
        n_classes: int = 28,
        proj_dim: int = 256,
        hidden_size: int = 512,
        n_layers: int = 3,
        dropout: float = 0.3,
        use_downsample: bool = True,
    ):
        super().__init__(n_channels, n_classes)
        self._use_downsample = use_downsample

        if use_downsample:
            # CNN front-end for 4x temporal reduction (stride=2 twice)
            self.downsample = nn.Sequential(
                nn.Conv1d(n_channels, 256, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
            )
            gru_input_size = 256
        else:
            # Original: linear projection (no downsampling)
            self.input_proj = nn.Sequential(
                nn.Linear(n_channels, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            gru_input_size = proj_dim

        # GRU
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, n_classes)

    @property
    def downsample_factor(self) -> int:
        return 4 if self._use_downsample else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, T, C] -> [B, T', n_classes]."""
        if self._use_downsample:
            x = x.transpose(1, 2)       # [B, T, C] -> [B, C, T]
            x = self.downsample(x)      # [B, 256, T//4]
            x = x.transpose(1, 2)       # [B, T//4, 256]
        else:
            x = self.input_proj(x)      # [B, T, proj_dim]

        x, _ = self.gru(x)             # [B, T', hidden_size]
        logits = self.output_proj(x)    # [B, T', n_classes]
        return logits
