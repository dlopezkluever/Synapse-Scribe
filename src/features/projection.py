"""Linear Projection + Positional Encoding (Pathway B).

Projects each timestep's channel vector into d_model space and adds
sinusoidal positional encodings. This is the Transformer's input embedding layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding from "Attention Is All You Need".

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length (default 4096).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but moves with .to(device))
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model] with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LinearProjection(nn.Module):
    """Linear projection with sinusoidal positional encoding.

    Projects channel vectors to d_model and adds positional encodings.

    Args:
        n_channels: Number of input channels.
        d_model: Target embedding dimension (default 512).
        max_len: Maximum sequence length for positional encoding (default 4096).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 512,
        max_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(n_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] input tensor.

        Returns:
            [B, T, d_model] projected tensor with positional encoding.
        """
        x = self.projection(x)      # [B, T, d_model]
        x = self.pos_encoding(x)     # [B, T, d_model]
        return x
