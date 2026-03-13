"""Hybrid CNN-Transformer Model (Model D).

Has its own integrated CNN front-end that reduces sequence length by 8x,
then feeds into a smaller Transformer encoder (4 layers).
Does NOT use Pathway A or B.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.base import BaseDecoder


class _CNNFrontEnd(nn.Module):
    """3-layer CNN front-end with stride-2 max pooling.

    Each layer: Conv1D → BatchNorm → ReLU → MaxPool(2).
    Total temporal reduction: 2^3 = 8x.
    """

    def __init__(
        self,
        n_channels: int,
        cnn_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = n_channels
        for _ in range(3):
            layers.extend([
                nn.Conv1d(in_ch, cnn_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_ch = cnn_channels
        self.layers = nn.Sequential(*layers)
        self.output_channels = cnn_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, cnn_channels, T//8]."""
        return self.layers(x)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the Transformer stage."""

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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CNNTransformer(BaseDecoder):
    """Hybrid CNN-Transformer decoder.

    Architecture:
        CNN front-end (3 layers, MaxPool(2) each → 8x temporal reduction)
        → Linear projection to d_model
        → Positional encoding
        → 4-layer Transformer encoder
        → Linear → logits

    Output time dimension is T // 8.

    Args:
        n_channels: Number of input channels (default 192).
        n_classes: Number of output classes (default 28).
        cnn_channels: Channels in CNN front-end (default 256).
        d_model: Transformer model dimension (default 512).
        n_heads: Number of attention heads (default 8).
        n_transformer_layers: Transformer layers (default 4).
        ffn_dim: Feed-forward dimension (default 2048).
        dropout: Dropout probability (default 0.1).
        max_seq_len: Maximum reduced sequence length (default 4096).
    """

    def __init__(
        self,
        n_channels: int = 192,
        n_classes: int = 28,
        cnn_channels: int = 256,
        d_model: int = 512,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__(n_channels, n_classes)

        # CNN front-end (8x temporal reduction)
        self.cnn = _CNNFrontEnd(n_channels, cnn_channels, dropout)

        # Project CNN output to d_model
        self.proj = nn.Linear(cnn_channels, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, n_classes)

    @property
    def downsample_factor(self) -> int:
        return 8  # 3 MaxPool(2) layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, C].

        Returns:
            Logits tensor [B, T//8, n_classes].
        """
        # CNN expects [B, C, T]
        x = x.transpose(1, 2)          # [B, C, T]
        x = self.cnn(x)                # [B, cnn_channels, T//8]
        x = x.transpose(1, 2)          # [B, T//8, cnn_channels]

        # Project to d_model + positional encoding
        x = self.proj(x)               # [B, T//8, d_model]
        x = self.pos_enc(x)            # [B, T//8, d_model]

        # Transformer
        x = self.transformer_encoder(x)  # [B, T//8, d_model]

        # Output
        logits = self.output_proj(x)   # [B, T//8, n_classes]
        return logits
