"""Transformer Encoder Model (Model C).

Uses Pathway B (linear projection + positional encoding) as its input stage.
N Transformer encoder layers with multi-head self-attention.

With use_downsample=True (default), a CNN front-end replaces the linear projection
for 4x temporal reduction before the expensive O(T^2) attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseDecoder
from src.features.projection import LinearProjection, SinusoidalPositionalEncoding


class TransformerDecoder(BaseDecoder):
    """Transformer encoder-based decoder for neural signals.

    Architecture (use_downsample=True):
        ConvEmbedding (4x temporal reduction) → PositionalEncoding
        → N × TransformerEncoderLayer → Linear → logits

    Architecture (use_downsample=False):
        LinearProjection (Pathway B) → N × TransformerEncoderLayer → Linear → logits

    Args:
        n_channels: Number of input channels (default 192).
        n_classes: Number of output classes (default 28).
        d_model: Transformer model dimension (default 512).
        n_heads: Number of attention heads (default 8).
        n_layers: Number of Transformer encoder layers (default 6).
        ffn_dim: Feed-forward network dimension (default 2048).
        dropout: Dropout probability (default 0.1).
        max_seq_len: Maximum sequence length for positional encoding (default 4096).
        use_downsample: If True, use CNN front-end with 4x temporal reduction.
    """

    def __init__(
        self,
        n_channels: int = 192,
        n_classes: int = 28,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        use_downsample: bool = True,
    ):
        super().__init__(n_channels, n_classes)
        self._use_downsample = use_downsample

        if use_downsample:
            # CNN front-end for 4x temporal reduction
            self.conv_embed = nn.Sequential(
                nn.Conv1d(n_channels, d_model, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
            )
            # Positional encoding sized for downsampled length
            self.pos_enc = SinusoidalPositionalEncoding(
                d_model, max_len=max_seq_len // 4, dropout=dropout,
            )
        else:
            # Pathway B: linear projection + positional encoding
            self.input_proj = LinearProjection(
                n_channels=n_channels,
                d_model=d_model,
                max_len=max_seq_len,
                dropout=dropout,
            )

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
            num_layers=n_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, n_classes)

    @property
    def downsample_factor(self) -> int:
        return 4 if self._use_downsample else 1

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, C].
            src_key_padding_mask: Optional boolean mask [B, T] where True
                indicates padded positions to be ignored.

        Returns:
            Logits tensor [B, T', n_classes].
        """
        if self._use_downsample:
            x = x.transpose(1, 2)       # [B, T, C] -> [B, C, T]
            x = self.conv_embed(x)      # [B, d_model, T//4]
            x = x.transpose(1, 2)       # [B, T//4, d_model]
            x = self.pos_enc(x)         # [B, T//4, d_model]
            # Padding mask not applicable after conv downsampling
            x = self.transformer_encoder(x)
        else:
            x = self.input_proj(x)      # [B, T, d_model]
            x = self.transformer_encoder(
                x, src_key_padding_mask=src_key_padding_mask,
            )

        logits = self.output_proj(x)    # [B, T', n_classes]
        return logits
