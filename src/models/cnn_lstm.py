"""CNN + LSTM Model (Model B).

Architecture: 3-layer Conv1D(256, kernel=7, BN+ReLU) -> BiLSTM(512, 2 layers) -> Linear -> logits.
Uses the firing-rate-binned Willett data (Pathway C output) as input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseDecoder


class CNNBlock(nn.Module):
    """Single Conv1D block: Conv1D + BatchNorm + ReLU + Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> [B, C_out, T]."""
        return self.dropout(self.relu(self.bn(self.conv(x))))


class CNNLSTM(BaseDecoder):
    """CNN + Bidirectional LSTM decoder.

    Args:
        n_channels: Number of input channels (default 192).
        n_classes: Number of output classes (default 28).
        conv_channels: Channels in Conv1D blocks (default 256).
        conv_kernel_size: Kernel size for Conv1D (default 7).
        conv_layers: Number of Conv1D blocks (default 3).
        lstm_hidden: LSTM hidden size (default 512).
        lstm_layers: Number of LSTM layers (default 2).
        dropout: Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_channels: int = 192,
        n_classes: int = 28,
        conv_channels: int = 256,
        conv_kernel_size: int = 7,
        conv_layers: int = 3,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__(n_channels, n_classes)

        # CNN front-end
        cnn_blocks = []
        in_ch = n_channels
        for _ in range(conv_layers):
            cnn_blocks.append(CNNBlock(in_ch, conv_channels, conv_kernel_size, dropout))
            in_ch = conv_channels
        self.cnn = nn.Sequential(*cnn_blocks)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Output projection (bidir doubles hidden size)
        self.output_proj = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, T, C] -> [B, T, n_classes]."""
        # CNN expects [B, C, T]
        x = x.transpose(1, 2)          # [B, C, T]
        x = self.cnn(x)                # [B, conv_channels, T]
        x = x.transpose(1, 2)          # [B, T, conv_channels]

        x, _ = self.lstm(x)            # [B, T, lstm_hidden * 2]
        logits = self.output_proj(x)   # [B, T, n_classes]
        return logits
