"""Temporal Convolution Feature Extractor (Pathway A).

Multi-scale 1D temporal convolution bank with kernel sizes [3, 7, 15].
Concatenates activations from all kernels, applies BatchNorm + ReLU.
Optionally applies max pooling (stride 2) to reduce sequence length.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvBank(nn.Module):
    """Multi-scale 1D temporal convolution bank.

    Applies convolutions with multiple kernel sizes in parallel, concatenates
    the outputs, and applies BatchNorm + ReLU.

    Args:
        n_channels: Number of input channels.
        out_channels_per_kernel: Output channels per kernel size (default 256).
        kernel_sizes: Tuple of kernel sizes (default (3, 7, 15)).
        use_pooling: Whether to apply max pooling with stride 2 (default False).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        n_channels: int,
        out_channels_per_kernel: int = 256,
        kernel_sizes: tuple[int, ...] = (3, 7, 15),
        use_pooling: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_sizes = kernel_sizes
        self.use_pooling = use_pooling

        # One Conv1D per kernel size
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                n_channels,
                out_channels_per_kernel,
                kernel_size=k,
                padding=k // 2,  # same padding
            )
            for k in kernel_sizes
        ])

        total_channels = out_channels_per_kernel * len(kernel_sizes)
        self.bn = nn.BatchNorm1d(total_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.output_channels = total_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, C] (batch, time, channels).

        Returns:
            Output tensor [B, T', C_out] where C_out = out_channels_per_kernel * n_kernels,
            and T' = T // 2 if use_pooling else T.
        """
        # Conv1D expects [B, C, T]
        x = x.transpose(1, 2)  # [B, C, T]

        # Apply each convolution branch
        branch_outputs = [conv(x) for conv in self.conv_branches]  # list of [B, out_ch, T]

        # Concatenate along channel dim
        x = torch.cat(branch_outputs, dim=1)  # [B, total_channels, T]

        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        if self.use_pooling:
            x = self.pool(x)  # [B, total_channels, T//2]

        # Back to [B, T', C_out]
        x = x.transpose(1, 2)
        return x
