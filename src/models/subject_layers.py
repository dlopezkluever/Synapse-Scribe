"""Subject-specific normalization layers for multi-subject training.

Provides learned per-subject bias and scale parameters that adapt a shared
decoder to individual neural signal distributions. This is the standard
approach for multi-subject BCI models (e.g., Willett et al. 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SubjectNormalization(nn.Module):
    """Learned per-subject affine transformation applied to input channels.

    For each subject, learns a scale (gamma) and bias (beta) vector of
    dimension ``n_channels``.  During forward, the subject ID selects the
    appropriate parameters and applies: ``x * gamma[subject] + beta[subject]``.

    Args:
        n_subjects: Total number of subjects.
        n_channels: Number of input channels per timestep.
    """

    def __init__(self, n_subjects: int, n_channels: int):
        super().__init__()
        self.n_subjects = n_subjects
        self.n_channels = n_channels

        # Initialise gamma=1, beta=0 so the layer is identity at init
        self.gamma = nn.Parameter(torch.ones(n_subjects, n_channels))
        self.beta = nn.Parameter(torch.zeros(n_subjects, n_channels))

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Apply subject-specific normalization.

        Args:
            x: Input tensor [B, T, C].
            subject_ids: Integer subject indices [B] in range [0, n_subjects).

        Returns:
            Normalized tensor [B, T, C].
        """
        # gamma/beta: [B, C]
        g = self.gamma[subject_ids]  # [B, C]
        b = self.beta[subject_ids]   # [B, C]
        # Broadcast over time: [B, 1, C]
        return x * g.unsqueeze(1) + b.unsqueeze(1)


class SubjectAwareModel(nn.Module):
    """Wraps any decoder with a subject-specific normalization front-end.

    The wrapper applies ``SubjectNormalization`` before feeding data into the
    base model.  The base model's ``forward`` signature remains unchanged for
    everything except the extra ``subject_ids`` argument.

    Args:
        base_model: Any decoder (BaseDecoder subclass).
        n_subjects: Number of subjects.
        n_channels: Number of input channels.
    """

    def __init__(self, base_model: nn.Module, n_subjects: int, n_channels: int):
        super().__init__()
        self.subject_norm = SubjectNormalization(n_subjects, n_channels)
        self.base_model = base_model

    def forward(
        self, x: torch.Tensor, subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional subject normalization.

        Args:
            x: [B, T, C] input features.
            subject_ids: [B] integer subject indices.  If ``None``, the
                subject normalization layer is bypassed (useful for inference
                on unknown subjects).

        Returns:
            Logits [B, T', n_classes].
        """
        if subject_ids is not None:
            x = self.subject_norm(x, subject_ids)
        return self.base_model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
