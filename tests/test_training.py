"""Tests for training infrastructure — CTC loss, scheduler, trainer."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.ctc_loss import CTCLossWrapper
from src.training.scheduler import cosine_warmup_scheduler


class TestCTCLossWrapper:
    def test_computes_loss(self):
        """CTC loss should return a finite scalar."""
        ctc = CTCLossWrapper(blank=0)
        B, T, C = 2, 50, 28
        logits = torch.randn(B, T, C, requires_grad=True)
        targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)  # 2 targets of length 3
        input_lengths = torch.tensor([50, 50], dtype=torch.long)
        target_lengths = torch.tensor([3, 3], dtype=torch.long)

        loss = ctc(logits, targets, input_lengths, target_lengths)
        assert loss.dim() == 0  # scalar
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        """Loss.backward() should produce gradients."""
        ctc = CTCLossWrapper(blank=0)
        logits = torch.randn(2, 30, 28, requires_grad=True)
        targets = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        input_lengths = torch.tensor([30, 30], dtype=torch.long)
        target_lengths = torch.tensor([2, 2], dtype=torch.long)

        loss = ctc(logits, targets, input_lengths, target_lengths)
        loss.backward()
        assert logits.grad is not None

    def test_clamps_input_lengths(self):
        """Input lengths exceeding T should be clamped."""
        ctc = CTCLossWrapper(blank=0)
        logits = torch.randn(1, 10, 28)
        targets = torch.tensor([1, 2], dtype=torch.long)
        input_lengths = torch.tensor([100], dtype=torch.long)  # exceeds T=10
        target_lengths = torch.tensor([2], dtype=torch.long)

        loss = ctc(logits, targets, input_lengths, target_lengths)
        assert torch.isfinite(loss)


class TestCosineWarmupScheduler:
    def test_warmup_phase(self):
        """LR should increase during warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # LR should be monotonically increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1]

    def test_cosine_decay_phase(self):
        """LR should decrease after warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=10, total_steps=100)

        # Skip warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        # Continue into cosine decay
        for _ in range(50):
            optimizer.step()
            scheduler.step()

        lr_after_decay = optimizer.param_groups[0]["lr"]
        assert lr_after_decay < lr_at_warmup_end

    def test_lr_never_zero(self):
        """LR should never go below min_lr_ratio."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = cosine_warmup_scheduler(
            optimizer, warmup_steps=10, total_steps=100, min_lr_ratio=0.01
        )

        for _ in range(200):  # go well past total_steps
            optimizer.step()
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        assert lr >= 0.01
