"""Learning rate schedulers for training.

Implements cosine annealing with linear warmup.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Create a cosine annealing scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (warmup + cosine).
        min_lr_ratio: Minimum LR as a fraction of the peak LR.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
