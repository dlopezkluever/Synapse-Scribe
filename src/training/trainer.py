"""Training loop with validation, checkpointing, and early stopping.

Supports CTC-based training for all decoder models.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.ctc_loss import CTCLossWrapper
from src.training.scheduler import cosine_warmup_scheduler
from src.decoding.greedy import greedy_decode_batch
from src.evaluation.metrics import compute_cer

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    """Training history for plotting and analysis."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_cers: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


class Trainer:
    """CTC-based trainer with early stopping and checkpointing.

    Args:
        model: Decoder model (BaseDecoder subclass).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW.
        max_epochs: Maximum training epochs.
        warmup_steps: LR warmup steps.
        grad_clip_max_norm: Max gradient norm for clipping.
        early_stopping_patience: Epochs to wait for improvement before stopping.
        checkpoint_dir: Directory to save checkpoints.
        mixed_precision: Whether to use fp16 (GPU only).
        device: Target device (auto-detected if None).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        warmup_steps: int = 500,
        grad_clip_max_norm: float = 1.0,
        early_stopping_patience: int = 20,
        checkpoint_dir: str = "./outputs/checkpoints",
        mixed_precision: bool = True,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.grad_clip_max_norm = grad_clip_max_norm
        self.patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        # Loss
        self.criterion = CTCLossWrapper(blank=0)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler
        total_steps = max_epochs * len(train_loader)
        self.scheduler = cosine_warmup_scheduler(
            self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )

        # Mixed precision
        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Tracking
        self.history = TrainHistory()
        self.best_val_cer = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(features)
                    loss = self.criterion(logits, targets, input_lengths, target_lengths)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(features)
                loss = self.criterion(logits, targets, input_lengths, target_lengths)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> tuple[float, float, list[str], list[str]]:
        """Run validation. Returns (val_loss, val_cer, predictions, references)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_predictions = []
        all_references = []

        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            label_texts = batch["label_texts"]

            logits = self.model(features)
            loss = self.criterion(logits, targets, input_lengths, target_lengths)

            total_loss += loss.item()
            n_batches += 1

            # Greedy decode
            decoded = greedy_decode_batch(logits)
            all_predictions.extend(decoded)
            all_references.extend(label_texts)

        avg_loss = total_loss / max(n_batches, 1)

        # Compute CER
        cer = compute_cer(all_predictions, all_references) if all_references else 1.0

        return avg_loss, cer, all_predictions, all_references

    def save_checkpoint(self, path: Path, epoch: int, val_cer: float) -> None:
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_cer": val_cer,
            "global_step": self.global_step,
        }, path)
        logger.info("Saved checkpoint to %s (val_cer=%.4f)", path, val_cer)

    def load_checkpoint(self, path: Path) -> dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint

    def train(self) -> TrainHistory:
        """Full training loop with early stopping.

        Returns:
            TrainHistory with per-epoch metrics.
        """
        model_name = self.model.__class__.__name__
        logger.info(
            "Starting training: model=%s, device=%s, epochs=%d, amp=%s",
            model_name, self.device, self.max_epochs, self.use_amp,
        )
        logger.info("Parameters: %d", sum(p.numel() for p in self.model.parameters()))

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_cer, preds, refs = self.validate()

            # Record
            lr = self.optimizer.param_groups[0]["lr"]
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.val_cers.append(val_cer)
            self.history.learning_rates.append(lr)

            elapsed = time.time() - t0

            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_cer=%.4f, "
                "lr=%.2e, time=%.1fs",
                epoch, self.max_epochs, train_loss, val_loss, val_cer, lr, elapsed,
            )

            # Show sample predictions
            if preds and refs:
                n_show = min(3, len(preds))
                for i in range(n_show):
                    logger.info("  [%d] pred=%-30s ref=%s", i, repr(preds[i][:50]), repr(refs[i][:50]))

            # Check for best model
            if val_cer < self.best_val_cer:
                self.best_val_cer = val_cer
                self.epochs_without_improvement = 0
                best_path = self.checkpoint_dir / f"{model_name}_best.pt"
                self.save_checkpoint(best_path, epoch, val_cer)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping: no improvement for %d epochs (best_cer=%.4f)",
                    self.patience, self.best_val_cer,
                )
                break

        logger.info("Training complete. Best val CER: %.4f", self.best_val_cer)
        return self.history
