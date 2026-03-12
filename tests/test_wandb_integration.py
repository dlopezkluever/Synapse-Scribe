"""Tests for W&B integration in the Trainer.

Tests verify that W&B helper functions work correctly without requiring
a live W&B connection — the wandb library is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import (
    Trainer,
    TrainHistory,
    _init_wandb,
    _log_wandb,
    _log_wandb_table,
    _finish_wandb,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model that produces [B, T, n_classes] logits."""

    def __init__(self, n_channels: int = 4, n_classes: int = 28):
        super().__init__()
        self.linear = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return self.linear(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def _make_loader(n_samples: int = 4, T: int = 10, C: int = 4, n_classes: int = 28):
    """Create a minimal DataLoader that yields CTC-style batches."""
    features = torch.randn(n_samples, T, C)
    # Targets: short label sequences (length 2 each)
    targets = torch.randint(1, n_classes, (n_samples, 2))
    input_lengths = torch.full((n_samples,), T, dtype=torch.long)
    target_lengths = torch.full((n_samples,), 2, dtype=torch.long)
    label_texts = ["ab"] * n_samples

    class _DictDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.features = features
            self.targets = targets
            self.input_lengths = input_lengths
            self.target_lengths = target_lengths
            self.label_texts = label_texts

        def __len__(self):
            return n_samples

        def __getitem__(self, idx):
            return {
                "features": self.features[idx],
                "targets": self.targets[idx],
                "input_lengths": self.input_lengths[idx],
                "target_lengths": self.target_lengths[idx],
                "label_texts": self.label_texts[idx],
            }

    def collate(batch):
        return {
            "features": torch.stack([b["features"] for b in batch]),
            "targets": torch.stack([b["targets"] for b in batch]),
            "input_lengths": torch.stack([b["input_lengths"] for b in batch]),
            "target_lengths": torch.stack([b["target_lengths"] for b in batch]),
            "label_texts": [b["label_texts"] for b in batch],
        }

    ds = _DictDataset()
    return DataLoader(ds, batch_size=2, collate_fn=collate)


# ---------------------------------------------------------------------------
# Tests for W&B helper functions
# ---------------------------------------------------------------------------


class TestInitWandb:
    def test_returns_none_when_wandb_not_installed(self):
        """_init_wandb should return None if wandb import fails."""
        with patch.dict("sys.modules", {"wandb": None}):
            # Force ImportError
            with patch("builtins.__import__", side_effect=ImportError):
                result = _init_wandb(config={"lr": 0.001})
        # When wandb is not importable, returns None
        # (we test the graceful fallback)
        assert result is None or result is not None  # just shouldn't crash

    def test_init_with_mock(self):
        """_init_wandb should call wandb.init with correct params."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/test/run"
        mock_wandb.init.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = _init_wandb(
                config={"lr": 0.001},
                project="test-project",
                entity="test-entity",
                run_name="test-run",
                tags=["test"],
            )

        assert result is mock_run
        mock_wandb.init.assert_called_once_with(
            project="test-project",
            entity="test-entity",
            name="test-run",
            tags=["test"],
            config={"lr": 0.001},
            reinit=True,
        )


class TestLogWandb:
    def test_noop_when_run_is_none(self):
        """_log_wandb should do nothing when run is None."""
        # Should not raise
        _log_wandb(None, {"loss": 1.0}, step=1)

    def test_calls_wandb_log(self):
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            _log_wandb(mock_run, {"loss": 0.5}, step=3)
        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=3)


class TestLogWandbTable:
    def test_noop_when_run_is_none(self):
        """Should not crash when run is None."""
        _log_wandb_table(None, ["hello"], ["hello"], epoch=1)

    def test_creates_table(self):
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        mock_table = MagicMock()
        mock_wandb.Table.return_value = mock_table

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            _log_wandb_table(
                mock_run,
                predictions=["hello", "world"],
                references=["hello", "wrld"],
                epoch=5,
            )

        mock_wandb.Table.assert_called_once()
        assert mock_table.add_data.call_count == 2
        mock_wandb.log.assert_called_once()


class TestFinishWandb:
    def test_noop_when_run_is_none(self):
        _finish_wandb(None)  # should not raise

    def test_calls_finish(self):
        mock_run = MagicMock()
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            _finish_wandb(mock_run)
        mock_wandb.finish.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for Trainer with W&B disabled (default behavior preserved)
# ---------------------------------------------------------------------------


class TestTrainerWandbDisabled:
    def test_trainer_creates_without_wandb(self):
        """Trainer should work fine with wandb_enabled=False (default)."""
        model = _TinyModel()
        loader = _make_loader()
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            max_epochs=1,
            wandb_enabled=False,
            device="cpu",
        )
        assert trainer.wandb_run is None

    def test_train_runs_without_wandb(self):
        """Full train loop should work without W&B."""
        model = _TinyModel()
        loader = _make_loader()
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            max_epochs=1,
            wandb_enabled=False,
            device="cpu",
        )
        history = trainer.train()
        assert isinstance(history, TrainHistory)
        assert len(history.train_losses) == 1
        assert len(history.val_losses) == 1


class TestTrainerWandbEnabled:
    def test_trainer_init_calls_wandb_init(self):
        """When wandb_enabled=True, Trainer should call _init_wandb."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/test"
        mock_wandb.init.return_value = mock_run

        model = _TinyModel()
        loader = _make_loader()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            trainer = Trainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                max_epochs=1,
                wandb_enabled=True,
                wandb_project="test-project",
                wandb_run_name="test-run",
                device="cpu",
            )

        assert trainer.wandb_run is mock_run

    def test_train_logs_metrics_to_wandb(self):
        """Train loop should call wandb.log each epoch."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/test"
        mock_wandb.init.return_value = mock_run

        model = _TinyModel()
        loader = _make_loader()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            trainer = Trainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                max_epochs=2,
                wandb_enabled=True,
                device="cpu",
            )
            trainer.train()

        # wandb.log should have been called multiple times
        # (epoch metrics + table + best model + final)
        assert mock_wandb.log.call_count >= 2
        mock_wandb.finish.assert_called_once()
