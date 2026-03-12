"""Multi-subject training utilities.

Provides helpers for pooling data across subjects, creating subject-aware
DataLoaders, and evaluating per-subject performance (within-subject vs.
cross-subject CER).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import (
    NeuralTrialDataset,
    ctc_collate_fn,
    split_trial_index,
    text_to_indices,
)
from src.decoding.greedy import greedy_decode_batch
from src.evaluation.metrics import compute_cer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subject-aware dataset
# ---------------------------------------------------------------------------

class SubjectAwareDataset(Dataset):
    """Wraps NeuralTrialDataset to also return a numeric subject ID per sample."""

    def __init__(
        self,
        trial_index: pd.DataFrame,
        subject_map: dict[str, int],
        t_max: int = 2000,
        transform=None,
        normalize: bool = False,
        channel_mean: np.ndarray | None = None,
        channel_std: np.ndarray | None = None,
    ):
        self.inner = NeuralTrialDataset(
            trial_index,
            t_max=t_max,
            transform=transform,
            normalize=normalize,
            channel_mean=channel_mean,
            channel_std=channel_std,
        )
        self.trial_index = self.inner.trial_index
        self.subject_map = subject_map
        # Expose normalization stats for sharing with val/test sets
        self.channel_mean = self.inner.channel_mean
        self.channel_std = self.inner.channel_std

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        item = self.inner[idx]
        row = self.inner.trial_index.iloc[idx]
        subject = str(row["subject"])
        item["subject_id"] = self.subject_map.get(subject, 0)
        return item


def subject_aware_collate_fn(batch: list[dict]) -> dict:
    """Collate that includes subject_id alongside CTC fields."""
    base = ctc_collate_fn(batch)
    base["subject_ids"] = torch.tensor(
        [item["subject_id"] for item in batch], dtype=torch.long,
    )
    return base


# ---------------------------------------------------------------------------
# Data pooling
# ---------------------------------------------------------------------------

def build_subject_map(trial_index: pd.DataFrame) -> dict[str, int]:
    """Create a mapping from subject string IDs to integer indices.

    Returns:
        Dict mapping subject string to int (sorted alphabetically).
    """
    subjects = sorted(trial_index["subject"].astype(str).unique())
    return {s: i for i, s in enumerate(subjects)}


def create_multi_subject_dataloaders(
    trial_index: pd.DataFrame,
    t_max: int = 2000,
    batch_size: int = 16,
    split_ratios: list[float] | None = None,
    train_transform=None,
    num_workers: int = 0,
    seed: int = 42,
    normalize: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Create subject-aware DataLoaders for multi-subject training.

    Returns:
        (train_loader, val_loader, test_loader, subject_map)
    """
    subject_map = build_subject_map(trial_index)
    logger.info("Subject map: %s", subject_map)

    train_df, val_df, test_df = split_trial_index(trial_index, split_ratios, seed)

    train_ds = SubjectAwareDataset(
        train_df, subject_map, t_max=t_max,
        transform=train_transform, normalize=normalize,
    )
    val_ds = SubjectAwareDataset(
        val_df, subject_map, t_max=t_max, normalize=normalize,
        channel_mean=train_ds.channel_mean, channel_std=train_ds.channel_std,
    )
    test_ds = SubjectAwareDataset(
        test_df, subject_map, t_max=t_max, normalize=normalize,
        channel_mean=train_ds.channel_mean, channel_std=train_ds.channel_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=subject_aware_collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=subject_aware_collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=subject_aware_collate_fn, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, subject_map


# ---------------------------------------------------------------------------
# Per-subject evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_per_subject(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
) -> dict:
    """Evaluate per-subject CER using a subject-aware model.

    The model is expected to accept ``(features, subject_ids)`` if it is a
    ``SubjectAwareModel``, otherwise ``(features,)`` alone.

    Args:
        model: Trained decoder (may be SubjectAwareModel).
        dataloader: DataLoader yielding dicts with 'features', 'label_texts',
            and optionally 'subject_ids'.
        device: Torch device.

    Returns:
        Dict with:
            per_subject: dict[subject_id, {"cer": float, "n_trials": int}]
            overall_cer: float
    """
    model.eval()
    model_device = next(model.parameters()).device

    subject_preds: dict[int, list[str]] = {}
    subject_refs: dict[int, list[str]] = {}
    all_preds: list[str] = []
    all_refs: list[str] = []

    for batch in dataloader:
        features = batch["features"].to(model_device)
        label_texts = batch["label_texts"]
        subject_ids = batch.get("subject_ids")

        # Forward
        if subject_ids is not None and hasattr(model, "subject_norm"):
            logits = model(features, subject_ids.to(model_device))
        else:
            logits = model(features) if subject_ids is None else model(features)

        decoded = greedy_decode_batch(logits)
        all_preds.extend(decoded)
        all_refs.extend(label_texts)

        if subject_ids is not None:
            for sid, pred, ref in zip(subject_ids.tolist(), decoded, label_texts):
                subject_preds.setdefault(sid, []).append(pred)
                subject_refs.setdefault(sid, []).append(ref)

    overall_cer = compute_cer(all_preds, all_refs) if all_refs else 1.0

    per_subject = {}
    for sid in sorted(subject_preds.keys()):
        preds = subject_preds[sid]
        refs = subject_refs[sid]
        cer = compute_cer(preds, refs) if refs else 1.0
        per_subject[sid] = {"cer": cer, "n_trials": len(refs)}

    return {"per_subject": per_subject, "overall_cer": overall_cer}
