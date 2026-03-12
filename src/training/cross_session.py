"""Cross-session generalization utilities.

Provides session-based data splitting, per-session z-score normalization
(to mitigate neural drift), and evaluation helpers for comparing
within-session vs. cross-session CER.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    NeuralTrialDataset,
    ctc_collate_fn,
)
from src.decoding.greedy import greedy_decode_batch
from src.evaluation.metrics import compute_cer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session-based splitting
# ---------------------------------------------------------------------------

def split_by_session(
    trial_index: pd.DataFrame,
    train_sessions: list[str | int],
    eval_sessions: list[str | int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split trial index by session ID.

    Args:
        trial_index: Full trial index with a 'session' column.
        train_sessions: Session IDs to use for training.
        eval_sessions: Session IDs to use for evaluation.

    Returns:
        (train_df, eval_df)
    """
    session_col = trial_index["session"].astype(str)
    train_mask = session_col.isin([str(s) for s in train_sessions])
    eval_mask = session_col.isin([str(s) for s in eval_sessions])

    train_df = trial_index[train_mask].reset_index(drop=True)
    eval_df = trial_index[eval_mask].reset_index(drop=True)

    logger.info(
        "Session split: train=%d trials (sessions %s), eval=%d trials (sessions %s)",
        len(train_df), train_sessions, len(eval_df), eval_sessions,
    )
    return train_df, eval_df


def get_available_sessions(trial_index: pd.DataFrame) -> list[str]:
    """Return sorted unique session IDs."""
    return sorted(trial_index["session"].astype(str).unique())


# ---------------------------------------------------------------------------
# Per-session normalization
# ---------------------------------------------------------------------------

class SessionNormalizer:
    """Per-session z-score normalization to mitigate neural drift.

    Computes per-channel mean and std independently for each session, so
    that session-specific distribution shifts are removed before training.

    Args:
        trial_index: DataFrame with 'session' and 'signal_path' columns.
        max_trials_per_session: Max trials sampled per session for stats.
    """

    def __init__(
        self,
        trial_index: pd.DataFrame,
        max_trials_per_session: int = 200,
    ):
        self.stats: dict[str, dict[str, np.ndarray]] = {}
        self._compute_stats(trial_index, max_trials_per_session)

    def _compute_stats(
        self, trial_index: pd.DataFrame, max_trials: int,
    ) -> None:
        sessions = trial_index["session"].astype(str).unique()
        for sess in sessions:
            sess_df = trial_index[trial_index["session"].astype(str) == sess]
            n = min(max_trials, len(sess_df))
            all_data = []
            for i in range(n):
                row = sess_df.iloc[i]
                data = np.load(row["signal_path"]).astype(np.float32)
                all_data.append(data)
            concat = np.concatenate(all_data, axis=0)
            self.stats[sess] = {
                "mean": concat.mean(axis=0),
                "std": concat.std(axis=0) + 1e-8,
            }
            logger.info(
                "Session '%s': computed stats from %d trials (mean=%.4f, std=%.4f)",
                sess, n, self.stats[sess]["mean"].mean(),
                self.stats[sess]["std"].mean(),
            )

    def normalize(
        self, features: np.ndarray, session: str,
    ) -> np.ndarray:
        """Apply session-specific z-score normalization.

        Args:
            features: [T, C] raw features.
            session: Session ID string.

        Returns:
            Normalized [T, C] array.
        """
        if session in self.stats:
            return (features - self.stats[session]["mean"]) / self.stats[session]["std"]
        # Fallback: global mean across all sessions
        logger.warning("Session '%s' not in stats; using global fallback", session)
        all_means = np.stack([s["mean"] for s in self.stats.values()])
        all_stds = np.stack([s["std"] for s in self.stats.values()])
        return (features - all_means.mean(0)) / all_stds.mean(0)


class SessionNormalizedDataset(torch.utils.data.Dataset):
    """Dataset with per-session normalization applied on the fly."""

    def __init__(
        self,
        trial_index: pd.DataFrame,
        normalizer: SessionNormalizer,
        t_max: int = 2000,
        transform=None,
    ):
        from src.data.dataset import text_to_indices
        self.trial_index = trial_index.reset_index(drop=True)
        self.normalizer = normalizer
        self.t_max = t_max
        self.transform = transform
        self._text_to_indices = text_to_indices

    def __len__(self) -> int:
        return len(self.trial_index)

    def __getitem__(self, idx: int) -> dict:
        from pathlib import Path

        row = self.trial_index.iloc[idx]
        features = np.load(row["signal_path"]).astype(np.float32)
        session = str(row["session"])

        # Per-session normalization
        features = self.normalizer.normalize(features, session)

        # Load label
        label_path = Path(row["label_path"])
        label_text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        target = self._text_to_indices(label_text)

        actual_length = min(features.shape[0], self.t_max)
        if features.shape[0] > self.t_max:
            features = features[: self.t_max]

        if self.transform is not None:
            features = self.transform(features)

        return {
            "features": torch.from_numpy(features),
            "target": torch.tensor(target, dtype=torch.long),
            "input_length": actual_length,
            "target_length": len(target),
            "label_text": label_text,
            "session": session,
        }


# ---------------------------------------------------------------------------
# Cross-session evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cross_session(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: str | torch.device = "cpu",
) -> dict:
    """Evaluate a model on a session-split eval set.

    Args:
        model: Trained decoder model.
        eval_loader: DataLoader for the held-out session(s).
        device: Torch device.

    Returns:
        Dict with 'cer', 'n_trials', 'predictions', 'references'.
    """
    model.eval()
    dev = next(model.parameters()).device

    all_preds: list[str] = []
    all_refs: list[str] = []

    for batch in eval_loader:
        features = batch["features"].to(dev)
        logits = model(features)
        decoded = greedy_decode_batch(logits)
        all_preds.extend(decoded)
        all_refs.extend(batch["label_texts"])

    cer = compute_cer(all_preds, all_refs) if all_refs else 1.0
    return {
        "cer": cer,
        "n_trials": len(all_refs),
        "predictions": all_preds,
        "references": all_refs,
    }


def cross_session_report(
    model: torch.nn.Module,
    trial_index: pd.DataFrame,
    t_max: int = 2000,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
    use_session_norm: bool = True,
) -> dict:
    """Generate a full cross-session generalization report.

    For each pair of sessions, trains on one and evaluates on the other,
    comparing within-session (train & eval on same session) vs. cross-session
    (train on one, eval on another) CER.

    This function only *evaluates* using the provided already-trained model,
    not re-trains.  For a full re-training comparison, use the Trainer with
    session-split dataloaders.

    Args:
        model: Trained decoder model.
        trial_index: Full trial index with 'session' column.
        t_max: Maximum timesteps.
        batch_size: Batch size for evaluation.
        device: Torch device.
        use_session_norm: Whether to apply per-session normalization.

    Returns:
        Dict with per-session CER results.
    """
    sessions = get_available_sessions(trial_index)
    logger.info("Evaluating cross-session on %d sessions: %s", len(sessions), sessions)

    normalizer = None
    if use_session_norm:
        normalizer = SessionNormalizer(trial_index)

    results: dict[str, dict] = {}
    for sess in sessions:
        sess_df = trial_index[trial_index["session"].astype(str) == sess].reset_index(drop=True)

        if len(sess_df) == 0:
            continue

        if normalizer is not None:
            ds = SessionNormalizedDataset(sess_df, normalizer, t_max=t_max)
        else:
            ds = NeuralTrialDataset(sess_df, t_max=t_max)

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=ctc_collate_fn,
        )

        res = evaluate_cross_session(model, loader, device)
        results[sess] = res
        logger.info("Session '%s': CER=%.4f (%d trials)", sess, res["cer"], res["n_trials"])

    return results
