"""CLI entrypoint for model training.

Usage:
    python scripts/train.py --model gru_decoder --dataset willett --epochs 200
    python scripts/train.py --model cnn_lstm --epochs 100 --batch-size 8

    # Train on single-letter trials only (fast, good for validation):
    python scripts/train.py --model gru_decoder --trial-type letters --t-max 250

    # Train on sentences only (slower, use larger t-max):
    python scripts/train.py --model gru_decoder --trial-type sentences --t-max 5000 --batch-size 4

    # Filter out trials longer than t-max (avoids truncation/label mismatch):
    python scripts/train.py --model gru_decoder --filter-by-length --t-max 4000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.dataset import (
    create_dataloaders, create_dataloaders_from_splits, ctc_collate_fn,
    NeuralTrialDataset, split_trial_index, split_trial_index_by_session,
)
from src.data.transforms import get_training_transforms
from src.models.gru_decoder import GRUDecoder
from src.models.cnn_lstm import CNNLSTM
from src.models.transformer import TransformerDecoder
from src.models.cnn_transformer import CNNTransformer
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "gru_decoder": GRUDecoder,
    "cnn_lstm": CNNLSTM,
    "transformer": TransformerDecoder,
    "cnn_transformer": CNNTransformer,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural decoder model")
    parser.add_argument(
        "--model", type=str, default="gru_decoder",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to train",
    )
    parser.add_argument("--dataset", type=str, default="willett")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-channels", type=int, default=192)
    parser.add_argument("--t-max", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/checkpoints")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--trial-type", type=str, default="all",
        choices=["all", "letters", "sentences"],
        help="Filter by trial type: 'letters' (single chars, fast), "
             "'sentences' (multi-char), or 'all' (default)",
    )
    parser.add_argument(
        "--filter-by-length", action="store_true",
        help="Drop trials longer than t-max (prevents truncation/label mismatch)",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Z-score normalize per channel (recommended for raw spike count data)",
    )
    # Firing rate features (Pathway C)
    parser.add_argument(
        "--use-firing-rates", action="store_true",
        help="Apply firing rate binning + sqrt transform (Pathway C)",
    )
    parser.add_argument(
        "--bin-width-ms", type=float, default=10.0,
        help="Bin width in ms for firing rate computation (default: 10.0)",
    )
    # Session-aware splitting
    parser.add_argument(
        "--session-split", action="store_true",
        help="Split by recording session (no session spans train/val/test)",
    )
    # Sentence oversampling
    parser.add_argument(
        "--oversample-sentences", type=float, default=1.0,
        help="Oversample sentence trials by this factor (e.g. 3.0 = 3x copies in train set)",
    )
    # Model hyperparameters (override defaults for faster training)
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="GRU/LSTM hidden size (default: 512 for GRU, config for others)")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Number of recurrent layers (default: 3 for GRU)")
    parser.add_argument("--proj-dim", type=int, default=None,
                        help="Input projection dim (default: 256)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (default: 0.3)")
    # W&B experiment tracking
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="brain-text-decoder",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team or username)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None,
                        help="W&B run tags (e.g. --wandb-tags baseline gru)")
    # Temporal downsampling
    parser.add_argument("--no-downsample", action="store_true",
                        help="Disable temporal downsampling (for ablation comparisons)")
    return parser.parse_args()


def _filter_trials(trial_index, trial_type: str, t_max: int, filter_by_length: bool):
    """Filter trial index by type and/or length."""
    import pandas as pd

    original_count = len(trial_index)

    # Filter by trial type using label length as proxy
    if trial_type != "all":
        label_lens = []
        for _, row in trial_index.iterrows():
            text = Path(row["label_path"]).read_text(encoding="utf-8").strip()
            label_lens.append(len(text))
        trial_index = trial_index.copy()
        trial_index["_label_len"] = label_lens

        if trial_type == "letters":
            trial_index = trial_index[trial_index["_label_len"] <= 1]
        elif trial_type == "sentences":
            trial_index = trial_index[trial_index["_label_len"] > 1]

        trial_index = trial_index.drop(columns=["_label_len"]).reset_index(drop=True)
        logger.info("Trial type filter '%s': %d -> %d trials",
                     trial_type, original_count, len(trial_index))

    # Filter by length
    if filter_by_length and "n_timesteps" in trial_index.columns:
        before = len(trial_index)
        trial_index = trial_index[trial_index["n_timesteps"] <= t_max].reset_index(drop=True)
        logger.info("Length filter (t_max=%d): %d -> %d trials",
                     t_max, before, len(trial_index))

    if len(trial_index) == 0:
        raise ValueError("No trials remain after filtering! Adjust --trial-type or --t-max.")

    # Log stats
    if "n_timesteps" in trial_index.columns:
        ts = trial_index["n_timesteps"]
        logger.info("Signal lengths: min=%d, max=%d, mean=%d, median=%d",
                     ts.min(), ts.max(), int(ts.mean()), int(ts.median()))
        n_truncated = (ts > t_max).sum()
        if n_truncated > 0:
            logger.warning("%d/%d trials will be truncated from >%d to %d timesteps",
                           n_truncated, len(trial_index), t_max, t_max)

    return trial_index


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(yaml_path=args.config, preset="willett_handwriting")

    # Load trial index
    from src.data.loader import load_willett_dataset
    logger.info("Loading dataset...")
    trial_index = load_willett_dataset(cfg)
    logger.info("Loaded %d trials", len(trial_index))

    # Filter trials
    trial_index = _filter_trials(
        trial_index, args.trial_type, args.t_max, args.filter_by_length
    )

    # Create data loaders with augmentation for training
    train_transform = get_training_transforms(
        n_masks=cfg.aug_time_mask_n,
        max_mask_ms=cfg.aug_time_mask_max_ms,
        channel_dropout_rate=cfg.aug_channel_dropout_rate,
        noise_std=cfg.aug_gaussian_noise_std,
        fs=cfg.target_fs,
    )

    # Split explicitly so we can apply per-split modifications
    if args.session_split:
        train_df, val_df, test_df = split_trial_index_by_session(
            trial_index, cfg.split_ratios, seed=42,
        )
    else:
        train_df, val_df, test_df = split_trial_index(
            trial_index, cfg.split_ratios, seed=42,
        )

    # Oversample sentence trials in train set only
    if args.oversample_sentences > 1.0:
        sentence_mask = train_df["n_timesteps"] > 300
        sentence_df = train_df[sentence_mask]
        n_copies = int(args.oversample_sentences) - 1
        if n_copies > 0 and len(sentence_df) > 0:
            train_df = pd.concat(
                [train_df] + [sentence_df] * n_copies, ignore_index=True,
            )
            logger.info(
                "Oversampled %d sentence trials x%d → train set now %d trials",
                len(sentence_df), n_copies + 1, len(train_df),
            )

    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        train_df, val_df, test_df,
        t_max=args.t_max,
        batch_size=args.batch_size,
        train_transform=train_transform,
        normalize=args.normalize,
        use_firing_rates=args.use_firing_rates,
        bin_width_ms=args.bin_width_ms,
    )

    # Create model (CLI args override defaults for faster training)
    ModelClass = MODEL_REGISTRY[args.model]
    dropout = args.dropout if args.dropout is not None else 0.3
    use_downsample = not args.no_downsample

    if args.model == "gru_decoder":
        model = ModelClass(
            n_channels=args.n_channels, n_classes=cfg.n_classes,
            proj_dim=args.proj_dim or 256,
            hidden_size=args.hidden_size or 512,
            n_layers=args.n_layers or 3,
            dropout=dropout,
            use_downsample=use_downsample,
        )
    elif args.model == "cnn_lstm":
        model = ModelClass(
            n_channels=args.n_channels,
            n_classes=cfg.n_classes,
            conv_channels=cfg.conv_channels,
            conv_kernel_size=cfg.conv_kernel_size,
            conv_layers=cfg.conv_layers,
            lstm_hidden=args.hidden_size or cfg.lstm_hidden,
            lstm_layers=args.n_layers or cfg.lstm_layers,
            dropout=args.dropout if args.dropout is not None else cfg.lstm_dropout,
            use_downsample=use_downsample,
        )
    elif args.model == "transformer":
        model = ModelClass(
            n_channels=args.n_channels,
            n_classes=cfg.n_classes,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=args.n_layers or cfg.transformer_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=args.dropout if args.dropout is not None else cfg.transformer_dropout,
            max_seq_len=cfg.max_seq_len,
            use_downsample=use_downsample,
        )
    elif args.model == "cnn_transformer":
        model = ModelClass(
            n_channels=args.n_channels,
            n_classes=cfg.n_classes,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_transformer_layers=args.n_layers or cfg.hybrid_transformer_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=args.dropout if args.dropout is not None else cfg.transformer_dropout,
            max_seq_len=cfg.max_seq_len,
        )
    else:
        model = ModelClass(n_channels=args.n_channels, n_classes=cfg.n_classes)

    logger.info("Model: %s (%d params)", args.model, model.count_parameters())

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=cfg.weight_decay,
        max_epochs=args.epochs,
        warmup_steps=cfg.warmup_steps,
        grad_clip_max_norm=cfg.grad_clip_max_norm,
        early_stopping_patience=cfg.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        mixed_precision=cfg.mixed_precision,
        device=args.device,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    history = trainer.train()
    logger.info("Best validation CER: %.4f", trainer.best_val_cer)


if __name__ == "__main__":
    main()
