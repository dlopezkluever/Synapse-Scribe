"""CLI entrypoint for model training.

Usage:
    python scripts/train.py --model gru_decoder --dataset willett --epochs 200
    python scripts/train.py --model cnn_lstm --epochs 100 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.data.dataset import create_dataloaders, ctc_collate_fn, NeuralTrialDataset, split_trial_index
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(yaml_path=args.config, preset="willett_handwriting")

    # Load trial index
    from src.data.loader import load_willett_dataset
    logger.info("Loading dataset...")
    trial_index = load_willett_dataset(cfg)
    logger.info("Loaded %d trials", len(trial_index))

    # Create data loaders with augmentation for training
    train_transform = get_training_transforms(
        n_masks=cfg.aug_time_mask_n,
        max_mask_ms=cfg.aug_time_mask_max_ms,
        channel_dropout_rate=cfg.aug_channel_dropout_rate,
        noise_std=cfg.aug_gaussian_noise_std,
        fs=cfg.target_fs,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        trial_index,
        t_max=args.t_max,
        batch_size=args.batch_size,
        split_ratios=cfg.split_ratios,
        train_transform=train_transform,
    )

    # Create model
    ModelClass = MODEL_REGISTRY[args.model]

    if args.model == "gru_decoder":
        model = ModelClass(n_channels=args.n_channels, n_classes=cfg.n_classes)
    elif args.model == "cnn_lstm":
        model = ModelClass(
            n_channels=args.n_channels,
            n_classes=cfg.n_classes,
            conv_channels=cfg.conv_channels,
            conv_kernel_size=cfg.conv_kernel_size,
            conv_layers=cfg.conv_layers,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
        )
    elif args.model == "transformer":
        model = ModelClass(
            n_channels=args.n_channels,
            n_classes=cfg.n_classes,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.transformer_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.transformer_dropout,
            max_seq_len=cfg.max_seq_len,
        )
    elif args.model == "cnn_transformer":
        model = ModelClass(
            n_channels=args.n_channels,
            n_classes=cfg.n_classes,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_transformer_layers=cfg.hybrid_transformer_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.transformer_dropout,
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
    )

    history = trainer.train()
    logger.info("Best validation CER: %.4f", trainer.best_val_cer)


if __name__ == "__main__":
    main()
