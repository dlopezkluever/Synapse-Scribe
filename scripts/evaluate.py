"""CLI entrypoint for model evaluation.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/GRUDecoder_best.pt --model gru_decoder
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import (
    NeuralTrialDataset, ctc_collate_fn, split_trial_index,
)
from src.decoding.greedy import greedy_decode_batch
from src.evaluation.metrics import compute_cer, compute_wer, exact_match_accuracy
from src.models.gru_decoder import GRUDecoder
from src.models.cnn_lstm import CNNLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "gru_decoder": GRUDecoder,
    "cnn_lstm": CNNLSTM,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained decoder model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--n-channels", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--t-max", type=int, default=2000)
    parser.add_argument("--results-dir", type=str, default="./outputs/results")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(yaml_path=args.config, preset="willett_handwriting")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load dataset
    from src.data.loader import load_willett_dataset
    trial_index = load_willett_dataset(cfg)
    _, _, test_df = split_trial_index(trial_index, cfg.split_ratios)

    test_ds = NeuralTrialDataset(test_df, t_max=args.t_max)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn
    )

    # Load model
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(n_channels=args.n_channels, n_classes=cfg.n_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Inference
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            logits = model(features)
            decoded = greedy_decode_batch(logits)
            all_preds.extend(decoded)
            all_refs.extend(batch["label_texts"])

    # Metrics
    cer = compute_cer(all_preds, all_refs)
    wer = compute_wer(all_preds, all_refs)
    em = exact_match_accuracy(all_preds, all_refs)

    logger.info("=== Test Results ===")
    logger.info("CER:          %.4f (%.1f%%)", cer, cer * 100)
    logger.info("WER:          %.4f (%.1f%%)", wer, wer * 100)
    logger.info("Exact Match:  %.4f (%.1f%%)", em, em * 100)
    logger.info("Samples:      %d", len(all_preds))

    # Save per-trial results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{args.model}_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_idx", "prediction", "reference"])
        for i, (p, r) in enumerate(zip(all_preds, all_refs)):
            writer.writerow([i, p, r])

    logger.info("Saved predictions to %s", csv_path)


if __name__ == "__main__":
    main()
