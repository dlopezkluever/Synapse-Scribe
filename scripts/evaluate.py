"""CLI entrypoint for model evaluation.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/GRUDecoder_best.pt --model gru_decoder
    python scripts/evaluate.py --checkpoint outputs/checkpoints/TransformerDecoder_best.pt --model transformer --beam-width 50
    python scripts/evaluate.py --checkpoint outputs/checkpoints/CNNTransformer_best.pt --model cnn_transformer --beam-width 100 --use-lm
    python scripts/evaluate.py --checkpoint outputs/checkpoints/CNNLSTM_best.pt --model cnn_lstm --output-format json
"""

from __future__ import annotations

import argparse
import csv
import json
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
from src.decoding.beam_search import beam_search_decode_batch
from src.decoding.lm_correction import load_lm_scorer, rescore_hypotheses
from src.evaluation.metrics import compute_cer, compute_wer, exact_match_accuracy
from src.models.gru_decoder import GRUDecoder
from src.models.cnn_lstm import CNNLSTM
from src.models.transformer import TransformerDecoder
from src.models.cnn_transformer import CNNTransformer

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
    parser = argparse.ArgumentParser(description="Evaluate a trained decoder model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to evaluate",
    )
    parser.add_argument("--n-channels", type=int, default=192,
                        help="Number of input channels (default: 192)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Evaluation batch size (default: 16)")
    parser.add_argument("--t-max", type=int, default=2000,
                        help="Maximum time steps per trial (default: 2000)")
    parser.add_argument("--normalize", action="store_true",
                        help="Z-score normalize per channel (must match training)")
    parser.add_argument("--filter-by-length", action="store_true",
                        help="Drop trials longer than t-max")
    parser.add_argument("--results-dir", type=str, default="./outputs/results",
                        help="Directory to save results (default: ./outputs/results)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")

    # Decoding options
    parser.add_argument("--beam-width", type=int, default=0,
                        help="Beam width for beam search decoding (0 = greedy, default: 0)")
    parser.add_argument("--use-lm", action="store_true",
                        help="Use language model rescoring with beam search")
    parser.add_argument("--lm-path", type=str, default=None,
                        help="Path to KenLM model file (required if --use-lm is set)")
    parser.add_argument("--lm-alpha", type=float, default=0.5,
                        help="LM interpolation weight (default: 0.5)")
    parser.add_argument("--lm-beta", type=float, default=0.0,
                        help="Word insertion bonus (default: 0.0)")
    parser.add_argument("--lm-type", type=str, default="char_ngram",
                        choices=["kenlm", "char_ngram", "gpt2"],
                        help="LM scorer type (default: char_ngram)")

    # Output format
    parser.add_argument("--output-format", type=str, default="csv",
                        choices=["csv", "json"],
                        help="Output format for per-trial results (default: csv)")

    return parser.parse_args()


def build_model(model_name: str, n_channels: int, cfg) -> torch.nn.Module:
    """Instantiate a model with architecture-specific hyperparameters."""
    ModelClass = MODEL_REGISTRY[model_name]

    if model_name == "gru_decoder":
        model = ModelClass(n_channels=n_channels, n_classes=cfg.n_classes)
    elif model_name == "cnn_lstm":
        model = ModelClass(
            n_channels=n_channels,
            n_classes=cfg.n_classes,
            conv_channels=cfg.conv_channels,
            conv_kernel_size=cfg.conv_kernel_size,
            conv_layers=cfg.conv_layers,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
        )
    elif model_name == "transformer":
        model = ModelClass(
            n_channels=n_channels,
            n_classes=cfg.n_classes,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.transformer_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.transformer_dropout,
            max_seq_len=cfg.max_seq_len,
        )
    elif model_name == "cnn_transformer":
        model = ModelClass(
            n_channels=n_channels,
            n_classes=cfg.n_classes,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_transformer_layers=cfg.hybrid_transformer_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.transformer_dropout,
            max_seq_len=cfg.max_seq_len,
        )
    else:
        model = ModelClass(n_channels=n_channels, n_classes=cfg.n_classes)

    return model


def decode_logits(logits, args, lm_scorer=None):
    """Decode logits using the configured decoding strategy."""
    if args.beam_width > 0:
        batch_hyps = beam_search_decode_batch(
            logits, beam_width=args.beam_width, top_k=5,
        )
        if args.use_lm and lm_scorer is not None:
            decoded = []
            for hyps in batch_hyps:
                rescored = rescore_hypotheses(
                    hyps, lm_scorer, args.lm_alpha, args.lm_beta,
                )
                decoded.append(rescored[0].text if rescored else "")
        else:
            decoded = [hyps[0].text if hyps else "" for hyps in batch_hyps]
    else:
        decoded = greedy_decode_batch(logits)
    return decoded


def save_results_csv(results_dir: Path, model_name: str, all_preds, all_refs, metrics):
    """Save per-trial predictions and summary metrics to CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{model_name}_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_idx", "prediction", "reference"])
        for i, (p, r) in enumerate(zip(all_preds, all_refs)):
            writer.writerow([i, p, r])

    logger.info("Saved predictions to %s", csv_path)

    # Also save summary metrics
    summary_path = results_dir / f"{model_name}_metrics.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    logger.info("Saved metrics summary to %s", summary_path)


def save_results_json(results_dir: Path, model_name: str, all_preds, all_refs, metrics):
    """Save per-trial predictions and summary metrics to JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{model_name}_results.json"

    output = {
        "metrics": metrics,
        "predictions": [
            {"trial_idx": i, "prediction": p, "reference": r}
            for i, (p, r) in enumerate(zip(all_preds, all_refs))
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved results to %s", json_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(yaml_path=args.config, preset="willett_handwriting")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Determine decoding method for logging
    if args.beam_width > 0:
        decoding_desc = f"beam search (width={args.beam_width})"
        if args.use_lm:
            decoding_desc += " + LM rescoring"
    else:
        decoding_desc = "greedy"

    logger.info("Evaluating model=%s, decoding=%s, device=%s", args.model, decoding_desc, device)

    # Load dataset
    from src.data.loader import load_willett_dataset
    trial_index = load_willett_dataset(cfg)

    # Filter by length if requested (must match training)
    if args.filter_by_length and "n_timesteps" in trial_index.columns:
        before = len(trial_index)
        trial_index = trial_index[trial_index["n_timesteps"] <= args.t_max].reset_index(drop=True)
        logger.info("Length filter (t_max=%d): %d -> %d trials", args.t_max, before, len(trial_index))

    train_df, _, test_df = split_trial_index(trial_index, cfg.split_ratios)

    # Build train dataset to get normalization stats, then test dataset
    if args.normalize:
        train_ds = NeuralTrialDataset(train_df, t_max=args.t_max, normalize=True)
        test_ds = NeuralTrialDataset(
            test_df, t_max=args.t_max, normalize=True,
            channel_mean=train_ds.channel_mean, channel_std=train_ds.channel_std,
        )
        logger.info("Using per-channel z-score normalization (from training set stats)")
    else:
        test_ds = NeuralTrialDataset(test_df, t_max=args.t_max)

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn
    )

    # Load model with architecture-specific hyperparameters
    model = build_model(args.model, args.n_channels, cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Model loaded: %s (%d params)", args.model, model.count_parameters())

    # Load LM scorer if requested
    lm_scorer = None
    if args.use_lm:
        if args.beam_width <= 0:
            logger.warning("--use-lm requires --beam-width > 0; falling back to greedy decoding")
        elif args.lm_path:
            lm_scorer = load_lm_scorer(args.lm_path, scorer_type=args.lm_type)
            if lm_scorer is not None:
                logger.info("Loaded LM scorer from %s", args.lm_path)
            else:
                logger.warning("Failed to load LM scorer from %s", args.lm_path)
        else:
            logger.warning("--use-lm specified but no --lm-path provided; skipping LM rescoring")

    # Inference
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            logits = model(features)
            decoded = decode_logits(logits, args, lm_scorer)
            all_preds.extend(decoded)
            all_refs.extend(batch["label_texts"])

    # Metrics
    cer = compute_cer(all_preds, all_refs)
    wer = compute_wer(all_preds, all_refs)
    em = exact_match_accuracy(all_preds, all_refs)

    logger.info("=== Test Results ===")
    logger.info("Model:        %s", args.model)
    logger.info("Decoding:     %s", decoding_desc)
    logger.info("CER:          %.4f (%.1f%%)", cer, cer * 100)
    logger.info("WER:          %.4f (%.1f%%)", wer, wer * 100)
    logger.info("Exact Match:  %.4f (%.1f%%)", em, em * 100)
    logger.info("Samples:      %d", len(all_preds))

    # Collect metrics dict
    metrics = {
        "model": args.model,
        "decoding": decoding_desc,
        "cer": round(cer, 6),
        "wer": round(wer, 6),
        "exact_match": round(em, 6),
        "n_samples": len(all_preds),
    }

    # Save results
    results_dir = Path(args.results_dir)
    if args.output_format == "json":
        save_results_json(results_dir, args.model, all_preds, all_refs, metrics)
    else:
        save_results_csv(results_dir, args.model, all_preds, all_refs, metrics)


if __name__ == "__main__":
    main()
