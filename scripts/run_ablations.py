#!/usr/bin/env python
"""Run ablation studies across models and decoding strategies.

Usage:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --models gru_decoder transformer
    python scripts/run_ablations.py --models cnn_lstm --decoding greedy beam
    python scripts/run_ablations.py --output-dir ./outputs/results/ablations --beam-width 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import (
    NeuralTrialDataset, ctc_collate_fn, split_trial_index,
)
from src.evaluation.ablations import run_ablation_suite, run_significance_tests
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

ALL_MODELS = list(MODEL_REGISTRY.keys())
ALL_DECODING = ["greedy", "beam", "beam+lm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation studies across models and decoding strategies",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help=f"Models to evaluate (default: all). Choices: {ALL_MODELS}",
    )
    parser.add_argument(
        "--decoding", type=str, nargs="+", default=["greedy", "beam"],
        choices=ALL_DECODING,
        help="Decoding strategies to test (default: greedy beam)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/results",
        help="Directory to save ablation results (default: ./outputs/results)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./outputs/checkpoints",
        help="Directory containing model checkpoints (default: ./outputs/checkpoints)",
    )
    parser.add_argument(
        "--n-channels", type=int, default=192,
        help="Number of input channels (default: 192)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--t-max", type=int, default=2000,
        help="Maximum time steps per trial (default: 2000)",
    )
    parser.add_argument(
        "--beam-width", type=int, default=100,
        help="Beam width for beam search (default: 100)",
    )
    parser.add_argument(
        "--lm-path", type=str, default=None,
        help="Path to KenLM model file (required for beam+lm decoding)",
    )
    parser.add_argument(
        "--lm-alpha", type=float, default=0.5,
        help="LM interpolation weight (default: 0.5)",
    )
    parser.add_argument(
        "--lm-beta", type=float, default=0.0,
        help="Word insertion bonus (default: 0.0)",
    )
    parser.add_argument(
        "--significance-test", action="store_true",
        help="Run paired bootstrap significance tests between models",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=1000,
        help="Number of bootstrap resamples for significance tests (default: 1000)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect)",
    )
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


# Checkpoint naming convention used by the Trainer
_CHECKPOINT_NAMES = {
    "gru_decoder": "GRUDecoder_best.pt",
    "cnn_lstm": "CNNLSTM_best.pt",
    "transformer": "TransformerDecoder_best.pt",
    "cnn_transformer": "CNNTransformer_best.pt",
}


def load_models(
    model_names: list[str],
    n_channels: int,
    cfg,
    checkpoint_dir: str,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    """Load trained models from checkpoints.

    Falls back to freshly initialized models if no checkpoint is found
    (useful for quick ablation runs / testing the pipeline).
    """
    checkpoint_dir = Path(checkpoint_dir)
    models = {}

    for name in model_names:
        model = build_model(name, n_channels, cfg)
        ckpt_name = _CHECKPOINT_NAMES.get(name, f"{name}_best.pt")
        ckpt_path = checkpoint_dir / ckpt_name

        if ckpt_path.exists():
            logger.info("Loading checkpoint for %s from %s", name, ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning(
                "No checkpoint found at %s — using randomly initialized %s "
                "(results will not be meaningful)",
                ckpt_path, name,
            )

        model.to(device)
        model.eval()
        models[name] = model

    return models


def print_comparison_table(results: list) -> None:
    """Print a formatted comparison table of ablation results."""
    if not results:
        return

    # Header
    header = f"{'Model':<20} {'Decoding':<12} {'CER':>8} {'WER':>8} {'EM':>8} {'Time (s)':>10} {'Samples':>8}"
    sep = "-" * len(header)

    print("\n" + sep)
    print("  ABLATION RESULTS — Model x Decoding Comparison")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        row = (
            f"{r['model_name']:<20} "
            f"{r['decoding_method']:<12} "
            f"{r['cer']:>8.4f} "
            f"{r['wer']:>8.4f} "
            f"{r['exact_match']:>8.4f} "
            f"{r['inference_time_s']:>10.2f} "
            f"{r['n_samples']:>8d}"
        )
        print(row)

    print(sep)

    # Find best CER
    if results:
        best = min(results, key=lambda r: r["cer"])
        print(
            f"\n  Best CER: {best['cer']:.4f} "
            f"({best['model_name']} / {best['decoding_method']})"
        )
    print("")


def main() -> None:
    args = parse_args()
    cfg = load_config(yaml_path=args.config, preset="willett_handwriting")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("=== BCI-2 Ablation Study ===")
    logger.info("Models:   %s", ", ".join(args.models))
    logger.info("Decoding: %s", ", ".join(args.decoding))
    logger.info("Device:   %s", device)
    logger.info("Output:   %s", args.output_dir)

    # Warn if beam+lm requested without LM path
    if "beam+lm" in args.decoding and args.lm_path is None:
        logger.warning(
            "beam+lm decoding requested but no --lm-path provided. "
            "Beam search will run without LM rescoring."
        )

    # Load dataset
    from src.data.loader import load_willett_dataset
    logger.info("Loading test dataset...")
    trial_index = load_willett_dataset(cfg)
    _, _, test_df = split_trial_index(trial_index, cfg.split_ratios)

    test_ds = NeuralTrialDataset(test_df, t_max=args.t_max)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn
    )
    logger.info("Test set: %d samples", len(test_ds))

    # Load models
    models = load_models(
        args.models, args.n_channels, cfg, args.checkpoint_dir, device,
    )

    # Run ablation suite
    logger.info("Running ablation suite...")
    results = run_ablation_suite(
        models=models,
        dataloader=test_loader,
        decoding_methods=args.decoding,
        lm_model_path=args.lm_path,
        beam_width=args.beam_width,
        lm_alpha=args.lm_alpha,
        lm_beta=args.lm_beta,
        device=device,
        output_dir=args.output_dir,
    )

    # Print comparison table
    results_dicts = [r.to_dict() for r in results]
    print_comparison_table(results_dicts)

    # Save results to JSON (the ablation suite already exports, but we also
    # save the full structured output for downstream consumption)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dicts, f, indent=2)
    logger.info("Saved ablation results to %s", json_path)

    # Optional: significance tests
    if args.significance_test and len(args.models) >= 2:
        logger.info("Running paired bootstrap significance tests (n=%d)...", args.n_resamples)
        sig_results = run_significance_tests(results, n_resamples=args.n_resamples)

        if sig_results:
            print("\n  SIGNIFICANCE TESTS (Paired Bootstrap)")
            print("  " + "-" * 70)
            for sr in sig_results:
                sig_marker = "*" if sr.significant else " "
                print(
                    f"  {sig_marker} {sr.model_a} vs {sr.model_b} ({sr.metric}): "
                    f"diff={sr.mean_diff:+.4f}, "
                    f"CI=[{sr.ci_lower:.4f}, {sr.ci_upper:.4f}], "
                    f"p={sr.p_value:.4f}"
                )
            print("  " + "-" * 70)
            print("  * = significant at p < 0.05\n")

            # Save significance results
            sig_path = output_dir / "significance_results.json"
            sig_data = [
                {
                    "model_a": sr.model_a,
                    "model_b": sr.model_b,
                    "metric": sr.metric,
                    "mean_diff": sr.mean_diff,
                    "ci_lower": sr.ci_lower,
                    "ci_upper": sr.ci_upper,
                    "p_value": sr.p_value,
                    "n_resamples": sr.n_resamples,
                    "significant": sr.significant,
                }
                for sr in sig_results
            ]
            with open(sig_path, "w", encoding="utf-8") as f:
                json.dump(sig_data, f, indent=2)
            logger.info("Saved significance results to %s", sig_path)

    logger.info("Ablation study complete.")


if __name__ == "__main__":
    main()
