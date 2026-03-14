"""Run beam search + LM evaluation across multiple decoding configurations.

Evaluates a trained model with:
  1. Greedy decoding (baseline)
  2. Beam search at widths 5, 10, 25, 50
  3. Beam search + character n-gram LM rescoring at best beam width

Automatically builds the character LM if it doesn't exist.

Usage:
    python scripts/run_beam_eval.py \
        --checkpoint outputs/checkpoints/GPU-3-13/CNNLSTM_best.pt \
        --model cnn_lstm

    python scripts/run_beam_eval.py \
        --checkpoint outputs/checkpoints/GPU-3-13/CNNTransformer_best.pt \
        --model cnn_transformer \
        --beam-widths 5 10 25 50 100 \
        --lm-alphas 0.1 0.3 0.5 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import (
    NeuralTrialDataset, ctc_collate_fn, split_trial_index,
)
from src.data.loader import load_willett_dataset
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
    parser = argparse.ArgumentParser(
        description="Comprehensive beam search + LM evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--n-channels", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--t-max", type=int, default=2000)
    parser.add_argument("--normalize", action="store_true",
                        help="Z-score normalize per channel (must match training)")
    parser.add_argument("--filter-by-length", action="store_true",
                        help="Drop trials longer than t-max")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Beam search grid
    parser.add_argument(
        "--beam-widths", type=int, nargs="+", default=[5, 10, 25, 50],
        help="Beam widths to evaluate (default: 5 10 25 50)",
    )

    # LM options
    parser.add_argument(
        "--lm-path", type=str, default="models/char_lm_5gram.json",
        help="Path to character n-gram LM (built automatically if missing)",
    )
    parser.add_argument(
        "--lm-order", type=int, default=5,
        help="N-gram order for auto-built LM (default: 5)",
    )
    parser.add_argument(
        "--lm-alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7],
        help="LM alpha weights to sweep (default: 0.1 0.3 0.5 0.7)",
    )
    parser.add_argument(
        "--lm-betas", type=float, nargs="+", default=[0.0, 1.0, 2.0],
        help="Word insertion bonus values to sweep (default: 0.0 1.0 2.0)",
    )

    # Output
    parser.add_argument(
        "--results-dir", type=str, default="./outputs/results",
        help="Directory to save results",
    )

    return parser.parse_args()


def build_model(model_name: str, n_channels: int, cfg) -> torch.nn.Module:
    """Instantiate a model with architecture-specific hyperparameters."""
    ModelClass = MODEL_REGISTRY[model_name]

    if model_name == "gru_decoder":
        return ModelClass(n_channels=n_channels, n_classes=cfg.n_classes)
    elif model_name == "cnn_lstm":
        return ModelClass(
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
        return ModelClass(
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
        return ModelClass(
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
        return ModelClass(n_channels=n_channels, n_classes=cfg.n_classes)


def run_inference(model, test_loader, device) -> tuple[list, list, list]:
    """Run model inference and cache logits.

    Returns (all_logits, all_refs, all_greedy_preds).
    """
    all_logits = []
    all_refs = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            logits = model(features)
            # Store logits on CPU for decoding
            all_logits.append(logits.cpu())
            all_refs.extend(batch["label_texts"])

    return all_logits, all_refs


def decode_greedy(all_logits) -> list[str]:
    """Greedy decode all cached logits."""
    preds = []
    for logits in all_logits:
        preds.extend(greedy_decode_batch(logits))
    return preds


def decode_beam(all_logits, beam_width: int) -> tuple[list[str], list]:
    """Beam search decode. Returns (best_texts, all_hypotheses)."""
    preds = []
    all_hyps = []
    for logits in all_logits:
        batch_hyps = beam_search_decode_batch(
            logits, beam_width=beam_width, top_k=10,
        )
        preds.extend([h[0].text if h else "" for h in batch_hyps])
        all_hyps.extend(batch_hyps)
    return preds, all_hyps


def decode_beam_lm(
    all_hyps: list, lm_scorer, alpha: float, beta: float
) -> list[str]:
    """Rescore cached beam hypotheses with LM."""
    preds = []
    for hyps in all_hyps:
        rescored = rescore_hypotheses(hyps, lm_scorer, alpha, beta)
        preds.append(rescored[0].text if rescored else "")
    return preds


def compute_metrics(preds: list[str], refs: list[str]) -> dict:
    """Compute CER, WER, exact match."""
    return {
        "cer": compute_cer(preds, refs),
        "wer": compute_wer(preds, refs),
        "exact_match": exact_match_accuracy(preds, refs),
    }


def ensure_lm_exists(lm_path: str, order: int, cfg) -> str:
    """Build the character LM if it doesn't exist yet."""
    lm_file = Path(lm_path)
    if lm_file.exists():
        logger.info("Character LM found at %s", lm_file)
        return str(lm_file)

    logger.info("Building character n-gram LM (order=%d)...", order)
    from scripts.build_char_lm import extract_training_texts, build_lm

    texts = extract_training_texts(cfg)
    lm = build_lm(texts, order=order)

    lm_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lm_file, "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False)

    logger.info("Saved character LM to %s", lm_file)
    return str(lm_file)


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 90)
    print(f"{'Decoding Method':<40} {'CER':>8} {'WER':>8} {'EM':>8} {'Time':>10}")
    print("-" * 90)
    for r in results:
        cer_pct = f"{r['cer']*100:.2f}%"
        wer_pct = f"{r['wer']*100:.2f}%"
        em_pct = f"{r['exact_match']*100:.1f}%"
        time_s = f"{r['time']:.1f}s"
        print(f"{r['method']:<40} {cer_pct:>8} {wer_pct:>8} {em_pct:>8} {time_s:>10}")
    print("=" * 90)


def print_sample_predictions(
    greedy_preds: list[str],
    best_preds: list[str],
    refs: list[str],
    n: int = 10,
) -> None:
    """Show side-by-side greedy vs best predictions."""
    print(f"\n{'='*90}")
    print("Sample Predictions (greedy vs best beam+LM)")
    print(f"{'='*90}")
    for i in range(min(n, len(refs))):
        print(f"\n[{i}] Reference:  '{refs[i]}'")
        print(f"    Greedy:     '{greedy_preds[i]}'")
        print(f"    Best:       '{best_preds[i]}'")
    print()


def main() -> None:
    args = parse_args()
    cfg = load_config(yaml_path=args.config, preset="willett_handwriting")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("Device: %s", device)
    logger.info("Model: %s", args.model)
    logger.info("Checkpoint: %s", args.checkpoint)

    # Load data
    trial_index = load_willett_dataset(cfg)

    # Filter by length if requested (must match training)
    if args.filter_by_length and "n_timesteps" in trial_index.columns:
        before = len(trial_index)
        trial_index = trial_index[trial_index["n_timesteps"] <= args.t_max].reset_index(drop=True)
        logger.info("Length filter (t_max=%d): %d -> %d trials", args.t_max, before, len(trial_index))

    train_df, _, test_df = split_trial_index(trial_index, cfg.split_ratios)

    # Match normalization from training
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
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn,
    )
    logger.info("Test set: %d samples", len(test_ds))

    # Load model
    model = build_model(args.model, args.n_channels, cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded: %d params", model.count_parameters())

    # Run inference once (cache logits)
    logger.info("Running inference...")
    t0 = time.time()
    all_logits, all_refs = run_inference(model, test_loader, device)
    inference_time = time.time() - t0
    logger.info("Inference done in %.1fs (%d samples)", inference_time, len(all_refs))

    results = []

    # 1. Greedy decoding
    logger.info("Decoding: greedy")
    t0 = time.time()
    greedy_preds = decode_greedy(all_logits)
    greedy_time = time.time() - t0
    greedy_metrics = compute_metrics(greedy_preds, all_refs)
    results.append({
        "method": "Greedy",
        **greedy_metrics,
        "time": greedy_time,
    })
    logger.info("  CER=%.4f WER=%.4f EM=%.4f (%.1fs)",
                greedy_metrics["cer"], greedy_metrics["wer"],
                greedy_metrics["exact_match"], greedy_time)

    # 2. Beam search at various widths
    best_beam_width = None
    best_beam_cer = float("inf")
    best_beam_hyps = None

    for bw in args.beam_widths:
        logger.info("Decoding: beam search (width=%d)", bw)
        t0 = time.time()
        beam_preds, beam_hyps = decode_beam(all_logits, beam_width=bw)
        beam_time = time.time() - t0
        beam_metrics = compute_metrics(beam_preds, all_refs)

        results.append({
            "method": f"Beam (w={bw})",
            **beam_metrics,
            "time": beam_time,
        })
        logger.info("  CER=%.4f WER=%.4f EM=%.4f (%.1fs)",
                    beam_metrics["cer"], beam_metrics["wer"],
                    beam_metrics["exact_match"], beam_time)

        if beam_metrics["cer"] < best_beam_cer:
            best_beam_cer = beam_metrics["cer"]
            best_beam_width = bw
            best_beam_hyps = beam_hyps

    # 3. Beam search + LM rescoring (sweep alpha and beta)
    lm_path = ensure_lm_exists(args.lm_path, args.lm_order, cfg)
    lm_scorer = load_lm_scorer(lm_path, scorer_type="char_ngram")

    logger.info("Sweeping LM params with best beam width=%d", best_beam_width)

    best_lm_preds = None
    best_lm_cer = float("inf")

    for alpha in args.lm_alphas:
        for beta in args.lm_betas:
            t0 = time.time()
            lm_preds = decode_beam_lm(best_beam_hyps, lm_scorer, alpha, beta)
            lm_time = time.time() - t0
            lm_metrics = compute_metrics(lm_preds, all_refs)

            method_name = f"Beam (w={best_beam_width}) + LM (a={alpha}, b={beta})"
            results.append({
                "method": method_name,
                **lm_metrics,
                "time": lm_time,
            })
            logger.info("  %s: CER=%.4f WER=%.4f EM=%.4f",
                        method_name, lm_metrics["cer"],
                        lm_metrics["wer"], lm_metrics["exact_match"])

            if lm_metrics["cer"] < best_lm_cer:
                best_lm_cer = lm_metrics["cer"]
                best_lm_preds = lm_preds

    # Print summary
    print_results_table(results)

    # Show sample predictions
    best_overall = best_lm_preds if best_lm_preds is not None else greedy_preds
    print_sample_predictions(greedy_preds, best_overall, all_refs)

    # Save full results as JSON
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{args.model}_beam_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)

    # Save detailed predictions for best config
    detail_path = results_dir / f"{args.model}_beam_predictions.json"
    detail = {
        "greedy": [
            {"ref": r, "pred": p} for r, p in zip(all_refs, greedy_preds)
        ],
        "best_beam_lm": [
            {"ref": r, "pred": p} for r, p in zip(all_refs, best_overall)
        ] if best_lm_preds else [],
    }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    logger.info("Saved detailed predictions to %s", detail_path)


if __name__ == "__main__":
    main()
