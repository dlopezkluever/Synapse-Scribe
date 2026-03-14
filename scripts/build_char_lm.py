"""Build a character-level n-gram language model from training data.

Extracts text from the training split of the Willett dataset and builds
a smoothed character n-gram model saved as JSON. No external dependencies
beyond numpy/pandas.

Usage:
    python scripts/build_char_lm.py --order 5 --output models/char_lm_5gram.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.dataset import split_trial_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Special tokens for BOS/EOS
BOS = "<s>"
EOS = "</s>"


def extract_training_texts(cfg) -> list[str]:
    """Load training split texts from the Willett dataset."""
    from src.data.loader import load_willett_dataset

    trial_index = load_willett_dataset(cfg)
    train_df, _, _ = split_trial_index(trial_index, cfg.split_ratios)

    texts = []
    for _, row in train_df.iterrows():
        label_path = Path(row["label_path"])
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip().lower()
            if text:
                texts.append(text)

    logger.info("Extracted %d training texts", len(texts))
    return texts


def build_ngram_counts(texts: list[str], order: int) -> dict[str, dict[str, int]]:
    """Build n-gram counts for orders 1..order.

    Returns a dict mapping context string -> {next_char: count}.
    Context is a string of (order-1) characters for the highest order,
    with shorter contexts for backoff.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for text in texts:
        # Pad with BOS/EOS markers
        padded = [BOS] * (order - 1) + list(text) + [EOS]

        for n in range(1, order + 1):
            for i in range(n - 1, len(padded)):
                context = tuple(padded[i - n + 1 : i])
                char = padded[i]
                ctx_key = "|".join(context)
                counts[ctx_key][char] += 1

    logger.info(
        "Built n-gram counts: %d contexts, order=%d",
        len(counts), order,
    )
    return {k: dict(v) for k, v in counts.items()}


def build_lm(
    texts: list[str], order: int, smoothing: float = 0.01
) -> dict:
    """Build a smoothed character n-gram LM.

    Uses simple add-k (Lidstone) smoothing with Katz-style backoff weights.

    Returns a serializable dict with:
        - order: int
        - smoothing: float
        - vocab: list of all observed characters
        - counts: {context_str: {char: count}}
        - totals: {context_str: total_count}
    """
    counts = build_ngram_counts(texts, order)

    # Collect vocabulary (all observed characters + EOS)
    vocab = set()
    for char_counts in counts.values():
        vocab.update(char_counts.keys())
    vocab = sorted(vocab)

    # Compute totals per context
    totals = {}
    for ctx, char_counts in counts.items():
        totals[ctx] = sum(char_counts.values())

    logger.info("Vocabulary size: %d characters", len(vocab))
    logger.info("Total contexts: %d", len(counts))

    return {
        "order": order,
        "smoothing": smoothing,
        "vocab": vocab,
        "counts": counts,
        "totals": totals,
    }


def main():
    parser = argparse.ArgumentParser(description="Build character n-gram LM")
    parser.add_argument(
        "--order", type=int, default=5, help="N-gram order (default: 5)"
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.01,
        help="Lidstone smoothing constant (default: 0.01)",
    )
    parser.add_argument(
        "--output", type=str, default="models/char_lm_5gram.json",
        help="Output path for the LM JSON file",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML"
    )
    args = parser.parse_args()

    cfg = load_config(yaml_path=args.config, preset="willett_handwriting")
    texts = extract_training_texts(cfg)

    if not texts:
        logger.error("No training texts found. Check data path.")
        sys.exit(1)

    # Print sample texts
    logger.info("Sample texts:")
    for t in texts[:5]:
        logger.info("  '%s'", t)

    lm = build_lm(texts, order=args.order, smoothing=args.smoothing)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Saved LM to %s (%.2f MB)", output_path, size_mb)

    # Quick test: score a few strings
    from src.decoding.lm_correction import CharNgramScorer

    scorer = CharNgramScorer(str(output_path))
    test_strings = [
        "hello world",
        "the quick brown fox",
        "zzzzz xxxxx",
        "a",
        "",
    ]
    logger.info("Test scores:")
    for s in test_strings:
        score = scorer.score(s)
        logger.info("  '%s' -> %.4f", s, score)


if __name__ == "__main__":
    main()
