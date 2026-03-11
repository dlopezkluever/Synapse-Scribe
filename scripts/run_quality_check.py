#!/usr/bin/env python
"""Run signal quality diagnostics on a dataset.

Usage:
    python scripts/run_quality_check.py --dataset willett
    python scripts/run_quality_check.py --npy path/to/signals.npy --fs 250
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config, load_config
from src.diagnostics.report_generator import generate_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural recording quality check")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., 'willett'). Loads trials from the standard directory.",
    )
    parser.add_argument(
        "--npy",
        type=str,
        default=None,
        help="Path to a single .npy signal file [T, C] to analyze.",
    )
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling rate in Hz.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/quality_reports",
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="session_0",
        help="Session identifier for report naming.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.npy:
        # Direct file mode
        signals = np.load(args.npy)
        if signals.ndim != 2:
            print(f"Error: expected 2-D array, got shape {signals.shape}", file=sys.stderr)
            sys.exit(1)

        summary = generate_quality_report(
            signals=signals,
            trials=None,
            fs=args.fs,
            session_id=args.session_id,
            output_dir=args.output_dir,
        )

    elif args.dataset:
        # Dataset mode: load trial index and concatenate signals
        cfg = get_default_config()
        data_dir = Path(cfg.data_path) / cfg.dataset

        trial_index_path = data_dir / "trial_index.csv"
        if not trial_index_path.exists():
            print(
                f"Error: trial index not found at {trial_index_path}. "
                "Run the data loader first to create the trial index.",
                file=sys.stderr,
            )
            sys.exit(1)

        import pandas as pd

        trial_index = pd.read_csv(trial_index_path)
        trials = []
        for _, row in trial_index.iterrows():
            sig_path = row["signal_path"]
            if Path(sig_path).exists():
                trials.append(np.load(sig_path))

        if not trials:
            print("Error: no trial signal files found.", file=sys.stderr)
            sys.exit(1)

        # Use concatenated signals for channel-level analysis
        signals = np.concatenate(trials, axis=0)

        summary = generate_quality_report(
            signals=signals,
            trials=trials,
            fs=args.fs,
            session_id=args.session_id,
            output_dir=args.output_dir,
        )

    else:
        parser.print_help()
        print("\nError: specify either --dataset or --npy.", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print("\n=== Quality Report Summary ===")
    ch = summary.get("channel_quality", {})
    print(f"Channels: {ch.get('n_good', '?')}/{ch.get('n_total', '?')} good")

    snr = summary.get("snr", {})
    print(f"SNR: mean={snr.get('mean_snr_db', '?'):.1f} dB, {snr.get('n_low_quality', '?')} low-quality channels")

    spec = summary.get("spectral", {})
    print(f"Line noise: {'detected' if spec.get('line_noise_detected') else 'not detected'}")
    print(f"High gamma: {'present' if spec.get('high_gamma_present') else 'not detected'}")

    tq = summary.get("trial_quality")
    if tq:
        print(f"Trials: {tq.get('n_usable', '?')}/{tq.get('n_total', '?')} usable")

    corr = summary.get("correlation", {})
    print(f"High-correlation pairs: {corr.get('n_high_corr_pairs', '?')}")

    print(f"\nFull report saved to: {args.output_dir}/{args.session_id}/")


if __name__ == "__main__":
    main()
