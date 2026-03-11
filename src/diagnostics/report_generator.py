"""Quality report generator that runs all diagnostics and produces a structured report.

Outputs a summary JSON and saves all diagnostic plots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from src.diagnostics.channel_quality import (
    detect_bad_channels,
    plot_channel_variance_heatmap,
    ChannelQualityResult,
)
from src.diagnostics.snr_analysis import (
    compute_snr,
    plot_snr_distribution,
    SNRResult,
)
from src.diagnostics.spectral_analysis import (
    compute_psd,
    plot_power_spectrum,
    SpectralResult,
)
from src.diagnostics.trial_quality import (
    detect_bad_trials,
    plot_trial_quality_histogram,
    TrialQualityResult,
)
from src.diagnostics.correlation_analysis import (
    compute_channel_correlation,
    plot_correlation_matrix,
    CorrelationResult,
)

logger = logging.getLogger(__name__)


def generate_quality_report(
    signals: np.ndarray,
    trials: Optional[list[np.ndarray]] = None,
    fs: float = 250.0,
    session_id: str = "session_0",
    output_dir: str | Path = "./outputs/quality_reports",
    save_plots: bool = True,
) -> dict:
    """Run all diagnostics and produce a structured quality report.

    Args:
        signals: Concatenated signals [T, C] for channel-level analyses (can be a
            single representative trial or the concatenation of all trials).
        trials: Optional list of per-trial arrays [T_i, C] for trial-level analysis.
            If None, trial quality is skipped.
        fs: Sampling rate in Hz.
        session_id: Identifier for the session (used in output paths).
        output_dir: Root directory for quality reports.
        save_plots: Whether to save diagnostic plots to disk.

    Returns:
        Summary dict with all metrics.
    """
    output_dir = Path(output_dir) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"session_id": session_id, "fs": fs}

    # --- 1. Channel quality ---
    logger.info("Running channel quality detection...")
    ch_result = detect_bad_channels(signals, fs=fs)
    summary["channel_quality"] = {
        "n_total": ch_result.n_total,
        "n_good": ch_result.n_good,
        "n_bad": ch_result.n_bad,
        "bad_indices": ch_result.bad_indices,
        "reasons": {str(k): v for k, v in ch_result.reasons.items()},
    }

    # --- 2. SNR analysis ---
    logger.info("Running SNR analysis...")
    snr_result = compute_snr(signals, fs=fs)
    summary["snr"] = {
        "mean_snr_db": float(np.mean(snr_result.snr_per_channel)),
        "median_snr_db": float(np.median(snr_result.snr_per_channel)),
        "n_low_quality": len(snr_result.low_quality_indices),
        "threshold_db": snr_result.threshold_db,
    }

    # --- 3. Spectral analysis ---
    logger.info("Running spectral analysis...")
    spec_result = compute_psd(signals, fs=fs)
    summary["spectral"] = {
        "line_noise_detected": spec_result.line_noise_detected,
        "line_noise_power": spec_result.line_noise_power,
        "high_gamma_present": spec_result.high_gamma_present,
        "high_gamma_power": spec_result.high_gamma_power,
    }

    # --- 4. Trial quality ---
    trial_result = None
    if trials is not None and len(trials) > 0:
        logger.info("Running trial quality detection...")
        trial_result = detect_bad_trials(trials)
        summary["trial_quality"] = {
            "n_total": trial_result.n_total,
            "n_usable": trial_result.n_usable,
            "n_rejected": trial_result.n_rejected,
            "rejected_indices": trial_result.rejected_indices,
            "reasons": {str(k): v for k, v in trial_result.reasons.items()},
        }
    else:
        summary["trial_quality"] = None

    # --- 5. Channel correlation ---
    logger.info("Running channel correlation analysis...")
    corr_result = compute_channel_correlation(signals)
    summary["correlation"] = {
        "n_high_corr_pairs": corr_result.n_high_corr_pairs,
        "threshold": corr_result.threshold,
        "high_corr_pairs": [
            {"ch_i": p[0], "ch_j": p[1], "corr": p[2]}
            for p in corr_result.high_corr_pairs
        ],
    }

    # --- Save summary JSON ---
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary to %s", summary_path)

    # --- Save plots ---
    if save_plots:
        _save_plots(output_dir, ch_result, snr_result, spec_result, trial_result, corr_result)

    return summary


def _save_plots(
    output_dir: Path,
    ch_result: ChannelQualityResult,
    snr_result: SNRResult,
    spec_result: SpectralResult,
    trial_result: Optional[TrialQualityResult],
    corr_result: CorrelationResult,
) -> None:
    """Save all diagnostic plots to disk."""
    dpi = 150

    fig = plot_channel_variance_heatmap(ch_result)
    fig.savefig(output_dir / "channel_variance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig = plot_snr_distribution(snr_result)
    fig.savefig(output_dir / "snr_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig = plot_power_spectrum(spec_result)
    fig.savefig(output_dir / "power_spectrum.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if trial_result is not None:
        fig = plot_trial_quality_histogram(trial_result)
        fig.savefig(output_dir / "trial_quality_histogram.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    fig = plot_correlation_matrix(corr_result)
    fig.savefig(output_dir / "channel_correlation_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved diagnostic plots to %s", output_dir)
