"""Neural recording diagnostics and signal quality control."""

from src.diagnostics.channel_quality import detect_bad_channels
from src.diagnostics.snr_analysis import compute_snr
from src.diagnostics.spectral_analysis import compute_psd
from src.diagnostics.trial_quality import detect_bad_trials
from src.diagnostics.correlation_analysis import compute_channel_correlation
from src.diagnostics.report_generator import generate_quality_report

__all__ = [
    "detect_bad_channels",
    "compute_snr",
    "compute_psd",
    "detect_bad_trials",
    "compute_channel_correlation",
    "generate_quality_report",
]
