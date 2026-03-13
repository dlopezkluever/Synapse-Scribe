"""Global configuration module for the Brain-Text Decoder.

Supports loading from defaults, YAML overrides, and dataset-specific presets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All configurable parameters for the pipeline."""

    # --- dataset ---
    dataset: str = "willett_handwriting"
    subjects: List[int] = field(default_factory=lambda: [1])
    data_path: str = "./data"
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])

    # --- preprocessing ---
    bandpass_low: float = 1.0
    bandpass_high: float = 200.0
    notch_freq: float = 60.0
    target_fs: float = 250.0
    t_max: int = 2000
    zscore_clip: float = 5.0
    artifact_threshold: float = 3.0
    bad_channel_var_threshold: float = 10.0

    # --- firing rate binning (Pathway C, Willett) ---
    use_firing_rates: bool = False
    bin_width_ms: float = 10.0

    # --- model ---
    model_type: str = "cnn_lstm"  # cnn_lstm | transformer | cnn_transformer
    n_classes: int = 28  # blank + a-z + space

    # CNN+LSTM (Model A)
    conv_channels: int = 256
    conv_kernel_size: int = 7
    conv_layers: int = 3
    lstm_hidden: int = 512
    lstm_layers: int = 2
    lstm_dropout: float = 0.5

    # Transformer (Model B)
    d_model: int = 512
    n_heads: int = 8
    transformer_layers: int = 6
    ffn_dim: int = 2048
    transformer_dropout: float = 0.1
    max_seq_len: int = 4096

    # Hybrid CNN-Transformer (Model C)
    hybrid_cnn_layers: int = 3
    hybrid_transformer_layers: int = 4

    # --- training ---
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 200
    warmup_steps: int = 500
    grad_clip_max_norm: float = 1.0
    early_stopping_patience: int = 20
    mixed_precision: bool = True

    # --- augmentation ---
    aug_time_mask_n: int = 3
    aug_time_mask_max_ms: float = 50.0
    aug_channel_dropout_rate: float = 0.2
    aug_gaussian_noise_std: float = 0.1

    # --- W&B experiment tracking ---
    wandb_enabled: bool = False
    wandb_project: str = "brain-text-decoder"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_log_interval: int = 1  # log every N epochs

    # --- GPT-2 LM re-ranking ---
    gpt2_model_name: str = "gpt2"  # HuggingFace model ID
    gpt2_lambda: float = 0.5  # interpolation weight: λ*CTC + (1-λ)*LM

    # --- paths ---
    checkpoint_dir: str = "./outputs/checkpoints"
    results_dir: str = "./outputs/results"
    figures_dir: str = "./outputs/figures"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError if the config is invalid."""
        # Split ratios must sum to 1
        total = sum(self.split_ratios)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split_ratios must sum to 1.0, got {self.split_ratios} (sum={total})"
            )
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must have exactly 3 elements [train, val, test]")

        # Frequency sanity
        if self.bandpass_low >= self.bandpass_high:
            raise ValueError("bandpass_low must be < bandpass_high")
        if self.target_fs <= 0:
            raise ValueError("target_fs must be positive")

        # Model type
        valid_models = {"gru_decoder", "cnn_lstm", "transformer", "cnn_transformer"}
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")

        # Dataset
        valid_datasets = {"willett_handwriting", "openneuro", "ucsf_ecog"}
        if self.dataset not in valid_datasets:
            raise ValueError(f"dataset must be one of {valid_datasets}")

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "willett_handwriting": {
        "dataset": "willett_handwriting",
        "bandpass_low": 1.0,
        "bandpass_high": 200.0,
        "target_fs": 250.0,
        "t_max": 2000,
    },
    "ecog_speech": {
        "dataset": "ucsf_ecog",
        "bandpass_low": 70.0,
        "bandpass_high": 150.0,
        "target_fs": 200.0,
        "t_max": 4000,
    },
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: Config, overrides: dict) -> None:
    """Apply a dict of overrides to a Config, ignoring unknown keys."""
    valid_names = {f.name for f in fields(Config)}
    for key, value in overrides.items():
        if key in valid_names:
            setattr(cfg, key, value)


def load_config(
    yaml_path: Optional[str | Path] = None,
    preset: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> Config:
    """Build a Config from defaults → preset → YAML file → explicit overrides.

    Layer priority (last wins): defaults < preset < YAML < overrides.
    """
    cfg = Config()

    # Apply preset if requested
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
        _apply_overrides(cfg, PRESETS[preset])

    # Apply YAML file
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
            _apply_overrides(cfg, yaml_data)

    # Apply explicit overrides
    if overrides:
        _apply_overrides(cfg, overrides)

    cfg.validate()
    return cfg


def get_default_config() -> Config:
    """Return validated default config (Willett preset)."""
    return load_config(preset="willett_handwriting")
