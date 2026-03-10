"""Tests for src/config.py — configuration loading and validation."""

import pytest

from src.config import Config, load_config, get_default_config, PRESETS


class TestConfigDefaults:
    def test_default_config_loads(self):
        cfg = get_default_config()
        assert isinstance(cfg, Config)

    def test_default_dataset(self):
        cfg = get_default_config()
        assert cfg.dataset == "willett_handwriting"

    def test_default_split_ratios_sum_to_one(self):
        cfg = get_default_config()
        assert abs(sum(cfg.split_ratios) - 1.0) < 1e-6

    def test_default_bandpass(self):
        cfg = get_default_config()
        assert cfg.bandpass_low < cfg.bandpass_high


class TestConfigValidation:
    def test_bad_split_ratios_sum(self):
        with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
            load_config(overrides={"split_ratios": [0.5, 0.2, 0.1]})

    def test_bad_split_ratios_length(self):
        with pytest.raises(ValueError, match="exactly 3 elements"):
            load_config(overrides={"split_ratios": [0.5, 0.5]})

    def test_bad_bandpass(self):
        with pytest.raises(ValueError, match="bandpass_low must be < bandpass_high"):
            load_config(overrides={"bandpass_low": 300.0, "bandpass_high": 100.0})

    def test_bad_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            load_config(overrides={"model_type": "invalid_model"})

    def test_bad_dataset(self):
        with pytest.raises(ValueError, match="dataset"):
            load_config(overrides={"dataset": "nonexistent"})


class TestConfigPresets:
    def test_willett_preset(self):
        cfg = load_config(preset="willett_handwriting")
        assert cfg.bandpass_low == 1.0
        assert cfg.bandpass_high == 200.0
        assert cfg.t_max == 2000

    def test_ecog_preset(self):
        cfg = load_config(preset="ecog_speech")
        assert cfg.bandpass_low == 70.0
        assert cfg.bandpass_high == 150.0
        assert cfg.t_max == 4000
        assert cfg.dataset == "ucsf_ecog"

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_config(preset="fake_preset")


class TestConfigOverrides:
    def test_override_learning_rate(self):
        cfg = load_config(overrides={"learning_rate": 1e-5})
        assert cfg.learning_rate == 1e-5

    def test_override_after_preset(self):
        cfg = load_config(preset="ecog_speech", overrides={"t_max": 9999})
        assert cfg.t_max == 9999
        # Preset values still applied for other fields
        assert cfg.bandpass_low == 70.0


class TestConfigYaml:
    def test_save_and_reload(self, tmp_path):
        cfg = get_default_config()
        yaml_path = tmp_path / "test_config.yaml"
        cfg.save_yaml(yaml_path)

        cfg2 = load_config(yaml_path=yaml_path)
        assert cfg2.dataset == cfg.dataset
        assert cfg2.learning_rate == cfg.learning_rate
        assert cfg2.split_ratios == cfg.split_ratios
