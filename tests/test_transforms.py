"""Tests for src/data/transforms.py — data augmentation."""

import numpy as np
import pytest

from src.data.transforms import (
    TimeMasking,
    ChannelDropout,
    GaussianNoise,
    Compose,
    get_training_transforms,
)


@pytest.fixture
def sample_signal():
    np.random.seed(42)
    return np.random.randn(500, 32).astype(np.float32)


class TestTimeMasking:
    def test_shape_preserved(self, sample_signal):
        aug = TimeMasking(n_masks=3, max_mask_ms=50.0, fs=250.0)
        result = aug(sample_signal)
        assert result.shape == sample_signal.shape

    def test_modifies_data(self, sample_signal):
        aug = TimeMasking(n_masks=3, max_mask_ms=50.0, fs=250.0)
        np.random.seed(123)
        result = aug(sample_signal)
        # Should have some zero regions
        assert not np.array_equal(result, sample_signal)
        # At least some values should be zero
        assert np.any(result == 0.0)

    def test_does_not_modify_original(self, sample_signal):
        original = sample_signal.copy()
        aug = TimeMasking()
        _ = aug(sample_signal)
        np.testing.assert_array_equal(sample_signal, original)


class TestChannelDropout:
    def test_shape_preserved(self, sample_signal):
        aug = ChannelDropout(dropout_rate=0.1)
        result = aug(sample_signal)
        assert result.shape == sample_signal.shape

    def test_modifies_data(self, sample_signal):
        aug = ChannelDropout(dropout_rate=0.1)
        np.random.seed(123)
        result = aug(sample_signal)
        assert not np.array_equal(result, sample_signal)

    def test_zeros_entire_channels(self, sample_signal):
        aug = ChannelDropout(dropout_rate=0.5)
        np.random.seed(42)
        result = aug(sample_signal)
        # Some channels should be all zero
        channel_sums = np.abs(result).sum(axis=0)
        assert np.any(channel_sums == 0.0)


class TestGaussianNoise:
    def test_shape_preserved(self, sample_signal):
        aug = GaussianNoise(std=0.01)
        result = aug(sample_signal)
        assert result.shape == sample_signal.shape

    def test_modifies_data(self, sample_signal):
        aug = GaussianNoise(std=0.01)
        np.random.seed(0)
        result = aug(sample_signal)
        assert not np.array_equal(result, sample_signal)

    def test_noise_magnitude(self, sample_signal):
        aug = GaussianNoise(std=0.01)
        np.random.seed(0)
        result = aug(sample_signal)
        diff = result - sample_signal
        # Noise std should be approximately 0.01
        assert abs(np.std(diff) - 0.01) < 0.005


class TestCompose:
    def test_applies_all_transforms(self, sample_signal):
        transforms = Compose([
            GaussianNoise(std=0.01),
            ChannelDropout(dropout_rate=0.1),
        ])
        np.random.seed(42)
        result = transforms(sample_signal)
        assert result.shape == sample_signal.shape
        assert not np.array_equal(result, sample_signal)


class TestGetTrainingTransforms:
    def test_creates_pipeline(self, sample_signal):
        transform = get_training_transforms()
        np.random.seed(42)
        result = transform(sample_signal)
        assert result.shape == sample_signal.shape
