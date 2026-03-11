"""Tests for src/preprocessing/segment.py — trial segmentation and padding."""

import numpy as np
import pytest

from src.preprocessing.segment import (
    segment_trials,
    pad_or_truncate,
    pad_or_truncate_batch,
)


class TestSegmentTrials:
    def test_basic_segmentation(self):
        signal = np.random.randn(10000, 10).astype(np.float32)
        onsets = np.array([100, 3000, 6000])
        offsets = np.array([500, 3500, 6800])
        trials = segment_trials(signal, onsets, offsets, fs=250.0)
        assert len(trials) == 3
        for trial in trials:
            assert trial.ndim == 2
            assert trial.shape[1] == 10

    def test_padding_applied(self):
        signal = np.random.randn(10000, 5).astype(np.float32)
        onsets = np.array([1000])
        offsets = np.array([2000])
        trials = segment_trials(signal, onsets, offsets, fs=250.0, pre_pad_ms=100, post_pad_ms=200)
        # pre_pad = 25 samples, post_pad = 50 samples
        # Duration = (2000 - 1000 + 25 + 50) = 1075
        expected_len = (2000 - 1000) + 25 + 50
        assert trials[0].shape[0] == expected_len

    def test_boundary_clamping(self):
        """Onset near start of signal should clamp to 0."""
        signal = np.random.randn(500, 3).astype(np.float32)
        onsets = np.array([5])
        offsets = np.array([100])
        trials = segment_trials(signal, onsets, offsets, fs=250.0, pre_pad_ms=100)
        assert trials[0].shape[0] > 0


class TestPadOrTruncate:
    def test_truncate(self):
        signal = np.random.randn(3000, 10).astype(np.float32)
        result = pad_or_truncate(signal, t_max=2000)
        assert result.shape == (2000, 10)

    def test_pad(self):
        signal = np.random.randn(500, 10).astype(np.float32)
        result = pad_or_truncate(signal, t_max=2000)
        assert result.shape == (2000, 10)
        # Padded region should be zeros
        np.testing.assert_array_equal(result[500:], 0.0)

    def test_exact_length(self):
        signal = np.random.randn(2000, 10).astype(np.float32)
        result = pad_or_truncate(signal, t_max=2000)
        np.testing.assert_array_equal(result, signal)


class TestPadOrTruncateBatch:
    def test_batch_output_shape(self):
        signals = [
            np.random.randn(100, 5).astype(np.float32),
            np.random.randn(200, 5).astype(np.float32),
            np.random.randn(300, 5).astype(np.float32),
        ]
        batch, lengths = pad_or_truncate_batch(signals, t_max=250)
        assert batch.shape == (3, 250, 5)
        assert lengths.shape == (3,)
        np.testing.assert_array_equal(lengths, [100, 200, 250])

    def test_batch_padding_is_zero(self):
        signals = [np.ones((50, 3), dtype=np.float32)]
        batch, _ = pad_or_truncate_batch(signals, t_max=100)
        np.testing.assert_array_equal(batch[0, 50:], 0.0)
