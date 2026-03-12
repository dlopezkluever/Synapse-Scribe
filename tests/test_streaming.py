"""Tests for real-time streaming inference (src/inference/streaming.py).

Verifies the StreamingBuffer, StreamingDecoder, latency tracking,
and simulate_streaming helper using a lightweight GRU model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference.streaming import (
    DecoderUpdate,
    LatencyStats,
    StreamingBuffer,
    StreamingDecoder,
    simulate_streaming,
)
from src.models.gru_decoder import GRUDecoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """Small GRU model for fast tests (4 channels, 28 classes)."""
    return GRUDecoder(
        n_channels=4, n_classes=28, proj_dim=8,
        hidden_size=16, n_layers=1, dropout=0.0,
    )


@pytest.fixture
def trial_data():
    """Synthetic trial: 200 timesteps, 4 channels."""
    rng = np.random.RandomState(42)
    return rng.randn(200, 4).astype(np.float32)


# ---------------------------------------------------------------------------
# StreamingBuffer
# ---------------------------------------------------------------------------

class TestStreamingBuffer:
    def test_initial_state(self):
        buf = StreamingBuffer(n_channels=4, max_length=100)
        assert buf.length == 0
        assert buf.total_samples_received == 0
        assert not buf.is_full

    def test_feed_increases_length(self):
        buf = StreamingBuffer(n_channels=4, max_length=100)
        buf.feed(np.zeros((10, 4), dtype=np.float32))
        assert buf.length == 10
        assert buf.total_samples_received == 10

    def test_feed_multiple_chunks(self):
        buf = StreamingBuffer(n_channels=4, max_length=100)
        buf.feed(np.zeros((30, 4)))
        buf.feed(np.zeros((20, 4)))
        assert buf.length == 50
        assert buf.total_samples_received == 50

    def test_max_length_trim(self):
        buf = StreamingBuffer(n_channels=4, max_length=50)
        buf.feed(np.zeros((60, 4)))
        assert buf.length == 50
        assert buf.total_samples_received == 60

    def test_rolling_trim(self):
        """Buffer should keep most recent samples when trimmed."""
        buf = StreamingBuffer(n_channels=1, max_length=5)
        data = np.arange(10).reshape(-1, 1).astype(np.float32)
        buf.feed(data)
        raw = buf.get_raw()
        np.testing.assert_array_equal(raw.flatten(), [5, 6, 7, 8, 9])

    def test_is_full(self):
        buf = StreamingBuffer(n_channels=4, max_length=20)
        buf.feed(np.zeros((20, 4)))
        assert buf.is_full

    def test_get_window_padded(self):
        """When buffer < max_length, get_window should zero-pad on left."""
        buf = StreamingBuffer(n_channels=2, max_length=10)
        buf.feed(np.ones((3, 2)))
        window = buf.get_window()
        assert window.shape == (10, 2)
        # First 7 rows should be zero (padding)
        np.testing.assert_array_equal(window[:7], 0.0)
        # Last 3 rows should be ones
        np.testing.assert_array_equal(window[7:], 1.0)

    def test_get_window_full(self):
        buf = StreamingBuffer(n_channels=2, max_length=5)
        data = np.arange(10).reshape(5, 2).astype(np.float32)
        buf.feed(data)
        window = buf.get_window()
        assert window.shape == (5, 2)
        np.testing.assert_array_equal(window, data)

    def test_wrong_channels_raises(self):
        buf = StreamingBuffer(n_channels=4, max_length=100)
        with pytest.raises(ValueError, match="Expected 4 channels"):
            buf.feed(np.zeros((10, 3)))

    def test_1d_input_reshaped(self):
        """1D input should be treated as single timestep."""
        buf = StreamingBuffer(n_channels=4, max_length=100)
        buf.feed(np.zeros(4))
        assert buf.length == 1

    def test_reset(self):
        buf = StreamingBuffer(n_channels=4, max_length=100)
        buf.feed(np.zeros((50, 4)))
        buf.reset()
        assert buf.length == 0
        assert buf.total_samples_received == 0


# ---------------------------------------------------------------------------
# LatencyStats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_empty(self):
        stats = LatencyStats()
        assert stats.count == 0
        assert stats.mean_ms == 0.0
        assert stats.max_ms == 0.0

    def test_record_and_summary(self):
        stats = LatencyStats()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            stats.record(v)
        assert stats.count == 5
        assert stats.mean_ms == 30.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 50.0

    def test_p95(self):
        stats = LatencyStats()
        for v in range(100):
            stats.record(float(v))
        assert stats.p95_ms >= 94.0

    def test_summary_dict(self):
        stats = LatencyStats()
        stats.record(5.0)
        s = stats.summary()
        assert "count" in s
        assert "mean_ms" in s
        assert "min_ms" in s
        assert "max_ms" in s
        assert "p95_ms" in s


# ---------------------------------------------------------------------------
# StreamingDecoder
# ---------------------------------------------------------------------------

class TestStreamingDecoder:
    def test_initial_state(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=10,
        )
        assert decoder.current_text == ""

    def test_feed_below_chunk_size_returns_none(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=20,
        )
        result = decoder.feed(np.zeros((5, 4), dtype=np.float32))
        assert result is None

    def test_feed_triggers_inference(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=10,
        )
        result = decoder.feed(np.zeros((10, 4), dtype=np.float32))
        assert result is not None
        assert isinstance(result, DecoderUpdate)

    def test_update_has_latency(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=10,
        )
        result = decoder.feed(np.zeros((10, 4), dtype=np.float32))
        assert result.latency_ms > 0

    def test_update_has_buffer_length(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=10,
        )
        result = decoder.feed(np.zeros((10, 4), dtype=np.float32))
        assert result.buffer_length == 10

    def test_text_is_string(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=10,
        )
        result = decoder.feed(np.zeros((10, 4), dtype=np.float32))
        assert isinstance(result.text, str)

    def test_multiple_feeds(self, tiny_model, trial_data):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=20,
        )
        updates = []
        for start in range(0, 200, 20):
            result = decoder.feed(trial_data[start : start + 20])
            if result is not None:
                updates.append(result)
        assert len(updates) >= 1

    def test_force_decode(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=50,
        )
        decoder.feed(np.zeros((10, 4), dtype=np.float32))  # below chunk_size
        result = decoder.force_decode()
        assert isinstance(result, DecoderUpdate)

    def test_reset_clears_state(self, tiny_model):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=10,
        )
        decoder.feed(np.zeros((10, 4), dtype=np.float32))
        decoder.reset()
        assert decoder.current_text == ""
        assert decoder.buffer.length == 0
        assert decoder.latency.count == 0

    def test_latency_tracked(self, tiny_model, trial_data):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=20,
        )
        for start in range(0, 200, 20):
            decoder.feed(trial_data[start : start + 20])
        assert decoder.latency.count > 0

    def test_latency_under_300ms(self, tiny_model):
        """Inference on a small model should be well under 300ms target."""
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=50,
        )
        result = decoder.feed(np.zeros((50, 4), dtype=np.float32))
        assert result.latency_ms < 300.0

    def test_stable_mode(self, tiny_model, trial_data):
        decoder = StreamingDecoder(
            model=tiny_model, n_channels=4, t_max=100, chunk_size=20,
            stable_mode=True, stability_window=3,
        )
        updates = []
        for start in range(0, 200, 20):
            result = decoder.feed(trial_data[start : start + 20])
            if result is not None:
                updates.append(result)
        # Stable mode should produce DecoderUpdate with is_stable field
        assert all(isinstance(u.is_stable, bool) for u in updates)


# ---------------------------------------------------------------------------
# simulate_streaming
# ---------------------------------------------------------------------------

class TestSimulateStreaming:
    def test_returns_result_dict(self, tiny_model, trial_data):
        result = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=50,
            n_channels=4,
            t_max=100,
        )
        assert "final_text" in result
        assert "updates" in result
        assert "latency" in result
        assert "n_updates" in result

    def test_updates_list_non_empty(self, tiny_model, trial_data):
        result = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=50,
            n_channels=4,
            t_max=100,
        )
        assert result["n_updates"] > 0

    def test_callback_invoked(self, tiny_model, trial_data):
        calls = []
        result = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=50,
            n_channels=4,
            t_max=100,
            callback=lambda u: calls.append(u),
        )
        assert len(calls) > 0
        assert all(isinstance(c, DecoderUpdate) for c in calls)

    def test_final_text_is_string(self, tiny_model, trial_data):
        result = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=50,
            n_channels=4,
            t_max=100,
        )
        assert isinstance(result["final_text"], str)

    def test_latency_stats_present(self, tiny_model, trial_data):
        result = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=50,
            n_channels=4,
            t_max=100,
        )
        latency = result["latency"]
        assert "mean_ms" in latency
        assert "p95_ms" in latency
        assert latency["count"] > 0

    def test_stable_mode_simulation(self, tiny_model, trial_data):
        result = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=50,
            n_channels=4,
            t_max=100,
            stable_mode=True,
        )
        assert isinstance(result["final_text"], str)

    def test_small_chunks(self, tiny_model, trial_data):
        """Small chunk_size should produce more updates."""
        result_small = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=10,
            n_channels=4,
            t_max=100,
        )
        result_large = simulate_streaming(
            model=tiny_model,
            trial_data=trial_data,
            chunk_size=100,
            n_channels=4,
            t_max=100,
        )
        assert result_small["n_updates"] >= result_large["n_updates"]
