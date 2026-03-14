"""Tests for the Streamlit dashboard helper functions (app/dashboard.py).

Tests the local decode fallback, demo signal generation, and utility functions.
Does not test Streamlit UI rendering (requires streamlit testing framework).
"""

from __future__ import annotations

import numpy as np
import pytest

# Import helpers — avoid importing streamlit UI code at module level
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestLocalDecode:
    """Test the _local_decode fallback function."""

    def _get_local_decode(self):
        from app.dashboard import _local_decode
        return _local_decode

    def test_local_decode_gru(self):
        local_decode = self._get_local_decode()
        features = np.random.randn(200, 192).astype(np.float32)
        result = local_decode(features, "gru_decoder", beam_width=3, use_lm=False)

        assert "predicted_text" in result
        assert "raw_ctc_output" in result
        assert "beam_hypotheses" in result
        assert "char_probabilities" in result
        assert "inference_time_ms" in result
        assert result["inference_time_ms"] >= 0

    def test_local_decode_cnn_lstm(self):
        local_decode = self._get_local_decode()
        features = np.random.randn(200, 192).astype(np.float32)
        result = local_decode(features, "cnn_lstm", beam_width=3, use_lm=False)
        assert "predicted_text" in result

    def test_local_decode_transformer(self):
        local_decode = self._get_local_decode()
        features = np.random.randn(200, 192).astype(np.float32)
        result = local_decode(features, "transformer", beam_width=3, use_lm=False)
        assert "predicted_text" in result

    def test_local_decode_cnn_transformer(self):
        local_decode = self._get_local_decode()
        features = np.random.randn(200, 192).astype(np.float32)
        result = local_decode(features, "cnn_transformer", beam_width=3, use_lm=False)
        assert "predicted_text" in result

    def test_local_decode_beam_hypotheses_present(self):
        local_decode = self._get_local_decode()
        features = np.random.randn(200, 192).astype(np.float32)
        result = local_decode(features, "gru_decoder", beam_width=5, use_lm=False)
        assert len(result["beam_hypotheses"]) > 0
        for h in result["beam_hypotheses"]:
            assert "text" in h
            assert "score" in h

    def test_local_decode_char_probs_shape(self):
        local_decode = self._get_local_decode()
        features = np.random.randn(200, 192).astype(np.float32)
        result = local_decode(features, "gru_decoder", beam_width=3, use_lm=False)
        probs = result["char_probabilities"]
        assert len(probs) > 0
        # Each row should have n_classes probabilities
        assert len(probs[0]) == 28

    def test_local_decode_with_lm(self):
        local_decode = self._get_local_decode()
        features = np.random.randn(200, 192).astype(np.float32)
        # Should not crash even without a real LM model
        result = local_decode(features, "gru_decoder", beam_width=3, use_lm=True)
        assert "predicted_text" in result


class TestDemoSignal:
    """Test the demo signal generation function."""

    def test_generate_demo_signal(self):
        from app.dashboard import _generate_demo_signal
        signal = _generate_demo_signal()
        assert isinstance(signal, np.ndarray)
        assert signal.ndim == 2
        assert signal.shape[1] == 192

    def test_generate_demo_signal_custom(self):
        from app.dashboard import _generate_demo_signal
        signal = _generate_demo_signal(n_channels=64, t_max=500)
        assert signal.ndim == 2
        # With real data on disk, the real file is returned; synthetic uses custom shape
        assert signal.shape[1] in (64, 192)

    def test_generate_demo_signal_deterministic(self):
        from app.dashboard import _generate_demo_signal
        s1 = _generate_demo_signal()
        s2 = _generate_demo_signal()
        np.testing.assert_array_equal(s1, s2)

    def test_generate_demo_signal_not_constant(self):
        from app.dashboard import _generate_demo_signal
        signal = _generate_demo_signal()
        # Should not be all zeros or constant
        assert signal.std() > 0


class TestAPIAvailability:
    """Test the API availability check."""

    def test_api_available_returns_bool(self):
        from app.dashboard import api_available
        result = api_available()
        assert isinstance(result, bool)
