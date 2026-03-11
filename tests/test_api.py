"""Tests for the FastAPI backend (app/api.py).

Tests each endpoint with valid and invalid inputs using TestClient.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import (
    app,
    _load_model,
    _run_inference,
    _generate_demo_sample,
    MODEL_CLASSES,
    DecodeResponse,
    HealthResponse,
    ModelInfoResponse,
)


@pytest.fixture(scope="module")
def client():
    """Create a test client — startup event runs automatically."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert isinstance(data["models_loaded"], list)

    def test_health_schema(self, client):
        r = client.get("/health")
        data = r.json()
        # Validate it matches our schema
        resp = HealthResponse(**data)
        assert resp.status == "ok"


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

class TestModelInfoEndpoint:
    def test_model_info_default(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.json()
        assert "parameter_count" in data
        assert data["parameter_count"] > 0
        assert data["model_name"] == "gru_decoder"
        assert data["n_classes"] == 28

    @pytest.mark.parametrize("model_type", list(MODEL_CLASSES.keys()))
    def test_model_info_all_models(self, client, model_type):
        r = client.get("/model/info", params={"model": model_type})
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == model_type
        assert data["parameter_count"] > 0
        assert model_type in data["available_models"]

    def test_model_info_invalid(self, client):
        r = client.get("/model/info", params={"model": "nonexistent"})
        assert r.status_code == 400

    def test_model_info_schema(self, client):
        r = client.get("/model/info")
        data = r.json()
        resp = ModelInfoResponse(**data)
        assert resp.n_channels == 192


# ---------------------------------------------------------------------------
# /decode
# ---------------------------------------------------------------------------

class TestDecodeEndpoint:
    def _make_npy_upload(self, arr: np.ndarray) -> tuple:
        """Create a file-like .npy upload from a numpy array."""
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        return ("file", ("test.npy", buf, "application/octet-stream"))

    def test_decode_valid_2d(self, client):
        features = np.random.randn(200, 192).astype(np.float32)
        r = client.post(
            "/decode",
            files=[self._make_npy_upload(features)],
            params={"model": "gru_decoder", "beam_width": 5},
        )
        assert r.status_code == 200
        data = r.json()
        assert "predicted_text" in data
        assert "raw_ctc_output" in data
        assert "beam_hypotheses" in data
        assert "char_probabilities" in data
        assert "inference_time_ms" in data
        assert isinstance(data["inference_time_ms"], float)
        assert data["inference_time_ms"] >= 0

    def test_decode_valid_3d(self, client):
        features = np.random.randn(1, 200, 192).astype(np.float32)
        r = client.post(
            "/decode",
            files=[self._make_npy_upload(features)],
            params={"model": "gru_decoder"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "predicted_text" in data

    @pytest.mark.parametrize("model_type", list(MODEL_CLASSES.keys()))
    def test_decode_all_models(self, client, model_type):
        features = np.random.randn(100, 192).astype(np.float32)
        r = client.post(
            "/decode",
            files=[self._make_npy_upload(features)],
            params={"model": model_type, "beam_width": 3},
        )
        assert r.status_code == 200
        data = r.json()
        assert "predicted_text" in data
        assert len(data["beam_hypotheses"]) > 0

    def test_decode_invalid_model(self, client):
        features = np.random.randn(100, 192).astype(np.float32)
        r = client.post(
            "/decode",
            files=[self._make_npy_upload(features)],
            params={"model": "nonexistent"},
        )
        assert r.status_code == 400

    def test_decode_wrong_file_extension(self, client):
        buf = io.BytesIO(b"not a numpy file")
        r = client.post(
            "/decode",
            files=[("file", ("test.txt", buf, "text/plain"))],
        )
        assert r.status_code == 400

    def test_decode_invalid_shape_1d(self, client):
        features = np.random.randn(100).astype(np.float32)
        r = client.post(
            "/decode",
            files=[self._make_npy_upload(features)],
        )
        assert r.status_code == 400

    def test_decode_response_schema(self, client):
        features = np.random.randn(100, 192).astype(np.float32)
        r = client.post(
            "/decode",
            files=[self._make_npy_upload(features)],
            params={"model": "gru_decoder", "beam_width": 3},
        )
        data = r.json()
        resp = DecodeResponse(**data)
        assert resp.inference_time_ms >= 0
        assert isinstance(resp.beam_hypotheses, list)
        assert isinstance(resp.char_probabilities, list)

    def test_decode_with_lm(self, client):
        features = np.random.randn(100, 192).astype(np.float32)
        r = client.post(
            "/decode",
            files=[self._make_npy_upload(features)],
            params={"model": "gru_decoder", "beam_width": 5, "use_lm": True},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /decode/demo
# ---------------------------------------------------------------------------

class TestDecodeDemoEndpoint:
    def test_decode_demo_default(self, client):
        r = client.get("/decode/demo")
        assert r.status_code == 200
        data = r.json()
        assert "predicted_text" in data
        assert "inference_time_ms" in data

    @pytest.mark.parametrize("model_type", list(MODEL_CLASSES.keys()))
    def test_decode_demo_all_models(self, client, model_type):
        r = client.get("/decode/demo", params={"model": model_type, "beam_width": 3})
        assert r.status_code == 200
        data = r.json()
        assert "predicted_text" in data

    def test_decode_demo_invalid_model(self, client):
        r = client.get("/decode/demo", params={"model": "nonexistent"})
        assert r.status_code == 400

    def test_decode_demo_with_lm(self, client):
        r = client.get("/decode/demo", params={"use_lm": True, "beam_width": 5})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_load_model_all_types(self):
        for model_type in MODEL_CLASSES:
            model = _load_model(model_type)
            assert model is not None
            assert model.n_classes == 28
            assert model.n_channels == 192

    def test_generate_demo_sample(self):
        sample = _generate_demo_sample()
        assert isinstance(sample, np.ndarray)
        assert sample.ndim == 2
        assert sample.shape[0] > 0  # has timesteps
        assert sample.shape[1] > 0  # has channels

    def test_generate_demo_sample_custom_shape(self):
        sample = _generate_demo_sample(n_channels=64, t_max=500)
        assert sample.shape == (500, 64)

    def test_run_inference(self):
        features = np.random.randn(200, 192).astype(np.float32)
        result = _run_inference(features, model_type="gru_decoder", beam_width=3)
        assert isinstance(result, DecodeResponse)
        assert result.inference_time_ms >= 0
        assert isinstance(result.predicted_text, str)
        assert isinstance(result.beam_hypotheses, list)
        assert len(result.char_probabilities) > 0

    def test_run_inference_beam_hypotheses_sorted(self):
        features = np.random.randn(200, 192).astype(np.float32)
        result = _run_inference(features, model_type="gru_decoder", beam_width=10)
        scores = [h["score"] for h in result.beam_hypotheses]
        assert scores == sorted(scores, reverse=True)
