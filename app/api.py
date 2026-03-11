"""FastAPI backend for the Brain-Text Decoder.

Endpoints:
    POST /decode       — Upload .npy file, preprocess, infer, return JSON
    GET  /decode/demo  — Decode a pre-loaded sample trial
    GET  /health       — Service status
    GET  /model/info   — Model architecture name, param count, test metrics
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from src.config import Config, load_config
from src.models.gru_decoder import GRUDecoder
from src.models.cnn_lstm import CNNLSTM
from src.models.transformer import TransformerDecoder
from src.models.cnn_transformer import CNNTransformer
from src.decoding.greedy import greedy_decode
from src.decoding.beam_search import beam_search_decode, Hypothesis
from src.decoding.lm_correction import load_lm_scorer, rescore_hypotheses

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class DecodeResponse(BaseModel):
    predicted_text: str
    raw_ctc_output: str
    beam_hypotheses: list[dict]
    char_probabilities: list[list[float]]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    architecture: str
    parameter_count: int
    n_channels: int
    n_classes: int
    available_models: list[str]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CLASSES = {
    "gru_decoder": GRUDecoder,
    "cnn_lstm": CNNLSTM,
    "transformer": TransformerDecoder,
    "cnn_transformer": CNNTransformer,
}

# Global state
_models: dict[str, torch.nn.Module] = {}
_config: Optional[Config] = None
_lm_scorer = None
_demo_sample: Optional[np.ndarray] = None


def _load_model(model_type: str, checkpoint_path: Optional[Path] = None) -> torch.nn.Module:
    """Instantiate a model and optionally load weights from a checkpoint."""
    cfg = _config or load_config(preset="willett_handwriting")
    n_channels = 192
    n_classes = cfg.n_classes

    if model_type == "gru_decoder":
        model = GRUDecoder(n_channels=n_channels, n_classes=n_classes)
    elif model_type == "cnn_lstm":
        model = CNNLSTM(
            n_channels=n_channels, n_classes=n_classes,
            conv_channels=cfg.conv_channels,
            conv_kernel_size=cfg.conv_kernel_size,
            conv_layers=cfg.conv_layers,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
        )
    elif model_type == "transformer":
        model = TransformerDecoder(
            n_channels=n_channels, n_classes=n_classes,
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.transformer_layers,
            ffn_dim=cfg.ffn_dim, dropout=cfg.transformer_dropout,
            max_seq_len=cfg.max_seq_len,
        )
    elif model_type == "cnn_transformer":
        model = CNNTransformer(
            n_channels=n_channels, n_classes=n_classes,
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_transformer_layers=cfg.hybrid_transformer_layers,
            ffn_dim=cfg.ffn_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint_path and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        logger.info("Loaded checkpoint for %s from %s", model_type, checkpoint_path)

    model.eval()
    return model


def _get_model(model_type: str) -> torch.nn.Module:
    """Get a loaded model by type, loading on first access."""
    if model_type not in _models:
        checkpoint_dir = Path(_config.checkpoint_dir if _config else "./outputs/checkpoints")
        checkpoint_path = checkpoint_dir / f"{model_type}_best.pt"
        _models[model_type] = _load_model(model_type, checkpoint_path)
    return _models[model_type]


def _run_inference(
    features: np.ndarray,
    model_type: str = "gru_decoder",
    beam_width: int = 10,
    use_lm: bool = False,
) -> DecodeResponse:
    """Run full decode pipeline on a numpy feature array."""
    model = _get_model(model_type)

    # Prepare input
    if features.ndim == 2:
        features = features[np.newaxis, ...]  # [1, T, C]

    tensor = torch.from_numpy(features.astype(np.float32))

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)  # [1, T', n_classes]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    logits_np = logits.detach().cpu().numpy()

    # Greedy decode (raw CTC output)
    raw_ctc = greedy_decode(logits_np[0])

    # Beam search
    hypotheses = beam_search_decode(
        logits_np[0], beam_width=beam_width, top_k=5,
    )

    # LM rescoring
    if use_lm and _lm_scorer is not None:
        hypotheses = rescore_hypotheses(hypotheses, _lm_scorer, alpha=0.3, beta=0.0)

    predicted_text = hypotheses[0].text if hypotheses else raw_ctc

    # Character probabilities (softmax of logits at each timestep)
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    # Subsample to keep response size manageable
    max_steps = 200
    step = max(1, probs.shape[0] // max_steps)
    char_probs = probs[::step].tolist()

    beam_hyp_dicts = [
        {"text": h.text, "score": round(h.score, 4)}
        for h in hypotheses
    ]

    return DecodeResponse(
        predicted_text=predicted_text,
        raw_ctc_output=raw_ctc,
        beam_hypotheses=beam_hyp_dicts,
        char_probabilities=char_probs,
        inference_time_ms=round(elapsed_ms, 2),
    )


def _generate_demo_sample(n_channels: int = 192, t_max: int = 2000) -> np.ndarray:
    """Generate or load a demo sample for the /decode/demo endpoint."""
    # Check for saved demo sample
    demo_path = Path("data/demo_sample.npy")
    if demo_path.exists():
        return np.load(demo_path)

    # Check for any trial data in the data directory
    data_dir = Path("data")
    npy_files = list(data_dir.rglob("trial_*_signals.npy"))
    if npy_files:
        return np.load(npy_files[0])

    # Generate synthetic demo signal
    rng = np.random.RandomState(42)
    t = np.linspace(0, 2.0, t_max)
    sample = np.zeros((t_max, n_channels), dtype=np.float32)
    for ch in range(n_channels):
        freq = 5 + ch * 0.5
        sample[:, ch] = (
            np.sin(2 * np.pi * freq * t)
            + 0.3 * rng.randn(t_max)
        ).astype(np.float32)
    return sample


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load config, default model, and LM scorer on startup."""
    global _config, _lm_scorer, _demo_sample

    yaml_path = Path("config.yaml")
    _config = load_config(
        yaml_path=yaml_path if yaml_path.exists() else None,
        preset="willett_handwriting",
    )

    # Pre-load the default model
    try:
        _get_model("gru_decoder")
        logger.info("Pre-loaded gru_decoder model")
    except Exception as e:
        logger.warning("Could not pre-load gru_decoder: %s", e)

    # Load LM scorer
    lm_path = Path("outputs/lm/char_5gram.binary")
    _lm_scorer = load_lm_scorer(lm_path if lm_path.exists() else None)

    # Prepare demo sample
    _demo_sample = _generate_demo_sample()
    logger.info("API startup complete")

    yield


app = FastAPI(
    title="Brain-Text Decoder API",
    description="Neural signal to text decoding with CTC-trained models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service status and loaded models."""
    return HealthResponse(
        status="ok",
        models_loaded=list(_models.keys()),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    model: str = Query("gru_decoder", description="Model type to query"),
):
    """Return model architecture info and parameter count."""
    if model not in MODEL_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(MODEL_CLASSES.keys())}",
        )

    m = _get_model(model)
    return ModelInfoResponse(
        model_name=model,
        architecture=m.__class__.__name__,
        parameter_count=m.count_parameters(),
        n_channels=m.n_channels,
        n_classes=m.n_classes,
        available_models=list(MODEL_CLASSES.keys()),
    )


@app.post("/decode", response_model=DecodeResponse)
async def decode(
    file: UploadFile = File(..., description="NumPy .npy file with neural features"),
    model: str = Query("gru_decoder", description="Model type"),
    beam_width: int = Query(10, ge=1, le=200, description="Beam search width"),
    use_lm: bool = Query(False, description="Apply LM rescoring"),
):
    """Decode uploaded neural recording.

    Accepts a .npy file containing features of shape [T, C] or [B, T, C].
    Returns predicted text, beam hypotheses, and character probabilities.
    """
    # Validate file type
    if file.filename and not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are accepted")

    if model not in MODEL_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(MODEL_CLASSES.keys())}",
        )

    # Read and parse the file
    try:
        contents = await file.read()
        features = np.load(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load .npy file: {e}")

    # Validate shape
    if features.ndim not in (2, 3):
        raise HTTPException(
            status_code=400,
            detail=f"Expected 2D [T, C] or 3D [B, T, C] array, got shape {features.shape}",
        )

    if features.ndim == 3:
        features = features[0]  # take first sample

    return _run_inference(features, model_type=model, beam_width=beam_width, use_lm=use_lm)


@app.get("/decode/demo", response_model=DecodeResponse)
async def decode_demo(
    model: str = Query("gru_decoder", description="Model type"),
    beam_width: int = Query(10, ge=1, le=200, description="Beam search width"),
    use_lm: bool = Query(False, description="Apply LM rescoring"),
):
    """Decode a pre-loaded demo sample (no upload needed)."""
    if model not in MODEL_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(MODEL_CLASSES.keys())}",
        )

    if _demo_sample is None:
        raise HTTPException(status_code=503, detail="Demo sample not loaded")

    return _run_inference(_demo_sample, model_type=model, beam_width=beam_width, use_lm=use_lm)
