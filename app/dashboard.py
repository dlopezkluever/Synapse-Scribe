"""Streamlit frontend for the Brain-Text Decoder.

Multi-page app with:
    1. Upload & Decode — upload .npy, run inference, view results
    2. Signal Viewer — interactive neural signal exploration & diagnostics
    3. Decoding Visualization — CTC heatmaps, beam hypotheses, comparison
    4. Benchmarks — model comparison table, CER/WER charts
    5. Neural Representations — embedding explorer, trajectories, saliency
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"
MODEL_OPTIONS = ["cnn_lstm", "gru_decoder", "transformer", "cnn_transformer"]
MODEL_DISPLAY = {
    "cnn_lstm": "CNN + LSTM — Best (0.43% CER)",
    "cnn_transformer": "Hybrid CNN-Transformer (35.2% CER)",
    "gru_decoder": "GRU Decoder — Failed (98.2% CER)",
    "transformer": "Transformer — Failed (100% CER)",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

_icon_path = Path(__file__).resolve().parent.parent / "icon.ico"
_page_icon = str(_icon_path) if _icon_path.exists() else "🧠"

st.set_page_config(
    page_title="Synapse Scribe",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — industrial instrument aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-root: #0a0a0e;
    --bg-surface: #111116;
    --bg-elevated: #19191f;
    --bg-input: #13131a;
    --border-subtle: #1f1f28;
    --border-default: #2a2a35;
    --border-accent: #c9a84c;
    --text-primary: #e8e6e3;
    --text-secondary: #807e88;
    --text-muted: #55535e;
    --accent: #c9a84c;
    --accent-hover: #d4b35c;
    --accent-dim: #8a7434;
    --accent-glow: rgba(201, 168, 76, 0.08);
    --green-ok: #5b8c5a;
    --red-err: #b54a4a;
    --radius: 2px;
    --transition-fast: 0.15s ease-out;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
    }
}

/* ── Global ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-root) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── Typography ─────────────────────────────────── */
h1 {
    font-weight: 700 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--border-accent) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1.5rem !important;
}
h2, [data-testid="stHeadingWithActionElements"] h2 {
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    margin-top: 2rem !important;
    margin-bottom: 0.75rem !important;
}
h3 {
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: var(--text-primary) !important;
}
p, li, label, .stMarkdown {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
}
/* Apply font to spans but not Streamlit's internal icon spans */
span:not([data-testid]):not(.material-symbols-rounded):not(.material-icons) {
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
/* Hide raw icon text if Material Icons font fails to load */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="collapsedControl"] span {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    overflow: hidden !important;
}
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-elevated) !important;
    color: var(--accent) !important;
    padding: 0.15em 0.4em !important;
    border-radius: var(--radius) !important;
    font-size: 0.88em !important;
}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.25rem !important;
    border-bottom: 1px solid var(--border-accent) !important;
    padding-bottom: 0.4rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 0.75rem 0 !important;
}
/* Sidebar radio buttons — navigation */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 2px !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius) !important;
    padding: 0.4rem 0.6rem !important;
    transition: background var(--transition-fast), border-color var(--transition-fast), color var(--transition-fast) !important;
    font-size: 0.9rem !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--accent-glow) !important;
    border-color: var(--border-subtle) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--accent-glow) !important;
    border-left: 2px solid var(--border-accent) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}

/* ── Inputs & Select ────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
.stSelectbox > div > div {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] ~ div div {
    background: var(--accent-dim) !important;
}

/* ── Buttons ────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.45rem 1.2rem !important;
    transition: background var(--transition-fast), border-color var(--transition-fast), color var(--transition-fast) !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-glow) !important;
}
.stButton > button:active {
    background: rgba(201, 168, 76, 0.15) !important;
    border-color: var(--accent) !important;
}
.stButton > button:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: var(--bg-root) !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
    color: var(--bg-root) !important;
}
.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="stBaseButton-primary"]:active {
    background: var(--accent-dim) !important;
    border-color: var(--accent-dim) !important;
}

/* ── Metrics ────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: 2px solid var(--accent-dim) !important;
    padding: 0.8rem 1rem !important;
    border-radius: var(--radius) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid var(--border-default) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.2rem !important;
    text-transform: uppercase !important;
    transition: color var(--transition-fast), border-color var(--transition-fast) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    border-bottom-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}

/* ── Expanders ──────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    transition: color var(--transition-fast) !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}

/* ── File uploader ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 1px dashed var(--border-default) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color var(--transition-fast) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-dim) !important;
}

/* ── Alerts / Info / Success / Warning ──────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    font-size: 0.88rem !important;
}
.stAlert [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
}

/* ── Captions ───────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

/* ── Dataframe ──────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

/* ── Checkbox ───────────────────────────────────── */
[data-testid="stCheckbox"] span[data-testid="stCheckbox-label"] {
    color: var(--text-primary) !important;
}

/* ── Multiselect ────────────────────────────────── */
.stMultiSelect [data-baseweb="tag"] {
    background: var(--accent-glow) !important;
    border: 1px solid var(--accent-dim) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

/* ── Divider ────────────────────────────────────── */
hr {
    border-color: var(--border-subtle) !important;
}

/* ── Plotly chart container — DO NOT touch internals */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 0.25rem !important;
    background: var(--bg-surface) !important;
}

/* ── Progress bar ───────────────────────────────── */
.stProgress > div > div {
    background-color: var(--accent) !important;
}

/* ── Spinner ────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ── Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-root); }
::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: var(--radius); }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Custom component classes ───────────────────── */
.ss-result-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    padding: 1rem 1.25rem;
    background: var(--bg-elevated);
    border-left: 3px solid var(--accent);
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    margin: 0.5rem 0 1rem 0;
    color: var(--text-primary);
    letter-spacing: 0.02em;
    line-height: 1.6;
}
.ss-status-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: var(--radius);
}
.ss-status-online {
    background: rgba(91, 140, 90, 0.15);
    color: #7db87c;
    border: 1px solid rgba(91, 140, 90, 0.3);
}
.ss-status-offline {
    background: rgba(181, 74, 74, 0.12);
    color: #c47a7a;
    border: 1px solid rgba(181, 74, 74, 0.25);
}
.ss-comparison-row {
    padding: 0.75rem 1rem;
    margin: 0.35rem 0;
    background: var(--bg-elevated);
    border-left: 2px solid var(--border-default);
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.92rem;
}
.ss-comparison-row.ss-best {
    border-left-color: var(--accent);
    background: var(--accent-glow);
}
.ss-comparison-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}
.ss-comparison-text {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    font-size: 0.92rem;
}
.ss-empty-state {
    text-align: left;
    padding: 2.5rem 1.5rem;
    border: 1px dashed var(--border-default);
    margin: 1rem 0;
    background: var(--bg-surface);
}
.ss-empty-state .ss-empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.ss-empty-state .ss-empty-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.5;
}
.ss-settings-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.35rem 0;
    font-size: 0.85rem;
    border-bottom: 1px solid var(--border-subtle);
}
.ss-settings-key {
    color: var(--text-muted);
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    padding-top: 0.1rem;
}
.ss-settings-val {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    font-size: 0.82rem;
}
.ss-hypothesis {
    padding: 0.6rem 1rem;
    margin: 0.25rem 0;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ss-hypothesis:first-child {
    border-left: 2px solid var(--accent);
}
.ss-hyp-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    min-width: 2rem;
}
.ss-hyp-text {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    flex: 1;
    margin: 0 1rem;
}
.ss-hyp-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--accent-dim);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_available() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_decode(features: np.ndarray, model: str, beam_width: int, use_lm: bool) -> dict | None:
    """Call the /decode endpoint with a numpy array."""
    buf = io.BytesIO()
    np.save(buf, features)
    buf.seek(0)

    try:
        r = requests.post(
            f"{API_BASE}/decode",
            files={"file": ("signal.npy", buf, "application/octet-stream")},
            params={"model": model, "beam_width": beam_width, "use_lm": use_lm},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
    except requests.ConnectionError:
        st.error("Cannot connect to API. Start the backend with: `uvicorn app.api:app`")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None


def call_decode_demo(model: str, beam_width: int, use_lm: bool) -> dict | None:
    """Call the /decode/demo endpoint."""
    try:
        r = requests.get(
            f"{API_BASE}/decode/demo",
            params={"model": model, "beam_width": beam_width, "use_lm": use_lm},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
    except requests.ConnectionError:
        st.error("Cannot connect to API. Start the backend with: `uvicorn app.api:app`")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None


def call_model_info(model: str) -> dict | None:
    """Call the /model/info endpoint."""
    try:
        r = requests.get(f"{API_BASE}/model/info", params={"model": model}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _local_decode(features: np.ndarray, model_type: str, beam_width: int, use_lm: bool) -> dict:
    """Run inference locally without the API server."""
    import torch
    from src.config import load_config
    from src.models.gru_decoder import GRUDecoder
    from src.models.cnn_lstm import CNNLSTM
    from src.models.transformer import TransformerDecoder
    from src.models.cnn_transformer import CNNTransformer
    from src.decoding.greedy import greedy_decode
    from src.decoding.beam_search import beam_search_decode
    from src.decoding.lm_correction import load_lm_scorer, rescore_hypotheses

    cfg = load_config(preset="willett_handwriting")
    n_ch = features.shape[-1]

    # Build model
    model_classes = {
        "gru_decoder": lambda: GRUDecoder(n_channels=n_ch, n_classes=cfg.n_classes),
        "cnn_lstm": lambda: CNNLSTM(n_channels=n_ch, n_classes=cfg.n_classes),
        "transformer": lambda: TransformerDecoder(n_channels=n_ch, n_classes=cfg.n_classes),
        "cnn_transformer": lambda: CNNTransformer(n_channels=n_ch, n_classes=cfg.n_classes),
    }

    model = model_classes[model_type]()

    # Map model type keys to actual checkpoint filenames
    _ckpt_names = {
        "gru_decoder": "GRUDecoder_best.pt",
        "cnn_lstm": "CNNLSTM_best.pt",
        "transformer": "TransformerDecoder_best.pt",
        "cnn_transformer": "CNNTransformer_best.pt",
    }
    ckpt_name = _ckpt_names.get(model_type, f"{model_type}_best.pt")
    # Prefer GPU-trained checkpoints
    gpu_ckpt = Path(cfg.checkpoint_dir) / "GPU-3-13" / ckpt_name
    base_ckpt = Path(cfg.checkpoint_dir) / ckpt_name
    ckpt = gpu_ckpt if gpu_ckpt.exists() else base_ckpt
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    model.eval()

    # Z-score normalize (must match training preprocessing)
    norm_path = Path("outputs/normalization_stats.npz")
    if norm_path.exists():
        stats = np.load(norm_path)
        features = (features.astype(np.float32) - stats["mean"]) / stats["std"]

    if features.ndim == 2:
        features = features[np.newaxis, ...]

    tensor = torch.from_numpy(features.astype(np.float32))

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    logits_np = logits.detach().cpu().numpy()
    raw_ctc = greedy_decode(logits_np[0])
    hypotheses = beam_search_decode(logits_np[0], beam_width=beam_width, top_k=5)

    if use_lm:
        lm_path = Path("outputs/lm/char_5gram.binary")
        lm = load_lm_scorer(lm_path if lm_path.exists() else None)
        hypotheses = rescore_hypotheses(hypotheses, lm, alpha=0.3)

    predicted = hypotheses[0].text if hypotheses else raw_ctc

    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    max_steps = 200
    step = max(1, probs.shape[0] // max_steps)
    char_probs = probs[::step].tolist()

    return {
        "predicted_text": predicted,
        "raw_ctc_output": raw_ctc,
        "beam_hypotheses": [{"text": h.text, "score": round(h.score, 4)} for h in hypotheses],
        "char_probabilities": char_probs,
        "inference_time_ms": round(elapsed_ms, 2),
        "_logits": logits_np,
    }


def smart_decode(features, model_type, beam_width, use_lm):
    """Try API first, fall back to local inference."""
    if api_available():
        return call_decode(features, model_type, beam_width, use_lm)
    return _local_decode(features, model_type, beam_width, use_lm)


def smart_decode_demo(model_type, beam_width, use_lm):
    """Try API demo endpoint first, fall back to local inference."""
    if api_available():
        return call_decode_demo(model_type, beam_width, use_lm)
    demo = _generate_demo_signal()
    return _local_decode(demo, model_type, beam_width, use_lm)


def _generate_demo_signal(n_channels: int = 192, t_max: int = 5000) -> np.ndarray:
    """Load a real demo signal, falling back to synthetic if unavailable."""
    # Try loading real demo sample
    demo_path = Path("data/demo_sample.npy")
    if demo_path.exists():
        sample = np.load(demo_path).astype(np.float32)
        if sample.shape[0] > t_max:
            sample = sample[:t_max]
        return sample

    # Try any real trial data
    data_dir = Path("data")
    npy_files = list(data_dir.rglob("trial_*_signals.npy"))
    if npy_files:
        sample = np.load(npy_files[0]).astype(np.float32)
        if sample.shape[0] > t_max:
            sample = sample[:t_max]
        return sample

    # Fallback: synthetic signal
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


# ===================================================================
# Sidebar
# ===================================================================

st.sidebar.title("Synapse Scribe")

# API status badge
api_status = api_available()
_status_class = "ss-status-online" if api_status else "ss-status-offline"
_status_label = "api connected" if api_status else "local mode"
st.sidebar.markdown(
    f'<span class="ss-status-badge {_status_class}">{_status_label}</span>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Upload & Decode",
        "Signal Viewer",
        "Decoding Visualization",
        "Benchmarks",
        "Neural Representations",
    ],
)

# Model settings
st.sidebar.markdown("---")
st.sidebar.subheader("Model")
selected_model = st.sidebar.selectbox(
    "Architecture",
    MODEL_OPTIONS,
    format_func=lambda x: MODEL_DISPLAY[x],
)
if selected_model in ("gru_decoder", "transformer"):
    st.sidebar.caption(
        "Did not converge during training. Use **CNN + LSTM** for accurate decoding."
    )
beam_width = st.sidebar.slider("Beam Width", 1, 100, 10)
use_lm = st.sidebar.checkbox("LM Rescoring", value=False)


# ===================================================================
# Page 1: Upload & Decode
# ===================================================================

def page_upload_decode():
    st.header("Upload & Decode")

    col1, col2 = st.columns([5, 2])

    with col1:
        uploaded = st.file_uploader(
            "Neural recording (.npy)",
            type=["npy"],
            help="NumPy array with shape [T, C] (timesteps x channels)",
        )

        col_btn1, col_btn2, _ = st.columns([1, 1, 2])
        with col_btn1:
            decode_btn = st.button("Decode", type="primary", disabled=uploaded is None)
        with col_btn2:
            demo_btn = st.button("Try Demo")

    with col2:
        st.markdown(
            f'<div style="padding: 0.75rem 0;">'
            f'<div class="ss-settings-row"><span class="ss-settings-key">Model</span>'
            f'<span class="ss-settings-val">{selected_model}</span></div>'
            f'<div class="ss-settings-row"><span class="ss-settings-key">Beam width</span>'
            f'<span class="ss-settings-val">{beam_width}</span></div>'
            f'<div class="ss-settings-row"><span class="ss-settings-key">LM rescore</span>'
            f'<span class="ss-settings-val">{"on" if use_lm else "off"}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Run decoding
    result = None
    features_data = None

    if decode_btn and uploaded:
        features_data = np.load(io.BytesIO(uploaded.read()))
        if features_data.ndim == 3:
            features_data = features_data[0]
        with st.spinner("Decoding..."):
            result = smart_decode(features_data, selected_model, beam_width, use_lm)

    if demo_btn:
        features_data = _generate_demo_signal()
        with st.spinner("Running demo..."):
            result = smart_decode_demo(selected_model, beam_width, use_lm)

    if result:
        st.session_state["last_result"] = result
        st.session_state["last_features"] = features_data

        st.subheader("Predicted Text")
        st.markdown(
            f'<div class="ss-result-box">'
            f'{result["predicted_text"] or "<span style=\'color:var(--text-muted)\'>(no output)</span>"}</div>',
            unsafe_allow_html=True,
        )

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Inference Time", f'{result["inference_time_ms"]:.1f} ms')
        with mcol2:
            if result["beam_hypotheses"]:
                st.metric("Top Score", f'{result["beam_hypotheses"][0]["score"]:.3f}')
        with mcol3:
            st.metric("Hypotheses", len(result["beam_hypotheses"]))

        # Raw CTC output
        with st.expander("Raw CTC output (greedy)"):
            st.code(result["raw_ctc_output"] or "(empty)")

        # Beam hypotheses
        with st.expander("Beam search hypotheses"):
            hyp_html = ""
            for i, h in enumerate(result["beam_hypotheses"]):
                hyp_html += (
                    f'<div class="ss-hypothesis">'
                    f'<span class="ss-hyp-rank">#{i+1}</span>'
                    f'<span class="ss-hyp-text">{h["text"] or "(empty)"}</span>'
                    f'<span class="ss-hyp-score">{h["score"]:.4f}</span>'
                    f'</div>'
                )
            st.markdown(hyp_html, unsafe_allow_html=True)

    # Show data from session state if available
    elif "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.info("Showing previous result. Upload new data or click Demo to decode again.")
        st.markdown(
            f'<div class="ss-result-box" style="border-left-color: var(--text-muted);">'
            f'{result["predicted_text"] or "(no output)"}</div>',
            unsafe_allow_html=True,
        )


# ===================================================================
# Page 2: Signal Viewer & Diagnostics
# ===================================================================

def page_signal_viewer():
    st.header("Signal Viewer")

    features = st.session_state.get("last_features")
    if features is None:
        uploaded = st.file_uploader("Upload .npy to view", type=["npy"], key="sig_upload")
        if uploaded:
            features = np.load(io.BytesIO(uploaded.read()))
            if features.ndim == 3:
                features = features[0]
            st.session_state["last_features"] = features
        else:
            col_e1, col_e2 = st.columns([3, 1])
            with col_e1:
                st.markdown(
                    '<div class="ss-empty-state">'
                    '<div class="ss-empty-title">No signal loaded</div>'
                    '<div class="ss-empty-desc">Upload a .npy recording above, or load a '
                    'demo signal to explore channels, heatmaps, quality diagnostics, and '
                    'power spectra.</div></div>',
                    unsafe_allow_html=True,
                )
            with col_e2:
                if st.button("Load Demo Signal"):
                    features = _generate_demo_signal()
                    st.session_state["last_features"] = features
            if features is None:
                return

    T, C = features.shape
    st.markdown(
        f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.82rem; '
        f'color:var(--text-muted);">{T} timesteps &times; {C} channels</span>',
        unsafe_allow_html=True,
    )

    # Channel selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("Channel Selection")
    max_ch_display = min(C, 20)
    channel_mode = st.sidebar.radio("Display mode", ["Select channels", "All channels"])

    if channel_mode == "Select channels":
        default_channels = list(range(min(5, C)))
        selected_channels = st.sidebar.multiselect(
            "Channels", list(range(C)),
            default=default_channels,
            format_func=lambda x: f"Ch {x}",
        )
        if not selected_channels:
            selected_channels = default_channels
    else:
        selected_channels = list(range(min(max_ch_display, C)))

    # Time range
    time_range = st.slider(
        "Time range (timesteps)", 0, T, (0, min(T, 500)),
        key="time_range",
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Time Series", "Channel Heatmap", "Channel Quality", "Power Spectrum",
    ])

    # Tab 1: Interactive time series
    with tab1:
        t_start, t_end = time_range
        fig = go.Figure()
        for ch in selected_channels:
            fig.add_trace(go.Scatter(
                x=list(range(t_start, t_end)),
                y=features[t_start:t_end, ch].tolist(),
                mode="lines",
                name=f"Ch {ch}",
                line=dict(width=1),
            ))
        fig.update_layout(
            title="Neural Signal Time Series",
            xaxis_title="Timestep",
            yaxis_title="Amplitude",
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Channel activity heatmap
    with tab2:
        t_start, t_end = time_range
        heatmap_data = features[t_start:t_end, :].T
        # Subsample for display
        max_display_t = 500
        step = max(1, heatmap_data.shape[1] // max_display_t)
        heatmap_sub = heatmap_data[:, ::step]

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_sub.tolist(),
            colorscale="RdBu_r",
            colorbar=dict(title="Amplitude"),
        ))
        fig.update_layout(
            title="Channel Activity Heatmap",
            xaxis_title="Timestep (subsampled)",
            yaxis_title="Channel",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Channel quality
    with tab3:
        channel_var = np.var(features, axis=0)
        channel_mean = np.mean(np.abs(features), axis=0)
        median_var = np.median(channel_var)
        bad_mask = (channel_var < 1e-10) | (channel_var > 10 * median_var)
        n_bad = int(np.sum(bad_mask))

        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            st.metric("Total Channels", C)
        with qcol2:
            st.metric("Good Channels", C - n_bad)
        with qcol3:
            st.metric("Bad Channels", n_bad)

        # Variance bar chart
        colors = ["red" if bad else "steelblue" for bad in bad_mask]
        fig = go.Figure(data=go.Bar(
            x=list(range(C)),
            y=channel_var.tolist(),
            marker_color=colors,
        ))
        fig.update_layout(
            title="Per-Channel Variance (red = flagged)",
            xaxis_title="Channel",
            yaxis_title="Variance",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # SNR estimation
        st.subheader("SNR Estimation")
        try:
            from src.diagnostics.snr_analysis import compute_snr
            snr_values = compute_snr(features, fs=250.0)
            fig = go.Figure(data=go.Bar(
                x=list(range(len(snr_values))),
                y=snr_values.tolist() if hasattr(snr_values, 'tolist') else list(snr_values),
            ))
            fig.update_layout(
                title="Per-Channel SNR",
                xaxis_title="Channel",
                yaxis_title="SNR (dB)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SNR computation unavailable: {e}")

    # Tab 4: Power spectrum
    with tab4:
        st.subheader("Power Spectral Density")
        psd_channels = st.multiselect(
            "Select channels for PSD",
            list(range(C)),
            default=list(range(min(3, C))),
            format_func=lambda x: f"Ch {x}",
            key="psd_ch",
        )

        if psd_channels:
            try:
                from scipy.signal import welch
                fs = 250.0
                fig = go.Figure()
                for ch in psd_channels:
                    freqs, psd = welch(features[:, ch], fs=fs, nperseg=min(256, T))
                    fig.add_trace(go.Scatter(
                        x=freqs.tolist(),
                        y=(10 * np.log10(psd + 1e-12)).tolist(),
                        mode="lines",
                        name=f"Ch {ch}",
                    ))
                fig.update_layout(
                    title="Power Spectral Density",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Power (dB)",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("scipy not available for PSD computation")


# ===================================================================
# Page 3: Decoding Visualization
# ===================================================================

def page_decoding_viz():
    st.header("Decoding Visualization")

    result = st.session_state.get("last_result")
    if result is None:
        st.markdown(
            '<div class="ss-empty-state">'
            '<div class="ss-empty-title">No decode results yet</div>'
            '<div class="ss-empty-desc">Run a decode on the Upload & Decode page to see '
            'CTC heatmaps, beam hypotheses, and method comparisons here.</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Run Demo Decode"):
            features = _generate_demo_signal()
            result = smart_decode_demo(selected_model, beam_width, use_lm)
            if result:
                st.session_state["last_result"] = result
                st.session_state["last_features"] = features
                st.rerun()
        return

    tab1, tab2, tab3 = st.tabs([
        "CTC Heatmap", "Beam Hypotheses", "Decoding Comparison",
    ])

    # Tab 1: CTC probability heatmap
    with tab1:
        char_probs = result.get("char_probabilities", [])
        if char_probs:
            probs_arr = np.array(char_probs)  # [T_sub, n_classes]
            n_classes = probs_arr.shape[1]

            # Character labels
            chars = ["_"] + [chr(ord("a") + i) for i in range(26)] + ["spc"]
            if n_classes < len(chars):
                chars = chars[:n_classes]

            fig = go.Figure(data=go.Heatmap(
                z=probs_arr.T.tolist(),
                x=list(range(probs_arr.shape[0])),
                y=chars,
                colorscale="Hot",
                colorbar=dict(title="Probability"),
            ))
            fig.update_layout(
                title="CTC Character Probabilities Over Time",
                xaxis_title="Timestep (subsampled)",
                yaxis_title="Character",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Character prediction timeline
            st.subheader("Character Prediction Timeline")
            predicted_chars = []
            prev_char = ""
            for t in range(probs_arr.shape[0]):
                best_idx = np.argmax(probs_arr[t])
                if best_idx == 0:
                    c = "_"
                elif best_idx == 27:
                    c = " "
                elif 1 <= best_idx <= 26:
                    c = chr(ord("a") + best_idx - 1)
                else:
                    c = "?"
                if c != prev_char:
                    predicted_chars.append((t, c, float(probs_arr[t, best_idx])))
                prev_char = c

            # Display as a timeline
            non_blank = [(t, c, p) for t, c, p in predicted_chars if c != "_"]
            if non_blank:
                timeline_text = '<div style="font-family:\'JetBrains Mono\',monospace; line-height:1.8; letter-spacing:0.05em;">'
                for t, c, p in non_blank:
                    # Amber (accent) for high confidence, muted for low
                    r = int(85 + 116 * p)  # 55 -> c9
                    g = int(83 + 85 * p)   # 53 -> a8
                    b = int(94 - 18 * p)   # 5e -> 4c
                    opacity = 0.45 + 0.55 * p
                    timeline_text += (
                        f'<span style="color:rgb({r},{g},{b}); opacity:{opacity:.2f}; '
                        f'font-size:1.35rem;" title="t={t}, p={p:.3f}">{c}</span>'
                    )
                timeline_text += '</div>'
                st.markdown(timeline_text, unsafe_allow_html=True)
                st.caption("Brighter = higher confidence")
        else:
            st.warning("No character probabilities available.")

    # Tab 2: Beam search hypotheses
    with tab2:
        hypotheses = result.get("beam_hypotheses", [])
        if hypotheses:
            st.subheader("Ranked Hypotheses")
            hyp_html = ""
            for i, h in enumerate(hypotheses):
                hyp_html += (
                    f'<div class="ss-hypothesis">'
                    f'<span class="ss-hyp-rank">#{i+1}</span>'
                    f'<span class="ss-hyp-text">{h["text"] or "(empty)"}</span>'
                    f'<span class="ss-hyp-score">{h["score"]:.4f}</span>'
                    f'</div>'
                )
            st.markdown(hyp_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="ss-empty-state">'
                '<div class="ss-empty-title">No hypotheses</div>'
                '<div class="ss-empty-desc">Beam search did not produce any hypotheses for this input.</div></div>',
                unsafe_allow_html=True,
            )

    # Tab 3: Decoding comparison
    with tab3:
        st.subheader("Decoding Method Comparison")

        greedy_text = result.get("raw_ctc_output", "")
        beam_text = result["beam_hypotheses"][0]["text"] if result.get("beam_hypotheses") else ""
        predicted = result.get("predicted_text", "")

        comp_data = {
            "Method": ["Greedy CTC", "Beam Search", "Beam + LM" if use_lm else "Best Result"],
            "Output": [greedy_text or "(empty)", beam_text or "(empty)", predicted or "(empty)"],
        }

        for i, (method, output) in enumerate(zip(comp_data["Method"], comp_data["Output"])):
            best = " ss-best" if i == len(comp_data["Method"]) - 1 else ""
            st.markdown(
                f'<div class="ss-comparison-row{best}">'
                f'<div class="ss-comparison-label">{method}</div>'
                f'<div class="ss-comparison-text">{output}</div></div>',
                unsafe_allow_html=True,
            )


# ===================================================================
# Page 4: Benchmarks
# ===================================================================

def page_benchmarks():
    st.header("Benchmarks")

    st.subheader("Model Architectures")

    cols = st.columns(len(MODEL_OPTIONS))
    for i, model_type in enumerate(MODEL_OPTIONS):
        with cols[i]:
            info = call_model_info(model_type) if api_available() else None
            st.markdown(f"**{MODEL_DISPLAY[model_type]}**")
            if info:
                st.metric("Parameters", f"{info['parameter_count']:,}")
            else:
                # Show param count from local model
                try:
                    import torch
                    if model_type == "gru_decoder":
                        from src.models.gru_decoder import GRUDecoder
                        m = GRUDecoder()
                    elif model_type == "cnn_lstm":
                        from src.models.cnn_lstm import CNNLSTM
                        m = CNNLSTM()
                    elif model_type == "transformer":
                        from src.models.transformer import TransformerDecoder
                        m = TransformerDecoder()
                    else:
                        from src.models.cnn_transformer import CNNTransformer
                        m = CNNTransformer()
                    st.metric("Parameters", f"{m.count_parameters():,}")
                except Exception:
                    st.metric("Parameters", "N/A")

    # Benchmark table
    st.subheader("Performance Comparison")

    # Check for saved results
    results_dir = Path("outputs/results")
    results_files = list(results_dir.glob("*.json")) if results_dir.exists() else []

    if results_files:
        import json
        all_results = []
        for rf in results_files:
            try:
                with open(rf) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "cer" in data:
                        all_results.append(data)
            except Exception:
                pass

        if all_results:
            import pandas as pd
            df = pd.DataFrame(all_results)
            st.dataframe(df, use_container_width=True)
    else:
        st.markdown(
            '<div class="ss-empty-state">'
            '<div class="ss-empty-title">No benchmark results saved</div>'
            '<div class="ss-empty-desc">Run evaluations to populate this table, or use '
            'the quick benchmark below to compare models on a demo signal.</div></div>',
            unsafe_allow_html=True,
        )

        # Show placeholder comparison
        st.markdown("**Expected architecture comparison (from task list):**")
        placeholder = {
            "Model": ["GRU Decoder", "CNN+LSTM", "Transformer", "CNN-Transformer"],
            "Type": ["RNN", "CNN+RNN", "Attention", "Hybrid"],
            "Temporal Reduction": ["1x", "1x", "1x", "8x"],
            "Bidirectional": ["No", "Yes", "N/A (self-attn)", "N/A (self-attn)"],
        }
        import pandas as pd
        st.table(pd.DataFrame(placeholder))

    # CER/WER bar charts
    st.subheader("Metrics Visualization")

    # Run a quick benchmark if user wants
    if st.button("Run Quick Benchmark (all models on demo)"):
        demo_features = _generate_demo_signal()
        results = {}
        progress = st.progress(0)
        for i, mt in enumerate(MODEL_OPTIONS):
            with st.spinner(f"Running {MODEL_DISPLAY[mt]}..."):
                r = _local_decode(demo_features, mt, beam_width=5, use_lm=False)
                results[mt] = {
                    "inference_ms": r["inference_time_ms"],
                    "output": r["predicted_text"],
                }
            progress.progress((i + 1) / len(MODEL_OPTIONS))

        # Display latency comparison
        fig = go.Figure(data=go.Bar(
            x=[MODEL_DISPLAY[m] for m in results],
            y=[results[m]["inference_ms"] for m in results],
            marker_color=["#00d2ff", "#ff6384", "#36a2eb", "#ffcd56"],
        ))
        fig.update_layout(
            title="Inference Latency (ms) - Demo Sample",
            yaxis_title="Milliseconds",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Output comparison
        st.subheader("Model Outputs")
        for mt, r in results.items():
            st.markdown(f"**{MODEL_DISPLAY[mt]}:** `{r['output'] or '(empty)'}` ({r['inference_ms']:.1f} ms)")

    # Model selector for detailed view
    st.subheader("Detailed Model View")
    detail_model = st.selectbox(
        "Select model for details",
        MODEL_OPTIONS,
        format_func=lambda x: MODEL_DISPLAY[x],
        key="bench_model",
    )

    info = call_model_info(detail_model) if api_available() else None
    if info:
        st.json(info)
    else:
        st.markdown(f"Architecture: **{MODEL_DISPLAY[detail_model]}**")


# ===================================================================
# Page 5: Neural Representations
# ===================================================================

def page_neural_representations():
    st.header("Neural Representations")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Embedding Scatter", "Neural Trajectories",
        "Electrode Importance", "Trial Similarity",
    ])

    # Check for saved embeddings
    emb_dir = Path("outputs/embeddings")
    emb_files = list(emb_dir.glob("*.npz")) if emb_dir.exists() else []

    # Tab 1: Embedding scatter
    with tab1:
        st.subheader("Embedding Visualization (t-SNE / PCA / UMAP)")

        if emb_files:
            selected_emb = st.selectbox(
                "Embedding file",
                emb_files,
                format_func=lambda x: x.stem,
            )
            data = np.load(selected_emb, allow_pickle=True)
            embeddings = data["embeddings"]
            labels = data["labels"].tolist()

            method = st.radio("Reduction method", ["PCA", "t-SNE"], horizontal=True)

            if embeddings.shape[0] > 0:
                from sklearn.decomposition import PCA
                if method == "PCA":
                    reducer = PCA(n_components=2)
                    coords = reducer.fit_transform(embeddings)
                else:
                    from sklearn.manifold import TSNE
                    perp = min(30, max(5, embeddings.shape[0] - 1))
                    reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
                    coords = reducer.fit_transform(embeddings)

                fig = px.scatter(
                    x=coords[:, 0], y=coords[:, 1],
                    color=labels,
                    title=f"Embeddings ({method})",
                    labels={"x": f"{method} 1", "y": f"{method} 2", "color": "Label"},
                    hover_data={"Label": labels},
                    height=600,
                )
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No embeddings found in file.")
        else:
            st.markdown(
                '<div class="ss-empty-state">'
                '<div class="ss-empty-title">No embeddings saved</div>'
                '<div class="ss-empty-desc">Generate embeddings from a model to explore '
                'learned representations with PCA or t-SNE projections.</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Generate Demo Embeddings"):
                with st.spinner("Generating embeddings from demo data..."):
                    _generate_demo_embeddings()
                st.rerun()

    # Tab 2: Neural trajectories
    with tab2:
        st.subheader("Neural State Trajectories")

        features = st.session_state.get("last_features")
        if features is not None:
            try:
                import torch
                from src.models.cnn_lstm import CNNLSTM

                model = CNNLSTM()
                ckpt = Path("outputs/checkpoints/GPU-3-13/CNNLSTM_best.pt")
                if ckpt.exists():
                    state = torch.load(ckpt, map_location="cpu", weights_only=True)
                    key = "model_state_dict" if "model_state_dict" in state else None
                    model.load_state_dict(state[key] if key else state)

                model.eval()

                # Get hidden states
                tensor = torch.from_numpy(features[np.newaxis, ...].astype(np.float32))
                captured = []

                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    captured.append(out.detach().cpu().numpy())

                # Hook into LSTM
                handle = None
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.LSTM):
                        handle = module.register_forward_hook(hook_fn)
                        break

                with torch.no_grad():
                    _ = model(tensor)

                if handle:
                    handle.remove()

                if captured:
                    hidden = captured[0][0]  # [T, D]
                    # PCA to 3D
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=3)
                    coords = pca.fit_transform(hidden[:min(500, len(hidden))])

                    fig = go.Figure(data=go.Scatter3d(
                        x=coords[:, 0].tolist(),
                        y=coords[:, 1].tolist(),
                        z=coords[:, 2].tolist(),
                        mode="lines+markers",
                        marker=dict(
                            size=2,
                            color=list(range(len(coords))),
                            colorscale="Viridis",
                            colorbar=dict(title="Timestep"),
                        ),
                        line=dict(width=2, color="rgba(100,100,255,0.3)"),
                    ))
                    fig.update_layout(
                        title="Neural State Trajectory (PCA 3D)",
                        scene=dict(
                            xaxis_title="PC1",
                            yaxis_title="PC2",
                            zaxis_title="PC3",
                        ),
                        height=600,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Color indicates timestep progression (purple=start, yellow=end)")
                else:
                    st.warning("Could not capture hidden states.")
            except Exception as e:
                st.warning(f"Trajectory visualization error: {e}")
        else:
            st.markdown(
                '<div class="ss-empty-state">'
                '<div class="ss-empty-title">No signal loaded</div>'
                '<div class="ss-empty-desc">Load a recording on the Upload & Decode or '
                'Signal Viewer page to visualize hidden-state trajectories in 3D.</div></div>',
                unsafe_allow_html=True,
            )

    # Tab 3: Electrode importance
    with tab3:
        st.subheader("Electrode Importance (Gradient Attribution)")

        features = st.session_state.get("last_features")
        if features is not None:
            target_char = st.selectbox(
                "Target character (or 'auto' for argmax)",
                ["auto"] + list("abcdefghijklmnopqrstuvwxyz") + ["space"],
                key="saliency_char",
            )

            if st.button("Compute Attribution", key="compute_attr"):
                with st.spinner("Computing gradient attribution..."):
                    try:
                        import torch
                        from src.models.cnn_lstm import CNNLSTM
                        from src.analysis.saliency import input_x_gradient, electrode_importance

                        model = CNNLSTM()
                        ckpt = Path("outputs/checkpoints/GPU-3-13/CNNLSTM_best.pt")
                        if ckpt.exists():
                            state = torch.load(ckpt, map_location="cpu", weights_only=True)
                            key = "model_state_dict" if "model_state_dict" in state else None
                            model.load_state_dict(state[key] if key else state)

                        target_cls = None
                        if target_char != "auto":
                            if target_char == "space":
                                target_cls = 27
                            else:
                                target_cls = ord(target_char) - ord("a") + 1

                        attr = input_x_gradient(model, features, target_class=target_cls)
                        importance = electrode_importance(attr)

                        # Importance bar chart
                        fig = go.Figure(data=go.Bar(
                            x=list(range(len(importance))),
                            y=importance.tolist(),
                            marker_color=px.colors.sequential.Hot[
                                : min(len(importance), len(px.colors.sequential.Hot))
                            ] if len(importance) <= 10 else "indianred",
                        ))
                        fig.update_layout(
                            title=f"Electrode Importance (target: {target_char})",
                            xaxis_title="Electrode",
                            yaxis_title="Importance Score",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Attribution heatmap
                        max_t = min(attr.shape[0], 500)
                        fig2 = go.Figure(data=go.Heatmap(
                            z=np.abs(attr[:max_t]).T.tolist(),
                            colorscale="Hot",
                            colorbar=dict(title="|Attribution|"),
                        ))
                        fig2.update_layout(
                            title="Attribution Heatmap (time x electrode)",
                            xaxis_title="Timestep",
                            yaxis_title="Electrode",
                            height=500,
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    except Exception as e:
                        st.error(f"Attribution failed: {e}")
        else:
            st.markdown(
                '<div class="ss-empty-state">'
                '<div class="ss-empty-title">No signal loaded</div>'
                '<div class="ss-empty-desc">Load a recording first to compute gradient '
                'attribution and electrode importance maps.</div></div>',
                unsafe_allow_html=True,
            )

    # Tab 4: Trial similarity
    with tab4:
        st.subheader("Trial Similarity Matrix")

        if emb_files:
            selected_emb = st.selectbox(
                "Embedding file",
                emb_files,
                format_func=lambda x: x.stem,
                key="sim_emb",
            )
            data = np.load(selected_emb, allow_pickle=True)
            embeddings = data["embeddings"]
            labels = data["labels"].tolist()

            if embeddings.shape[0] > 0:
                from src.analysis.similarity_matrix import (
                    compute_cosine_similarity,
                    compute_class_similarity,
                )

                view_mode = st.radio(
                    "View mode",
                    ["Class-averaged similarity", "Full trial similarity"],
                    horizontal=True,
                    key="sim_mode",
                )

                if view_mode == "Class-averaged similarity":
                    class_sim, class_labels = compute_class_similarity(embeddings, labels)
                    fig = go.Figure(data=go.Heatmap(
                        z=class_sim.tolist(),
                        x=class_labels,
                        y=class_labels,
                        colorscale="RdBu_r",
                        zmin=-1, zmax=1,
                        colorbar=dict(title="Cosine Similarity"),
                    ))
                    fig.update_layout(
                        title="Character-Grouped Similarity",
                        height=600,
                    )
                else:
                    sim = compute_cosine_similarity(embeddings[:100])  # limit for perf
                    disp_labels = labels[:100]
                    fig = go.Figure(data=go.Heatmap(
                        z=sim.tolist(),
                        colorscale="RdBu_r",
                        zmin=-1, zmax=1,
                        colorbar=dict(title="Cosine Similarity"),
                    ))
                    fig.update_layout(
                        title="Trial-Level Similarity Matrix",
                        height=600,
                    )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(
                '<div class="ss-empty-state">'
                '<div class="ss-empty-title">No embeddings for similarity</div>'
                '<div class="ss-empty-desc">Generate or load embeddings to visualize '
                'trial-level and class-averaged cosine similarity matrices.</div></div>',
                unsafe_allow_html=True,
            )


def _generate_demo_embeddings():
    """Generate and save demo embeddings using the CNN-LSTM model on synthetic data."""
    import torch
    from src.models.cnn_lstm import CNNLSTM

    model = CNNLSTM()
    ckpt = Path("outputs/checkpoints/GPU-3-13/CNNLSTM_best.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        key = "model_state_dict" if "model_state_dict" in state else None
        model.load_state_dict(state[key] if key else state)
    model.eval()

    # Generate synthetic trials
    rng = np.random.RandomState(42)
    n_trials = 50
    labels = [chr(ord("a") + i % 26) for i in range(n_trials)]
    embeddings = []

    for i in range(n_trials):
        x = rng.randn(1, 200, 192).astype(np.float32)
        tensor = torch.from_numpy(x)

        captured = []
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            captured.append(out.detach().cpu().numpy())

        handle = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LSTM):
                handle = module.register_forward_hook(hook_fn)
                break

        with torch.no_grad():
            _ = model(tensor)

        if handle:
            handle.remove()
        if captured:
            emb = captured[0][0].mean(axis=0)  # mean pool over time
            embeddings.append(emb)

    if embeddings:
        emb_arr = np.stack(embeddings)
        out_dir = Path("outputs/embeddings")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "demo_embeddings.npz",
            embeddings=emb_arr,
            labels=np.array(labels, dtype=object),
            layer_name="gru",
        )


# ===================================================================
# Route to selected page
# ===================================================================

PAGES = {
    "Upload & Decode": page_upload_decode,
    "Signal Viewer": page_signal_viewer,
    "Decoding Visualization": page_decoding_viz,
    "Benchmarks": page_benchmarks,
    "Neural Representations": page_neural_representations,
}

PAGES[page]()
