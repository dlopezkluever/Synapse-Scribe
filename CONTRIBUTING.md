# Contributing

## Prerequisites

- Python 3.10 or later
- pip (latest recommended)
- Git

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd BCI-2

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA before running `pip install -r requirements.txt`. See https://pytorch.org/get-started/locally/.

## Running Tests

```bash
# Run the full suite (383+ tests)
pytest

# Run a single module
pytest tests/test_models.py -v

# Run tests matching a keyword
pytest -k "beam_search" -v
```

Test configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Code Structure

| Directory | Purpose |
|-----------|---------|
| `src/config.py` | Central `Config` dataclass -- all pipeline parameters |
| `src/data/` | Data loading (Willett, UCSF ECoG, OpenNeuro), dataset class, augmentation transforms |
| `src/diagnostics/` | Signal quality checks -- channel quality, SNR, spectral, trial quality |
| `src/preprocessing/` | Bandpass/notch filtering, z-score normalization, segmentation |
| `src/features/` | Feature extraction -- firing-rate binning, temporal convolution, linear projection |
| `src/models/` | Decoder architectures -- GRU, CNN+LSTM, Transformer, CNN-Transformer (all inherit `BaseDecoder`) |
| `src/training/` | Training loop (`Trainer`), CTC loss wrapper, LR scheduler |
| `src/decoding/` | Greedy decode, beam search, KenLM language model correction |
| `src/evaluation/` | CER/WER metrics, ablation runner |
| `src/analysis/` | Embedding extraction, neural trajectories, similarity matrices, saliency maps |
| `src/visualization/` | Plotting utilities for signals, CTC outputs, embeddings |
| `app/api.py` | FastAPI backend -- `/health`, `/model/info`, `/decode`, `/decode/demo` |
| `app/dashboard.py` | Streamlit frontend -- 5-page interactive demo |
| `scripts/` | CLI entrypoints for training, evaluation, and quality checks |
| `tests/` | One test module per source module |

## Pull Request Guidelines

1. **Branch from `main`** -- create a feature branch (`feature/your-change`) or bugfix branch (`fix/your-change`).
2. **Keep changes focused** -- one logical change per PR.
3. **Add or update tests** -- new functionality should have corresponding tests in `tests/`.
4. **Run `pytest` before pushing** -- all tests must pass.
5. **Write a clear PR description** -- summarize what changed and why.
