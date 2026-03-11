# Implementation Task List — Brain → Text Decoder Simulator

> Iterative development plan from barebones setup → MVP → feature-rich polished product.
> Each phase delivers a functional, testable state of the project.

---

## Phase 0: Project Setup (Barebones Foundation)

**Goal:** A running project skeleton with environment, data downloaded, and basic signal loading verified. Nothing decodes yet, but the scaffolding is in place and you can load and visualize raw neural data.

---

### 0.1 Repository & Environment Setup

- [ ] Initialize git repo with `.gitignore` (ignore `data/`, `outputs/`, `__pycache__/`, `*.pyc`, `.env`)
- [ ] Create `environment.yml` with core dependencies: `python=3.10`, `pytorch>=2.1`, `numpy`, `scipy`, `mne`, `pandas`, `matplotlib`
- [ ] Create `requirements.txt` mirroring the conda environment for pip-based installs
- [ ] Create the full directory structure from Section 8 of the PRD (all `src/`, `app/`, `notebooks/`, `scripts/`, `tests/`, `outputs/` folders with `__init__.py` files)
- [ ] Verify environment activates and `import torch; import mne; import scipy` all succeed

### 0.2 Global Configuration Module

- [ ] Create `src/config.py` with a dataclass holding all configurable parameters (dataset name, paths, split ratios, preprocessing defaults, model hyperparameters)
- [ ] Support loading overrides from a `config.yaml` file (use `dataclasses` + `yaml` or `pydantic`)
- [ ] Add separate preset configs for `willett_handwriting` and `ecog_speech` (different bandpass, T_MAX, etc.)
- [ ] Validate config on load (e.g., split ratios sum to 1.0, paths are valid)

### 0.3 Willett Handwriting Dataset Download & Parsing

- [ ] Create `src/data/loader.py` with a `download_willett()` function that fetches the dataset from the Dryad DOI URL
- [ ] Implement checksum verification after download to confirm file integrity
- [ ] Parse `.mat` files using `scipy.io.loadmat`; extract neural signals and character labels per trial
- [ ] Convert to standardized internal format: `data/willett_handwriting/sub-{id}/ses-{s}/neural/trial_{n}_signals.npy` + `labels/trial_{n}_transcript.txt`
- [ ] Build the trial index DataFrame: `[subject, session, trial_id, signal_path, label_path, n_timesteps, n_channels]`

### 0.4 Basic Signal Visualization

- [ ] Create `src/visualization/signal_plots.py` with a function to plot raw multi-channel time series for a single trial (matplotlib)
- [ ] Add a channel heatmap function: `[time × channels]` color-coded by amplitude
- [ ] Create `notebooks/01_data_exploration.ipynb` that loads 5 sample trials and displays raw signals + labels
- [ ] Verify signal shapes, sampling rates, and label content match expected values from the Willett paper

### 0.5 Basic Test Scaffold

- [ ] Create `tests/test_loader.py` with a test that the trial index DataFrame has the correct columns and non-zero rows
- [ ] Create `tests/test_config.py` with a test that the default config loads without errors
- [ ] Add a `pytest.ini` or `pyproject.toml` section for test configuration
- [ ] Verify `pytest` runs and passes from the project root

---

## Phase 0.6: Neural Recording Diagnostics & Signal QC

**Goal:** Before any decoding, automatically evaluate whether neural recordings are usable. Generate per-session quality reports with channel diagnostics, SNR estimates, spectral analysis, and trial quality metrics. This mirrors the first step in real BCI lab pipelines (e.g., BrainGate, UCSF, Stanford NPTL).

---

### 0.6.1 Channel Quality Detection

- [ ] Create `src/diagnostics/channel_quality.py` with `detect_bad_channels()`: flag channels with zero variance, variance > 10× median, excessive line noise, or flatline segments
- [ ] Output a summary: total channels, good channels, bad channels, and per-channel quality labels
- [ ] Create a channel variance heatmap visualization (192-channel grid color-coded by variance)
- [ ] Write `tests/test_channel_quality.py`: verify detection on synthetic data with known bad channels

### 0.6.2 Signal-to-Noise Ratio (SNR) Analysis

- [ ] Create `src/diagnostics/snr_analysis.py` with `compute_snr()`: estimate SNR as `power(signal_band) / power(noise_band)` per channel
- [ ] Support configurable signal band (default: 70–150 Hz high gamma) and noise band (default: 55–65 Hz line noise)
- [ ] Output per-channel SNR values and flag low-quality electrodes (SNR < configurable threshold)
- [ ] Create SNR distribution histogram visualization
- [ ] Write `tests/test_snr.py`: verify SNR computation on synthetic signals with known power ratios

### 0.6.3 Power Spectrum Analysis

- [ ] Create `src/diagnostics/spectral_analysis.py` with `compute_psd()`: power spectral density per channel using Welch's method (`scipy.signal.welch`)
- [ ] Generate frequency-vs-power plots for individual channels and session-average
- [ ] Detect presence of expected neural bands (high gamma) and contamination (60 Hz line noise + harmonics)
- [ ] Write `tests/test_spectral.py`: verify PSD output shapes and frequency resolution

### 0.6.4 Trial Quality Detection

- [ ] Create `src/diagnostics/trial_quality.py` with `detect_bad_trials()`: flag trials with variance > threshold or amplitude spikes exceeding threshold
- [ ] Output summary: total trials, usable trials, rejected trials, rejection reasons per trial
- [ ] Create trial quality histogram visualization (distribution of per-trial variance)
- [ ] Write `tests/test_trial_quality.py`: verify detection on synthetic trials with injected artifacts

### 0.6.5 Channel Correlation Analysis

- [ ] Create `src/diagnostics/correlation_analysis.py` with `compute_channel_correlation()`: compute pairwise correlation matrix across all channels
- [ ] Flag highly correlated channel pairs (> 0.95) indicating reference contamination, amplifier noise, or cable problems
- [ ] Create 192×192 correlation heatmap visualization
- [ ] Write `tests/test_correlation.py`: verify correlation matrix properties (symmetric, diagonal = 1)

### 0.6.6 Quality Report Generator

- [ ] Create `src/diagnostics/report_generator.py` with `generate_quality_report()`: run all diagnostics and produce a structured report
- [ ] Output `summary.json` with all metrics (channel counts, SNR, trial counts, line noise level)
- [ ] Save all diagnostic plots to `outputs/quality_reports/session_{id}/` (channel_variance.png, power_spectrum.png, snr_distribution.png, trial_quality_histogram.png, channel_correlation_matrix.png)
- [ ] Create `scripts/run_quality_check.py` CLI entrypoint: `python scripts/run_quality_check.py --dataset willett`
- [ ] Create `notebooks/02_signal_diagnostics.ipynb`: run full diagnostics on loaded data, display report inline

---

## Phase 1: MVP (End-to-End Decoding Pipeline)

**Goal:** A complete, working pipeline that takes raw Willett neural data, preprocesses it, trains a CNN+LSTM model and a Willett-style GRU decoder with CTC loss, and outputs decoded character strings. Greedy decoding produces readable (if imperfect) text. CER is reported on a held-out test set.

---

### 1.1 Signal Preprocessing — Filtering

- [ ] Implement `src/preprocessing/filter.py` with `bandpass_filter()`: 4th-order Butterworth via `scipy.signal.butter` + `filtfilt`, configurable low/high cutoffs
- [ ] Implement `notch_filter()`: 60 Hz + 120 Hz harmonic removal via `scipy.signal.iirnotch`
- [ ] Add gross artifact rejection: flag and exclude trials with > 3× median channel variance
- [ ] Write `tests/test_filter.py`: verify filtered signal shape is preserved, frequency content is attenuated outside passband (check with FFT)

### 1.2 Signal Preprocessing — Normalization & Channel Rejection

- [ ] Implement `src/preprocessing/normalize.py` with `zscore_normalize()`: per-channel z-score using training-set statistics only
- [ ] Clip normalized values to `[-5, 5]`
- [ ] Consume bad channel list from Phase 0.6 diagnostics output; remove flagged channels and record removed indices in trial metadata
- [ ] If diagnostics have not been run, fall back to inline detection: flag channels with zero variance or > 10× session median variance
- [ ] Write `tests/test_normalize.py`: verify output is zero-mean unit-variance per channel on training data

### 1.3 Signal Preprocessing — Downsampling & Segmentation

- [ ] Implement temporal downsampling in `src/preprocessing/filter.py` using `scipy.signal.decimate` to target rate (default 250 Hz)
- [ ] Implement `src/preprocessing/segment.py` with `segment_trials()`: extract per-trial windows using onset/offset annotations with configurable padding (100 ms pre, 200 ms post)
- [ ] Pad or truncate all trials to fixed `T_MAX` (default 2000 timesteps)
- [ ] Create `notebooks/03_preprocessing_qc.ipynb`: visualize before/after filtering, show power spectra, confirm downsampling

### 1.3b Neural Temporal Smoothing

- [ ] Implement `GaussianTemporalSmoothing` in `src/preprocessing/filter.py`: apply Gaussian kernel smoothing (configurable σ, default 20–40 ms) across the time axis of spike count data
- [ ] This converts noisy spike counts into smooth firing rate estimates — a standard step in real neural decoding pipelines
- [ ] Apply after downsampling but before feature extraction
- [ ] Write test in `tests/test_filter.py`: verify smoothed output has same shape and reduced high-frequency variance compared to input

### 1.4 Feature Extraction — Firing Rate Binning (Pathway C)

> **Pathway–Model mapping:**
> | Pathway | What It Does | Used By |
> |---------|-------------|---------|
> | **C** — Firing Rate Binning | Bins Willett spike counts → sqrt transform | **All models** when using Willett data (data-level step applied before any model pathway) |
> | Model A's **built-in linear projection** | Projects 192 → 256, ReLU, Dropout | **Model A (GRU Decoder)** — has its own input stage, does NOT use Pathway A or B |
> | **A** — Temporal Convolution | Multi-scale 1D conv bank over time axis | **Model B (CNN+LSTM)** — standalone feature front-end (also used in ablation studies with other models) |
> | **B** — Linear Projection + Positional Encoding | Projects channel vectors to d_model | **Model C (Transformer)** — its input embedding layer |
> | Model D's **built-in CNN front-end** | 3-layer CNN with stride-2 pooling (reduces T by 4×) | **Model D (Hybrid CNN-Transformer)** — has its own feature extractor, does NOT use Pathway A or B |
>
> Pathway C is a **data preprocessing** step for the Willett dataset specifically. Pathways A and B are **model input stages** — each model uses exactly one. Models A and D have their own built-in input stages instead of using a separate pathway.

- [ ] Implement `src/features/firing_rate.py` with `bin_firing_rates()`: bin spike counts into 10 ms non-overlapping windows
- [ ] Apply square-root transform to stabilize variance: `sqrt(firing_rate)`
- [ ] Output shape: `[n_bins, n_electrodes]` per trial — this feeds into **all models** as the Willett-specific data representation
- [ ] Write `tests/test_features.py`: verify output shapes and that square-root transform produces non-negative values

### 1.5 PyTorch Dataset & DataLoader

- [ ] Implement `src/data/dataset.py` with a `NeuralTrialDataset(torch.utils.data.Dataset)` class
- [ ] `__getitem__` loads a single trial's preprocessed features + label (character index sequence)
- [ ] Define the character vocabulary: `{blank=0, a=1, ..., z=26, space=27}` — 28 classes
- [ ] Implement a custom `collate_fn` that pads variable-length sequences and returns `(padded_features, targets, input_lengths, target_lengths)` for CTC
- [ ] Create train/val/test split from the trial index DataFrame (default 80/10/10, stratified by subject)

### 1.6 Willett-Style GRU Decoder (Model A — Primary Baseline)

> **Replicates the architecture from the Willett handwriting BCI paper.** Uses Pathway C output (firing-rate-binned Willett data) as input. This is the primary baseline because it matches the known high-performance architecture for this dataset.

- [ ] Implement `src/models/base.py` with an abstract base class: `forward(x: [B,T,C]) → logits: [B,T,n_classes]`
- [ ] Implement `src/models/gru_decoder.py`: Linear projection (192 → 256, ReLU, Dropout) → 3-layer unidirectional GRU (512 hidden) → Linear → logits
- [ ] Verify forward pass produces correct output shape with a dummy input `[4, 2000, 192]`
- [ ] Write `tests/test_models.py`: check output shape, check gradient flow (loss.backward() doesn't error)

### 1.6b CNN+LSTM Model (Model B — Alternative Baseline)

> **Uses Pathway C output** (firing-rate-binned Willett data) as input. The Conv1D blocks inside Model B serve as its temporal feature extraction front-end — this is distinct from Pathway A (the standalone conv bank), which is an alternative feature extractor benchmarked in Phase 3 ablations.

- [ ] Implement `src/models/cnn_lstm.py`: 3-layer Conv1D blocks (256 channels, kernel 7, BatchNorm + ReLU) → Bidirectional LSTM (512 hidden, 2 layers, dropout 0.5) → Linear → logits
- [ ] Verify forward pass produces correct output shape with a dummy input `[4, 2000, 192]`
- [ ] Add to `tests/test_models.py`: check output shape and gradient flow

### 1.7 CTC Loss & Training Loop

- [ ] Implement `src/training/ctc_loss.py`: wrapper around `torch.nn.CTCLoss(reduction='mean', zero_infinity=True)` that handles the log-softmax and time-first transpose
- [ ] Implement `src/training/scheduler.py`: cosine annealing with linear warmup (500 steps)
- [ ] Implement `src/training/trainer.py` with a `Trainer` class:
  1. Train loop: forward → CTC loss → backward → gradient clip (max norm 1.0) → optimizer step
  2. Validation loop: compute CTC loss + greedy decode CER on val set each epoch
  3. Checkpointing: save best model (by val CER) to `outputs/checkpoints/`
  4. Early stopping: patience = 20 epochs on val CER
- [ ] Support mixed precision (fp16) via `torch.cuda.amp` when GPU available
- [ ] Create `scripts/train.py` CLI entrypoint: `python scripts/train.py --model cnn_lstm --dataset willett --epochs 200`

### 1.8 Greedy CTC Decoding

- [ ] Implement `src/decoding/greedy.py` with `greedy_decode(logits) → str`: argmax per timestep → collapse repeats → remove blanks
- [ ] Handle edge cases: all-blank output, repeated characters (e.g., "ll" in "hello" requires a blank separator)
- [ ] Integrate into the Trainer's validation loop to compute decoded strings per trial
- [ ] Write `tests/test_decoding.py`: verify known logit sequences produce expected strings (e.g., `h h _ e l l _ o` → `"helo"`)

### 1.9 Evaluation Metrics

- [ ] Implement `src/evaluation/metrics.py` with `compute_cer()` and `compute_wer()` using the `jiwer` library
- [ ] Implement `exact_match_accuracy()`: fraction of trials with CER = 0
- [ ] Create `scripts/evaluate.py` CLI entrypoint: load checkpoint → run inference on test set → print CER, WER, exact match
- [ ] Log per-trial predictions vs. ground truth to a CSV in `outputs/results/`

### 1.10 Data Augmentation

- [ ] Implement `src/data/transforms.py` with three augmentation classes (applied only during training):
  1. `TimeMasking`: randomly zero out 1–3 time windows of 10–50 ms
  2. `ChannelDropout`: randomly zero out 5–10% of channels per sample
  3. `GaussianNoise`: add noise with σ = 0.01
- [ ] Integrate augmentations into `NeuralTrialDataset` with a `train=True/False` flag
- [ ] Write `tests/test_transforms.py`: verify shapes are preserved and augmentation actually modifies the data

### 1.11 MVP Training Run & Validation

- [ ] Run a full training of the Willett-style GRU decoder on Willett data (single subject) and verify CTC loss converges (loss decreasing over epochs)
- [ ] Run a full training of the CNN+LSTM on the same data for baseline comparison
- [ ] Monitor blank token probability — if model outputs all blanks, debug label lengths vs. input lengths
- [ ] Report val CER after training — target: CER < 40% (minimum success criterion)
- [ ] Compare GRU decoder vs. CNN+LSTM on the same val set — document which is closer to known Willett paper results
- [ ] Create `notebooks/04_model_training.ipynb`: plot training/val loss curves for both models, sample decoded outputs vs. ground truth
- [ ] Save the best checkpoints for use in later phases

---

## Phase 2: Enhanced Models & Decoding

**Goal:** Add the Transformer model and hybrid CNN-Transformer as alternative architectures. Implement beam search decoding and KenLM language model correction. Compare all four models on the same test set.

---

### 2.1 Feature Extraction — Temporal Convolution (Pathway A)

> **Standalone feature extractor** — used as the default front-end for Model B (CNN+LSTM) and as an alternative front-end for other models in ablation studies. Also usable with any model to test whether the multi-scale conv bank outperforms Model B's built-in single-kernel Conv1D blocks.

- [ ] Implement `src/features/temporal_conv.py`: 1D temporal convolution bank with kernel sizes [3, 7, 15]
- [ ] Concatenate activations from all kernels → `[T, C × n_kernels]`
- [ ] Apply BatchNorm + ReLU after convolution
- [ ] Add optional max pooling (stride 2) to reduce sequence length
- [ ] Write tests verifying output shapes for various input dimensions

### 2.2 Feature Extraction — Linear Projection (Pathway B)

> **Input embedding for Model B (Transformer).** This is the Transformer's equivalent of patch embedding — each timestep's channel vector becomes a token projected into d_model space.

- [ ] Implement `src/features/projection.py`: learned linear projection `[T, C] → [T, d_model]`
- [ ] Implement sinusoidal positional encoding (standard Transformer PE)
- [ ] Apply positional encoding after projection
- [ ] Write tests verifying output shape `[B, T, d_model]` and that positional encodings vary across positions

### 2.3 Transformer Encoder Model (Model C)

> **Uses Pathway B** (linear projection + positional encoding) as its input stage. Receives Pathway C output (firing-rate-binned data) when running on Willett data.

- [ ] Implement `src/models/transformer.py`: integrate Pathway B (projection.py) as the input layer → N Transformer encoder layers (d_model=512, 8 heads, 6 layers, FFN=2048) → Linear → logits
- [ ] Use `torch.nn.TransformerEncoderLayer` with `batch_first=True`
- [ ] Add a causal or padding mask to handle variable-length sequences properly
- [ ] Verify forward pass with dummy input `[4, 2000, 192]` → `[4, 2000, 28]`
- [ ] Write tests for shape correctness and gradient flow

### 2.4 Hybrid CNN-Transformer Model (Model D)

> **Does NOT use Pathway A or B.** Model D has its own integrated CNN front-end that acts as both the feature extractor AND sequence length reducer (T → T/4). This reduced-length sequence then feeds into a smaller Transformer encoder (4 layers instead of 6).

- [ ] Implement `src/models/cnn_transformer.py`: 3-layer CNN front-end with stride-2 pooling (reduces T by 4×) → 4-layer Transformer encoder → Linear → logits
- [ ] CNN front-end: Conv1D + BatchNorm + ReLU + MaxPool(2) × 3 layers
- [ ] Verify sequence length reduction: input T=2000 → CNN output T=250 → Transformer → logits `[B, 250, 28]`
- [ ] Write tests for end-to-end shape and gradient flow

### 2.5 Beam Search Decoding

- [ ] Implement `src/decoding/beam_search.py` with `beam_search_decode(logits, beam_width=100) → List[Hypothesis]`
- [ ] Each `Hypothesis` contains: `text: str`, `score: float`, `char_probs: List[float]`
- [ ] Attempt `ctcdecode` library integration; if build fails, implement a pure-Python prefix beam search as fallback
- [ ] Return top-k hypotheses ranked by log-probability
- [ ] Write tests: verify beam search produces equal or better results than greedy on known examples

### 2.6 KenLM Language Model Integration

- [ ] Train a character-level 5-gram KenLM model on a text corpus (e.g., Wikipedia dump or similar CC-licensed corpus)
- [ ] Implement `src/decoding/lm_correction.py` with shallow fusion: `score = α * CTC_score + (1-α) * LM_score + β * len(hypothesis)`
- [ ] Integrate LM scoring into beam search via `alpha` and `beta` parameters
- [ ] Tune `alpha` and `beta` on the validation set (grid search over reasonable ranges)
- [ ] Report WER improvement from LM correction vs. raw beam search

### 2.7 Train & Compare All Four Models

- [ ] Train Transformer (Model C) on Willett data with same train/val/test split as Phase 1
- [ ] Train Hybrid CNN-Transformer (Model D) on the same data
- [ ] Evaluate all four models (GRU, CNN+LSTM, Transformer, Hybrid) with greedy decoding, beam search, and beam search + LM
- [ ] Create a comparison table: Model × Decoding Method → CER, WER, exact match
- [ ] Include the Willett-style GRU decoder as the reference architecture for comparison against known BCI results
- [ ] Save all checkpoints to `outputs/checkpoints/`

---

## Phase 3: Evaluation, Ablations & Multi-Dataset

**Goal:** Rigorous evaluation with ablation studies, per-character error analysis, and extension to additional datasets (UCSF ECoG). Produce publication-quality results tables and figures.

---

### 3.1 Full Evaluation Suite

- [ ] Implement `src/evaluation/ablations.py` with a runner that systematically evaluates model × decoding × augmentation combinations
- [ ] Generate per-character confusion matrix: which characters are most confused with which
- [ ] Compute per-subject metrics for multi-subject datasets
- [ ] Implement statistical significance testing: paired bootstrap resampling (n=1000) for model comparisons
- [ ] Export all results to JSON + CSV in `outputs/results/`

### 3.2 Ablation Studies

- [ ] **Decoding ablation:** Greedy vs. beam search (beam_width=100) — quantify WER improvement
- [ ] **LM ablation:** No LM vs. KenLM vs. GPT-2 re-ranking — measure CER/WER delta
- [ ] **Architecture ablation:** GRU Decoder (Willett-style) vs. CNN+LSTM vs. Transformer vs. Hybrid — same data, same splits, same training budget
- [ ] **Augmentation ablation:** Full augmentation vs. no augmentation — measure overfitting gap
- [ ] **Feature ablation:** Test swapping pathways across models — e.g., feed Pathway A (temporal conv) output into the Transformer instead of Pathway B, or feed Pathway B (linear projection) into the CNN+LSTM instead of its built-in Conv1D. Pathway C (firing rate binning) is always applied as a data-level step for Willett data and is not swappable. Model A's built-in linear projection and Model D's built-in CNN front-end are tested as-is (not swappable).

### 3.3 Visualization of Results

- [ ] Create `src/visualization/ctc_plots.py`: CTC probability heatmap `[time × characters]` for individual trials
- [ ] Create per-character error bar charts (which characters have highest/lowest CER)
- [ ] Create training curve comparison plots: all four models on one figure
- [ ] Create `src/visualization/embedding_plots.py`: t-SNE/UMAP of learned feature representations colored by character class
- [ ] Export all figures to `outputs/figures/` as high-resolution PNGs

### 3.3b Neural Latent Representation Analysis

> Extends the embedding visualization into a full neural representation analysis suite. This connects the project to modern neural manifold research performed at labs like Stanford NPTL and UCSF.

- [ ] Create `src/analysis/embeddings.py`: extract hidden-layer embeddings from any trained model for all trials, save to `outputs/embeddings/`
- [ ] Create `src/analysis/trajectory_plots.py`: visualize neural state trajectories through latent space during word production (project hidden states at each timestep to 2D via PCA/UMAP, plot as animated or color-coded path)
- [ ] Create `src/analysis/similarity_matrix.py`: compute pairwise cosine similarity between trial embeddings, generate character-grouped similarity heatmap (reveals whether similar letters produce similar neural representations)
- [ ] Create `src/analysis/saliency.py`: gradient-based electrode importance maps (integrated gradients or input×gradient) showing which electrodes contribute most to decoding each character
- [ ] Create `notebooks/05b_neural_representations.ipynb`: character clustering plots, neural trajectories for sample words, electrode importance heatmaps, trial similarity matrix

### 3.4 UCSF ECoG Dataset Integration

- [ ] Extend `src/data/loader.py` with `download_ucsf_ecog()` or manual-download instructions (researcher release)
- [ ] Implement ECoG-specific preprocessing: high-gamma bandpass (70–150 Hz), Hilbert transform for analytic amplitude
- [ ] Adapt the trial segmentation for speech data (different trial structure, phoneme/word labels)
- [ ] Run the best-performing model on ECoG data and report cross-dataset metrics
- [ ] Update comparison table with ECoG results alongside Willett results

### 3.5 OpenNeuro Dataset Integration

- [ ] Implement OpenNeuro download via `openneuro-py` in `src/data/loader.py` with configurable dataset ID
- [ ] Parse BIDS-format metadata (JSON sidecars, TSV event files) into the standardized internal format
- [ ] Validate on at least one OpenNeuro ECoG dataset (e.g., ds003688)
- [ ] Report metrics and add to the cross-dataset comparison table

### 3.6 Evaluation Notebook

- [ ] Create `notebooks/05_evaluation.ipynb` with the full ablation table rendered as a formatted DataFrame
- [ ] Include confusion matrices, error analysis, and CER/WER bar charts for all four models
- [ ] Show side-by-side decoded samples: raw CTC → beam search → beam search + LM
- [ ] Include t-SNE/UMAP embedding plots showing learned neural representations
- [ ] Include electrode importance heatmaps and trial similarity matrices from the latent representation analysis
- [ ] Add brief written interpretation of results (which model/method works best and why, how the GRU decoder compares to the original Willett results)

---

## Phase 4: Interactive Demo

**Goal:** A fully functional web-based demo with FastAPI backend and Streamlit frontend. Users can upload neural recordings, run inference, and explore decoding visualizations interactively.

---

### 4.1 FastAPI Backend

- [ ] Implement `app/api.py` with FastAPI app and the following endpoints:
  - `POST /decode`: accepts `.npy` file upload → preprocess → infer → return JSON response
  - `GET /decode/demo`: decode a pre-loaded sample trial (no upload needed)
  - `GET /health`: return service status
  - `GET /model/info`: return current model architecture name, parameter count, and test set metrics
- [ ] Response schema for `/decode`: `{predicted_text, raw_ctc_output, beam_hypotheses, char_probabilities, inference_time_ms}`
- [ ] Load the best model checkpoint on server startup; support selecting model via query parameter
- [ ] Add input validation: reject files that are not `.npy`, check shape compatibility
- [ ] Write `tests/test_api.py`: test each endpoint with valid and invalid inputs

### 4.2 Streamlit Frontend — Upload & Decode View

- [ ] Implement `app/dashboard.py` with Streamlit app
- [ ] File uploader widget for `.npy` neural recordings
- [ ] "Decode" button that calls the FastAPI `/decode` endpoint and displays predicted text
- [ ] Show inference time and confidence score
- [ ] Add a "Try Demo" button that calls `/decode/demo` for zero-setup exploration

### 4.3 Streamlit Frontend — Signal Viewer & Diagnostics

- [ ] Interactive time-series plot of uploaded neural channels using Plotly
- [ ] Channel selector sidebar: choose specific electrodes to display or show all
- [ ] Color-code channels by activity level (variance or mean amplitude)
- [ ] Zoomable time axis for inspecting specific trial segments
- [ ] Channel quality indicators: mark bad channels (from diagnostics module) with visual warnings
- [ ] SNR per-channel overlay: display signal-to-noise ratio alongside channel traces
- [ ] Power spectrum viewer: show PSD for selected channels inline

### 4.4 Streamlit Frontend — Decoding Visualization

- [ ] CTC probability heatmap: `[time × characters]` rendered as an interactive Plotly heatmap
- [ ] Animated character prediction timeline: characters appear sequentially as the model processes the signal
- [ ] Beam search hypotheses display: ranked list with log-probability scores
- [ ] Side-by-side comparison: greedy vs. beam search vs. beam search + LM outputs

### 4.5 Streamlit Frontend — Benchmarks & Embeddings

- [ ] Static table comparing all four model architectures on the test set (CER, WER, exact match, latency)
- [ ] CER/WER bar charts rendered with matplotlib in Streamlit
- [ ] Model selector dropdown to switch between model architectures

### 4.6 Streamlit Frontend — Neural Representation Explorer

> An interactive page for exploring neural latent representations — transforms the demo from a decoder into a neural analysis workstation.

- [ ] 2D embedding scatter plot (t-SNE/UMAP) with points colored by character class, interactive tooltips showing trial details (Plotly)
- [ ] Neural trajectory viewer: select a trial and watch the neural state evolve through latent space during character production
- [ ] Electrode importance heatmap: gradient-based attribution map showing which electrodes drive predictions for a selected character
- [ ] Trial similarity matrix: interactive heatmap showing cosine similarity between trial embeddings, grouped by character

---

## Phase 5: Polish, Packaging & Release

**Goal:** Production-ready code with Docker support, cleaned notebooks, comprehensive documentation, and a GitHub release. The project is portfolio-ready and reproducible by anyone.

---

### 5.1 Docker Containerization

- [ ] Create `Dockerfile`: multi-stage build with conda environment, copy source code, install dependencies
- [ ] Include pre-built `ctcdecode` wheel to avoid C++ build issues for users
- [ ] Create `docker-compose.yml` running both FastAPI backend (port 8000) and Streamlit frontend (port 8501)
- [ ] Verify the full demo works inside the container: `docker-compose up` → navigate to Streamlit → decode a sample
- [ ] Document Docker usage in README

### 5.2 Notebook Cleanup

- [ ] Clean `notebooks/01_data_exploration.ipynb`: clear stale outputs, add markdown explanations, ensure reproducible execution order
- [ ] Clean `notebooks/02_signal_diagnostics.ipynb`: channel quality reports, SNR distributions, spectral analysis, trial QC results
- [ ] Clean `notebooks/03_preprocessing_qc.ipynb`: include before/after comparisons, power spectra, artifact rejection stats
- [ ] Clean `notebooks/04_model_training.ipynb`: training curves, sample predictions, hyperparameter summary
- [ ] Clean `notebooks/05_evaluation.ipynb`: ablation tables, confusion matrices, final results
- [ ] Clean `notebooks/05b_neural_representations.ipynb`: character clusters, neural trajectories, electrode importance, similarity matrices
- [ ] Create `notebooks/06_demo_visualization.ipynb`: walkthrough of the full decoding pipeline on a single trial with inline visualizations

### 5.3 Scripts & CLI Polish

- [ ] Polish `scripts/download_data.sh`: handle all three datasets, add progress bars, verify checksums
- [ ] Polish `scripts/train.py`: add argparse with all configurable parameters, help text, and sensible defaults
- [ ] Polish `scripts/evaluate.py`: add argparse, support evaluating any saved checkpoint, export results to file
- [ ] Add a `scripts/run_ablations.py` that runs the full ablation suite and generates the comparison table
- [ ] Ensure all scripts can be run from the project root without path issues

### 5.4 Testing & CI

- [ ] Achieve test coverage for all critical modules: data loading, preprocessing, models, decoding, metrics
- [ ] Add integration test: synthetic data → full pipeline → decoded string (verifies end-to-end wiring)
- [ ] Create a `Makefile` or `justfile` with common commands: `make test`, `make train`, `make demo`, `make docker`
- [ ] Add a GitHub Actions workflow for running tests on push (`.github/workflows/test.yml`)

### 5.5 Documentation & README

- [ ] Write comprehensive `README.md`: project overview, architecture diagram, quick start, results summary, screenshots of the demo
- [ ] Document all configuration parameters in the README or a dedicated `docs/configuration.md`
- [ ] Add inline docstrings to all public functions in `src/`
- [ ] Create a `CONTRIBUTING.md` with setup instructions for new developers
- [ ] Tag a GitHub release (v1.0) with the best model checkpoint attached as a release asset

---

## Phase 6: Advanced Features & Future Extensions (Post-MVP)

**Goal:** Optional enhancements that push the project beyond MVP toward research-grade quality. These are stretch goals — pursue based on time and interest.

---

### 6.1 GPT-2 Re-ranking for LM Correction

- [ ] Implement Option B from Section 5.7: pass top-k beam hypotheses through GPT-2 to score each
- [ ] Compute combined score: `λ * CTC_score + (1-λ) * LM_score`
- [ ] Tune λ on validation set
- [ ] Compare against KenLM: measure CER/WER delta
- [ ] Report results in ablation table

### 6.2 Multi-Subject Training

- [ ] Pool data across multiple subjects from the Willett dataset
- [ ] Add subject-specific normalization layers (subject ID → learned bias/scale per channel)
- [ ] Train a single model on pooled data and evaluate within-subject vs. cross-subject performance
- [ ] Report whether multi-subject training improves or degrades per-subject CER

### 6.3 Experiment Tracking with Weights & Biases

- [ ] Integrate W&B logging into `Trainer`: log loss, CER, WER, learning rate per epoch
- [ ] Log sample decoded outputs as W&B Tables
- [ ] Log training curves, confusion matrices, and embedding plots as W&B artifacts
- [ ] Add W&B sweep config for hyperparameter tuning (learning rate, model size, augmentation strength)

### 6.4 Cross-Session Generalization

- [ ] Train on session 1, evaluate on session 2 (for datasets with multiple sessions per subject)
- [ ] Implement per-session z-score normalization as a mitigation for neural drift
- [ ] Report cross-session CER vs. within-session CER
- [ ] Document findings and limitations

### 6.5 Neural Latent Dynamics & Manifold Analysis

> Advanced neural representation analysis connecting the project to cutting-edge neural manifold research. Goes beyond basic t-SNE clustering to study how neural population dynamics evolve during handwriting.

- [ ] Implement `src/analysis/manifold.py`: fit UMAP and PCA models to population-level neural embeddings across all trials
- [ ] Implement per-character neural manifold analysis: compute cluster centroids, inter-class distances, and silhouette scores to quantify how well the model separates characters in latent space
- [ ] Implement neural dynamics analysis: compute velocity fields in latent space to characterize how neural states transition between characters
- [ ] Create animated neural trajectory visualizations: 3D latent space with time-colored paths showing state evolution during full sentence production
- [ ] Integrate manifold analysis into the Streamlit Neural Representation Explorer as an advanced tab
- [ ] Document findings on neural manifold structure in the evaluation notebook

### 6.6 Real-Time Inference Prototype

- [ ] Implement a streaming inference pipeline: buffer incoming data → sliding window → model inference → incremental character display
- [ ] Target < 300 ms latency per character update
- [ ] Create a simple terminal-based demo: simulate streaming by feeding trial data chunk-by-chunk
- [ ] Document the architecture for future LSL (Lab Streaming Layer) integration

---

## Summary Table

| Phase | Name | Key Deliverable | Depends On |
|-------|------|----------------|------------|
| **0** | Project Setup | Repo structure, environment, data downloaded, raw signal visualization | — |
| **0.6** | Signal Diagnostics | Neural recording quality reports: channel QC, SNR, spectral analysis, trial rejection | Phase 0 |
| **1** | MVP | End-to-end pipeline: preprocess → GRU Decoder + CNN+LSTM → CTC → greedy decode → CER reported | Phase 0.6 |
| **2** | Enhanced Models | Transformer + Hybrid models, beam search, KenLM, 4-model comparison table | Phase 1 |
| **3** | Evaluation & Ablations | Full ablation suite, neural representation analysis, confusion matrices, multi-dataset results | Phase 2 |
| **4** | Interactive Demo | FastAPI + Streamlit app with signal viewer, diagnostics, CTC heatmaps, neural representation explorer | Phase 2 |
| **5** | Polish & Release | Docker, cleaned notebooks, documentation, CI, GitHub release v1.0 | Phase 3 + 4 |
| **6** | Advanced Features | GPT-2 LM, multi-subject, W&B tracking, cross-session, neural manifold analysis, real-time prototype | Phase 5 |

---

> **Note:** Phases 3 and 4 can be developed in parallel after Phase 2 is complete. Phase 6 items are independent stretch goals.
