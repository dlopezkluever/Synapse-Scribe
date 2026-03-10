# Brain → Text Decoder Simulator — Product Requirements Document
**Neural Speech & Handwriting BCI Decoding System**
Version 1.0 · March 2026

> Convert neural activity recordings into predicted characters and words — end-to-end, from raw signal to readable text.

---

## 1. Project Overview

### 1.1 Purpose
This document defines the full product requirements for a Brain → Text Decoder Simulator: a system that takes neural recordings (ECoG or LFP) and decodes them into character sequences using modern sequence modeling. The project replicates the core architecture used in state-of-the-art speech BCI labs — most notably the UCSF Chang Lab and the work behind neural handwriting decoding at Stanford — without requiring implanted hardware. All development uses publicly available neural datasets paired with text labels.

### 1.2 The Problem
People with paralysis, ALS, or locked-in syndrome lose the ability to communicate through speech or movement. Implanted BCIs can read neural signals and reconstruct intended speech or text — but the decoding systems powering these devices are complex, rarely open-sourced, and almost never accessible for independent developers to study or build on. This project closes that gap by providing a complete, reproducible decoder built on real neural data.

### 1.3 Core Architecture

```
neural signals
      ↓
  preprocessing
  (filter, normalize, segment)
      ↓
  feature extraction
  (temporal convolution / linear projection)
      ↓
  sequence model
  (LSTM / Transformer)
      ↓
  character probabilities
  (CTC output head)
      ↓
  beam search decoder
      ↓
  [optional] language model correction
      ↓
  decoded text
```

### 1.4 Why This Project Matters
- Mirrors real neurotechnology pipelines used in clinical BCI research
- Demonstrates neural signal processing, time-series ML, and sequence modeling in a single system
- Produces an interactive demo that makes the decoding process visible and interpretable
- Positions as a strong portfolio project for neurotech, robotics, and AI research roles
- Lays the foundation for real-time decoding once live hardware is accessible

---

## 2. Goals & Success Criteria

### 2.1 Primary Goal
Build an offline, end-to-end neural text decoder that accepts real ECoG or LFP recordings as input and outputs predicted character sequences, achieving character error rates (CER) that demonstrate meaningful decoding above chance across held-out trials.

### 2.2 Success Metrics

| Metric | Minimum | Target | Stretch |
|---|---|---|---|
| Character Error Rate (CER) | < 40% | < 20% | < 10% |
| Word Error Rate (WER) | < 60% | < 35% | < 15% |
| Trials decoded correctly (exact match) | 5% | 20% | 40% |
| Inference latency per trial | < 5s | < 1s | < 200ms |
| Subjects evaluated | 1 | 3 | All available |
| Model architectures benchmarked | 1 | 2 | 3 |

> **Baseline context:** Published results on UCSF ECoG speech datasets achieve ~3% WER with language model correction. A well-implemented CNN+LSTM system without LM correction typically achieves 20–40% WER on the same data. Matching or approaching this is the target.

### 2.3 Non-Goals
- Real-time decoding from live implanted hardware is out of scope for v1
- The system is not a medical device and makes no clinical claims
- Training from scratch on proprietary hospital data is not planned
- Multi-speaker generalization is not required for v1 (within-subject evaluation acceptable)
- Mobile or embedded deployment is not in scope

---

## 3. Dataset Specification

### 3.1 Primary Dataset — UCSF ECoG Speech Data

| Field | Value |
|---|---|
| Source | UCSF Chang Lab research releases |
| Modality | Electrocorticography (ECoG) — high-density cortical surface electrodes |
| Subjects | Multiple participants with speech or motor tasks |
| Channels | 64–256 ECoG channels |
| Sampling Rate | 400 Hz – 1000 Hz (dataset-dependent) |
| Labels | Phoneme or word-level transcripts aligned to neural recordings |
| Format | .mat, .npy, or .hdf5 depending on release |
| Access | Direct researcher release pages; some available on Zenodo |
| Key Papers | Chang et al. (Nature, 2021); Moses et al. (NEJM, 2021) |

### 3.2 Secondary Dataset — OpenNeuro

| Field | Value |
|---|---|
| Source | openneuro.org |
| Modality | ECoG, sEEG, or EEG depending on study |
| Format | BIDS-compatible (standardized folder structure + JSON metadata) |
| Key Datasets | ds003688 (ECoG speech), ds002718 (intracranial recordings) |
| Access | Fully open, no registration required for most datasets |
| Tools | OpenNeuro Python client: `pip install openneuro-py` |

### 3.3 Tertiary Dataset — Neural Handwriting (Willett et al., 2021)

| Field | Value |
|---|---|
| Source | Stanford Neural Prosthetics Translational Laboratory |
| Modality | Utah array spiking activity (192 electrodes, motor cortex) |
| Task | Imagined handwriting of individual characters |
| Sampling Rate | 30 kHz (spike-sorted to ~250 Hz binned firing rates) |
| Labels | Character-level ground truth (a–z + special characters) |
| Format | .mat files, released alongside paper |
| Why Use This | Clean, well-labeled, character-level task — ideal for a first decoder |
| Access | https://doi.org/10.5061/dryad.wh70rxwmv |

> **Recommendation:** Start with the Willett handwriting dataset. It is the cleanest, most directly character-aligned neural dataset available and the closest analog to a typed text decoder. Move to UCSF ECoG for speech-based decoding once the pipeline is validated.

### 3.4 Dataset Structure Convention

All datasets should be normalized to the following internal structure after download:

```
data/
└── {dataset_name}/
    └── sub-{subject_id}/
        └── ses-{session}/
            ├── neural/
            │   └── trial_{n}_signals.npy     # shape: [time_steps, channels]
            └── labels/
                └── trial_{n}_transcript.txt  # plain text ground truth
```

---

## 4. System Architecture

### 4.1 Full Pipeline Overview

The system is composed of five independent, testable modules:

```
[1. Data Loader] → [2. Preprocessor] → [3. Feature Extractor] → [4. Sequence Model] → [5. Decoder + LM]
                                                                          ↓
                                                                   [6. Demo Interface]
```

### 4.2 Module Responsibilities

| Module | Responsibility | Primary Library |
|---|---|---|
| Data Loader | Download, parse, and normalize raw neural + label data | `openneuro-py`, `scipy.io`, `numpy` |
| Preprocessor | Bandpass filter, z-score normalize, segment into trials | `MNE-Python`, `scipy.signal` |
| Feature Extractor | Extract temporal features or project to embedding space | `numpy`, `torch` |
| Sequence Model | Map feature sequences to character probability sequences | `PyTorch` |
| CTC Decoder | Collapse probability sequences into character strings | `torch.nn.CTCLoss`, `ctcdecode` |
| LM Correction | Re-rank decoded hypotheses using a character-level LM | `kenlm` or `transformers` |
| Demo Interface | Accept neural file upload, run inference, display output | `FastAPI`, `Streamlit` |

---

## 5. Detailed Functional Requirements

### 5.1 Module 1 — Data Loading

**Requirements:**
- Support download of the Willett handwriting dataset via direct URL with checksum verification
- Support OpenNeuro dataset download via `openneuro-py` with configurable dataset ID
- Parse `.mat` files using `scipy.io.loadmat`; parse `.npy` files using `numpy.load`
- Convert all datasets to the standardized internal format (see Section 3.4)
- Build a trial index: a DataFrame with columns `[subject, session, trial_id, signal_path, label_path, n_timesteps, n_channels]`
- Support configurable train/val/test split at the trial level (default: 80/10/10), stratified by subject

**Configuration Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `DATASET` | `willett_handwriting` | Dataset to load (`willett_handwriting`, `openneuro`, `ucsf_ecog`) |
| `SUBJECTS` | `[1]` | Subject IDs to include |
| `DATA_PATH` | `./data/` | Local cache root |
| `SPLIT` | `[0.8, 0.1, 0.1]` | Train/val/test proportions |

---

### 5.2 Module 2 — Signal Preprocessing

Neural signals require careful preparation before feature extraction. Steps must be applied in the following order.

**Step 1: Bandpass Filtering**
- For handwriting / motor imagery data: bandpass 1–200 Hz (captures spiking and LFP components)
- For ECoG speech data: apply high-gamma bandpass at 70–150 Hz (the dominant speech-related band)
- Implementation: `scipy.signal.butter` (4th order Butterworth), zero-phase via `filtfilt`
- Reject trials with > 3× median channel variance (gross artifact flag)

**Step 2: Notch Filtering**
- Apply 60 Hz notch (and 120 Hz harmonic) using `scipy.signal.iirnotch`

**Step 3: Channel Normalization**
- Z-score each channel independently across the full session: `(x - μ) / σ`
- Compute μ and σ on training set only; apply to val/test to prevent data leakage
- Clip normalized values to [-5, 5] to reduce outlier influence

**Step 4: Bad Channel Detection & Removal**
- Flag channels with zero variance or variance > 10× session median as bad
- Remove bad channels; record removed channel indices in trial metadata

**Step 5: Temporal Downsampling**
- Downsample to a target rate (default: 250 Hz) using `scipy.signal.decimate`
- Downsampling reduces computational cost while preserving the relevant frequency content

**Step 6: Trial Segmentation**
- Segment continuous recordings into per-trial windows using provided onset/offset annotations
- Add 100 ms pre-onset padding and 200 ms post-offset padding
- Pad or truncate all trials to a fixed maximum length `T_max` (configurable; default: 2000 timesteps at 250 Hz = 8 seconds)

**Preprocessing Configuration:**

| Parameter | Handwriting Default | ECoG Speech Default |
|---|---|---|
| `BANDPASS_LOW` | 1 Hz | 70 Hz |
| `BANDPASS_HIGH` | 200 Hz | 150 Hz |
| `TARGET_FS` | 250 Hz | 200 Hz |
| `T_MAX` | 2000 timesteps | 4000 timesteps |
| `ZSCORE_CLIP` | 5.0 | 5.0 |

---

### 5.3 Module 3 — Feature Extraction

Feature extraction converts raw preprocessed signals into representations suitable for sequence modeling. Two pathways are required.

#### Pathway A: Temporal Convolution Features (for CNN-based models)
- Apply a 1D temporal convolution bank over the time axis independently per channel
- Kernel sizes: [3, 7, 15] samples — captures multi-scale temporal patterns
- Output: concatenate activations from all kernel sizes → `[time_steps, channels × n_kernels]`
- Apply batch normalization and ReLU after convolution
- Optionally apply max pooling (stride 2) to reduce sequence length

#### Pathway B: Linear Projection (for Transformer-based models)
- Treat each timestep's channel vector as a token: input shape `[T, C]`
- Project to model dimension d_model via a learned linear layer: `[T, d_model]`
- Add sinusoidal positional encodings
- This is the direct analog of patch embedding in Vision Transformers, applied to time-series

#### Pathway C: Firing Rate Binning (for Willett spiking data specifically)
- Bin spike counts into 10 ms non-overlapping windows
- Result: `[n_bins, n_electrodes]` matrix of binned firing rates
- Apply square-root transform to stabilize variance: `sqrt(firing_rate)`
- This is the standard preprocessing step used in the original Willett et al. paper

---

### 5.4 Module 4 — Sequence Models

Three model architectures must be implemented as swappable backends with a common interface: `forward(x: Tensor[B, T, C]) → logits: Tensor[B, T, n_classes]`.

#### Model A: CNN + LSTM (Baseline)

The classical BCI decoding architecture. Computationally cheap, interpretable, and well-validated on small neural datasets.

```
Input: [B, T, C]
  ↓
Conv1D blocks (temporal feature extraction)
  [B, T', F]
  ↓
Bidirectional LSTM (sequence modeling)
  [B, T', 2*H]
  ↓
Linear projection → character logits
  [B, T', n_classes]
```

| Hyperparameter | Default |
|---|---|
| Conv channels | 256 |
| Conv kernel size | 7 |
| Conv layers | 3 |
| LSTM hidden size (H) | 512 |
| LSTM layers | 2 |
| Dropout | 0.5 |
| n_classes | 29 (a–z + space + blank) |

#### Model B: Transformer Encoder (Primary)

More capable for long sequences. Captures global context that LSTM misses. Matches modern BCI research architectures.

```
Input: [B, T, C]
  ↓
Linear projection to d_model + positional encoding
  [B, T, d_model]
  ↓
N × Transformer Encoder Layer
  (multi-head self-attention + feedforward + LayerNorm)
  [B, T, d_model]
  ↓
Linear projection → character logits
  [B, T, n_classes]
```

| Hyperparameter | Default |
|---|---|
| d_model | 512 |
| n_heads | 8 |
| n_layers | 6 |
| FFN dim | 2048 |
| Dropout | 0.1 |
| Max sequence length | 4096 |
| n_classes | 29 |

#### Model C: Hybrid CNN-Transformer

Combines local feature extraction (CNN) with global sequence modeling (Transformer). Best of both architectures.

```
Input: [B, T, C]
  ↓
CNN front-end (3 layers, stride-2 pooling)
  [B, T/4, F]
  ↓
Transformer Encoder (4 layers)
  [B, T/4, d_model]
  ↓
Linear → logits
  [B, T/4, n_classes]
```

The CNN front-end acts as a learned feature extractor and reduces sequence length by 4× before the Transformer, significantly improving computational efficiency.

---

### 5.5 Module 5 — CTC Loss & Training

#### Why CTC
Neural signals and character labels are not frame-aligned. A 4-second recording of someone imagining the word "hello" does not have a precise timestamp for each letter. Connectionist Temporal Classification (CTC) solves this by marginalizing over all valid alignments during training, requiring only sequence-level labels.

#### CTC Setup
- Add a `blank` token to the vocabulary as class 0
- Full vocabulary: `{blank=0, a=1, b=2, ..., z=26, space=27}` → 28 classes
- Loss: `torch.nn.CTCLoss(reduction='mean', zero_infinity=True)`
- Input to CTCLoss: log-softmax of model logits `[T, B, n_classes]` (time-first)
- Target: concatenated character index sequences with `target_lengths` tensor

#### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| LR scheduler | Cosine annealing with warmup (500 steps) |
| Batch size | 16 trials |
| Max epochs | 200 |
| Early stopping | Patience = 20 epochs on val CER |
| Gradient clipping | Max norm = 1.0 |
| Weight decay | 1e-4 |
| Mixed precision | fp16 (if GPU available) |

#### Data Augmentation
- **Time masking:** randomly zero out 1–3 time windows of 10–50 ms during training (SpecAugment-style)
- **Channel dropout:** randomly zero out 5–10% of channels per batch
- **Gaussian noise:** add low-level Gaussian noise (σ = 0.01 post-normalization)
- All augmentations applied only during training

---

### 5.6 Module 6 — CTC Decoding

Raw model output is a `[T, n_classes]` probability matrix. Two decoding strategies are required.

#### Greedy Decoding (fast, for development)
```
for each timestep t:
    take argmax over character classes
collapse repeated characters
remove blank tokens
→ predicted string
```

Example:
```
h h blank e e l l l blank o → hello
```

#### Beam Search Decoding (better, for evaluation)
- Use `ctcdecode` library (`pip install ctcdecode`)
- Beam width: 100 (configurable)
- Returns top-k hypotheses with log-probabilities
- LM integration: pass external KenLM binary for shallow fusion (see Module 7)

---

### 5.7 Module 7 — Language Model Correction (Optional but High-Impact)

A language model correction layer dramatically improves decoded text accuracy by re-ranking or correcting phonetically/spatially plausible but wrong outputs.

#### Option A: KenLM Shallow Fusion (lightweight, recommended for v1)
- Train a character-level n-gram language model (5-gram) on a large text corpus (e.g., Wikipedia or BooksCorpus)
- Integrate directly into beam search via `ctcdecode`'s `alpha` (LM weight) and `beta` (word insertion penalty) parameters
- Tune `alpha` and `beta` on the validation set

#### Option B: GPT-2 Re-ranking (higher quality, more compute)
- After beam search, pass top-k hypotheses through GPT-2 to score each candidate
- Select hypothesis with highest combined score: `λ * CTC_score + (1-λ) * LM_score`
- Tune λ on validation set

Example improvement:
```
raw beam output:  "helo wrld"
LM-corrected:     "hello world"
```

---

## 6. Evaluation

### 6.1 Metrics

| Metric | Formula | Tool |
|---|---|---|
| Character Error Rate (CER) | (substitutions + insertions + deletions) / reference_length | `jiwer` library |
| Word Error Rate (WER) | Same, at word level | `jiwer` library |
| Exact Match Accuracy | Fraction of trials with perfect decoding | custom |
| Bits Per Character (BPC) | Cross-entropy at character level | custom |

### 6.2 Evaluation Protocol
- Evaluate on held-out test set (never seen during training or hyperparameter tuning)
- Report metrics with and without LM correction
- Report per-character confusion matrix — reveals which characters are most often confused
- Report per-subject metrics for datasets with multiple subjects
- Statistical significance: paired bootstrap resampling (n=1000) for model comparisons

### 6.3 Ablation Studies
Run the following ablations and report results in a comparison table:

| Ablation | What It Tests |
|---|---|
| Greedy vs. beam search decoding | Value of beam search |
| No LM vs. KenLM vs. GPT-2 | Value of language model |
| CNN+LSTM vs. Transformer vs. Hybrid | Architecture comparison |
| With vs. without augmentation | Value of data augmentation |
| Pathway A vs. B vs. C features | Best feature extraction approach |
| Within-session vs. cross-session | Generalization of the decoder |

---

## 7. Interactive Demo

The demo is a first-class deliverable — it makes the decoding process tangible and is the primary way to communicate the project to an outside audience.

### 7.1 Backend (FastAPI)

**Endpoints:**

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/decode` | POST | `.npy` file (neural recording) | `{predicted_text, confidence, char_probs}` |
| `/decode/demo` | GET | None | Decode a pre-loaded sample trial |
| `/health` | GET | None | Service status |
| `/model/info` | GET | None | Current model architecture + metrics |

**`/decode` response schema:**
```json
{
  "predicted_text": "hello world",
  "raw_ctc_output": "hh blank ee ll blank oo",
  "beam_hypotheses": ["hello world", "hello werd", "helo world"],
  "char_probabilities": [[0.9, 0.01, ...], ...],
  "inference_time_ms": 142
}
```

### 7.2 Frontend (Streamlit)

**Views:**

**1. Upload & Decode**
- File uploader for `.npy` neural recording
- "Decode" button → calls backend → displays predicted text
- Shows inference time and confidence

**2. Signal Viewer**
- Interactive time-series plot of uploaded neural channels (plotly)
- Channel selector (show all or highlight specific electrodes)
- Color-coded by channel activity level

**3. Decoding Visualization**
- Heatmap: `[time_steps × characters]` showing CTC output probabilities over time
- Animated character prediction timeline: watch letters appear as the model processes the signal
- Beam search hypotheses ranked by score

**4. Model Benchmarks**
- Static table comparing all three architectures on the test set
- CER / WER bar charts (matplotlib rendered in Streamlit)

**5. t-SNE Embedding Viewer**
- Pre-computed t-SNE projection of neural features colored by character class
- Shows whether the model has learned separable representations

---

## 8. Repository Structure

```
brain-text-decoder/
│
├── data/                          # Raw + processed datasets (gitignored)
│   ├── willett_handwriting/
│   ├── openneuro/
│   └── ucsf_ecog/
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Global configuration dataclass
│   ├── data/
│   │   ├── loader.py              # Dataset download + parsing
│   │   ├── dataset.py             # PyTorch Dataset + DataLoader
│   │   └── transforms.py         # Augmentation transforms
│   ├── preprocessing/
│   │   ├── filter.py              # Bandpass, notch, downsample
│   │   ├── normalize.py           # Z-score, channel rejection
│   │   └── segment.py             # Trial segmentation
│   ├── features/
│   │   ├── temporal_conv.py       # Pathway A
│   │   ├── projection.py          # Pathway B
│   │   └── firing_rate.py         # Pathway C (Willett)
│   ├── models/
│   │   ├── base.py                # Common interface
│   │   ├── cnn_lstm.py            # Model A
│   │   ├── transformer.py         # Model B
│   │   └── cnn_transformer.py     # Model C
│   ├── training/
│   │   ├── trainer.py             # Training loop, checkpointing
│   │   ├── ctc_loss.py            # CTC loss wrapper
│   │   └── scheduler.py           # LR warmup + cosine decay
│   ├── decoding/
│   │   ├── greedy.py              # Greedy CTC decoder
│   │   ├── beam_search.py         # Beam search with LM
│   │   └── lm_correction.py       # KenLM / GPT-2 re-ranking
│   ├── evaluation/
│   │   ├── metrics.py             # CER, WER, exact match
│   │   └── ablations.py           # Ablation study runner
│   └── visualization/
│       ├── signal_plots.py        # Channel heatmaps, time series
│       ├── ctc_plots.py           # Probability timeline, confusion matrix
│       └── embedding_plots.py     # t-SNE projections
│
├── app/
│   ├── api.py                     # FastAPI backend
│   └── dashboard.py               # Streamlit frontend
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_qc.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_demo_visualization.ipynb
│
├── outputs/
│   ├── checkpoints/               # Saved model weights
│   ├── results/                   # JSON + CSV metrics
│   └── figures/                   # Exported plots
│
├── scripts/
│   ├── download_data.sh
│   ├── train.py                   # CLI training entrypoint
│   └── evaluate.py                # CLI evaluation entrypoint
│
├── tests/
├── requirements.txt
├── environment.yml
├── Dockerfile
└── README.md
```

---

## 9. Technical Stack

| Component | Library / Tool | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Primary language |
| Deep Learning | PyTorch | >= 2.1 | All model training and inference |
| Neural Signal I/O | MNE-Python | >= 1.6 | EEG/ECoG loading, filtering, epoching |
| Signal Processing | SciPy | >= 1.11 | Butterworth filters, decimation |
| Numerical | NumPy | >= 1.24 | Array operations |
| Dataset Access | openneuro-py | latest | OpenNeuro dataset download |
| CTC Decoding | ctcdecode | >= 0.4 | Beam search with LM integration |
| Language Model | kenlm | latest | N-gram LM for beam search |
| Metrics | jiwer | >= 3.0 | CER / WER computation |
| Visualization | Matplotlib, Seaborn, Plotly | latest | Signal + metric plots |
| Data Tables | Pandas | >= 2.0 | Trial index, results tables |
| Backend API | FastAPI + Uvicorn | latest | Inference API |
| Frontend Demo | Streamlit | >= 1.30 | Interactive web interface |
| Experiment Tracking | Weights & Biases | latest | Run tracking (optional) |
| Containerization | Docker | latest | Reproducible deployment |
| Environment | conda | latest | Dependency management |

---

## 10. Implementation Phases & Timeline

| # | Phase | Key Deliverables | Duration | Depends On |
|---|---|---|---|---|
| 1 | Environment & Data | conda env, Willett dataset downloaded and parsed, trial index DataFrame, basic signal plots | 2–3 days | — |
| 2 | Preprocessing Pipeline | Filter, normalize, segment pipeline; QC plots; artifact rejection; verify signal quality | 3–4 days | Phase 1 |
| 3 | Feature Extraction | All 3 pathways implemented; shapes verified; PCA/t-SNE shows class structure in features | 2–3 days | Phase 2 |
| 4 | CTC Training (Baseline) | CNN+LSTM trained and converging; greedy CER reported on val set; training curves plotted | 3–5 days | Phase 3 |
| 5 | Transformer Model | Transformer encoder trained; compared to CNN+LSTM; beam search decoder integrated | 3–5 days | Phase 4 |
| 6 | LM Correction | KenLM integrated into beam search; WER improvement measured and reported | 2–3 days | Phase 5 |
| 7 | Evaluation & Ablations | Full ablation table; confusion matrix; per-character error analysis; final test set results | 2–3 days | Phase 6 |
| 8 | Demo Interface | FastAPI backend + Streamlit frontend; signal viewer; CTC heatmap; deployed locally | 3–5 days | Phase 5 |
| 9 | Polish & Documentation | README, Dockerfile, notebooks cleaned, results exported, GitHub release | 2–3 days | Phase 8 |

**Total estimated effort: 22–34 focused development days.**

---

## 11. Key Algorithms — Deep Dives

### 11.1 Connectionist Temporal Classification (CTC)

CTC enables training a sequence model when input and output sequences have different (and unknown) alignments. For a neural recording of "hello," the model doesn't know which timestep corresponds to 'h' vs 'e'. CTC marginalizes over all valid alignments using dynamic programming.

During training, CTC loss minimizes: `-log P(y | x)` where the probability is summed over all paths that collapse to the target sequence `y`. The blank token absorbs ambiguous timesteps and separates repeated characters.

During inference: beam search explores the highest-probability paths through the character probability matrix and selects the most likely sequence.

### 11.2 High-Gamma ECoG Features

For speech ECoG data, the high-gamma band (70–150 Hz) is the most speech-informative frequency range. This reflects local cortical population activity in language and motor areas. Power in this band tracks syllable production, articulatory movement, and phoneme identity. Filtering and computing the analytic amplitude (via Hilbert transform) of this band yields a smooth, reliable neural feature for speech decoding.

### 11.3 Transformer Self-Attention for Neural Sequences

Standard RNNs process neural sequences recurrently, making them slow and prone to vanishing gradients on long sequences (> 1000 timesteps). Transformer self-attention computes relationships between all pairs of timesteps in parallel. For motor imagery and speech, this allows the model to learn that a late timestamp (e.g., end of vowel) provides context for decoding an early timestamp (e.g., consonant onset) — a bidirectional relationship LSTMs handle poorly.

### 11.4 CTC + LM Shallow Fusion

CTC alone produces acoustically plausible but linguistically noisy outputs. A language model provides a prior over likely character sequences. In shallow fusion, the LM score is added to the CTC score at each beam search step:

```
score(hypothesis) = λ * log P_CTC(hypothesis | neural) 
                  + (1-λ) * log P_LM(hypothesis)
                  + β * len(hypothesis)
```

Where `β` is a word insertion bonus that encourages longer, more complete hypotheses. This is the same technique used in modern ASR systems (DeepSpeech, Whisper).

---

## 12. Risks & Mitigations

| Risk | Severity | Description | Mitigation |
|---|---|---|---|
| Dataset access changes | High | Researcher-released datasets can disappear or move | Download and cache locally on first access; document DOIs and archive links in README |
| CTC not converging | High | CTC training can silently fail — loss stays high and model outputs blanks | Monitor blank token probability during training; verify label lengths < input lengths; use `zero_infinity=True`; start with very short trials |
| Overfitting on small neural datasets | High | Most neural datasets have < 1000 trials per subject — insufficient for large models | Use Model A (CNN+LSTM) first; apply all augmentations; use within-subject evaluation; EarlyStopping |
| Cross-session generalization failure | Medium | Neural recordings drift between sessions — models trained on session 1 may fail on session 2 | Document this limitation clearly; implement z-score normalization per session; explore Riemannian alignment as future work |
| Data format inconsistency | Medium | UCSF and OpenNeuro datasets have very different formats and metadata conventions | Build a unified adapter layer per dataset; test each adapter with a format validation script |
| CTC decoder library compatibility | Medium | `ctcdecode` has build issues on some platforms (requires C++ compilation) | Provide fallback to greedy decoding; include pre-built wheels in Docker image |
| LM training corpus licensing | Low | Some large text corpora have restrictive licenses | Use Wikipedia dump (CC BY-SA) or OpenWebText; document corpus source |

---

## 13. Future Extensions (v2+)

### Short-Term
- **Real EEG integration:** Replace ECoG input with EEG from OpenBCI or g.tec — lower signal quality but accessible without surgery
- **Multi-subject training:** Pool data across subjects with subject-specific normalization layers; significantly increases training set size
- **Phoneme-level decoding:** Add a phoneme intermediate representation between neural features and characters for speech datasets
- **Character-level GPT-2 fine-tuning:** Fine-tune a small GPT-2 on the decoded text domain for stronger LM correction

### Medium-Term
- **Real-time inference pipeline:** Stream neural data via LSL → buffer → model inference → character display, targeting < 300 ms latency
- **Online adaptation:** Implement continual learning to fine-tune the model on new subject data with minimal labeled trials (few-shot adaptation)
- **Word-level decoding:** Move from character-level CTC to word-piece tokenization, enabling direct word prediction
- **Confidence calibration:** Add calibrated uncertainty estimates per character so downstream systems know when to ask for clarification

### Long-Term
- **Hardware integration:** Deploy to OpenBCI Cyton + custom amplifier for real scalp-EEG BCI typing experiments
- **Robotic / assistive device output:** Route decoded text to a text-to-speech engine or robotic control interface
- **Closed-loop feedback:** Display decoded text in real time to the subject, enabling neurofeedback-guided improvement of imagery quality
- **Cross-modal grounding:** Align neural embeddings to language model embeddings (like CLIP but for brain-text), enabling zero-shot decoding of unseen words

---

## 14. Project Name Options

| Name | Rationale |
|---|---|
| **NeuroType** | Direct — brain signals → typed text |
| **CortexText** | Emphasizes cortical (ECoG) origin |
| **NeuralScribe** | Evokes handwriting + neural roots |
| **SynapseScript** | Memorable, technically evocative |
| **CortexDecode** | Clearest for a GitHub repo name |

> Recommendation: **NeuroType** or **CortexDecode** — both communicate `brain signals → language` immediately, which is what matters for a GitHub landing page.

---

## 15. References

- Willett, F.R. et al. (2021). High-performance brain-to-text communication via handwriting. *Nature*, 593: 249–254.
- Moses, D.A. et al. (2021). Neuroprosthesis for Decoding Speech in a Paralyzed Person with Anarthria. *New England Journal of Medicine*, 385: 217–227.
- Chang, E.F. et al. (2020). The auditory representation of speech sounds in human motor cortex. *eLife*.
- Lawhern, V.J. et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*, 15(5).
- Graves, A. et al. (2006). Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. *ICML 2006*.
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Park, D.S. et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Interspeech 2019*.
- MNE-Python documentation: [mne.tools](https://mne.tools)
- OpenNeuro: [openneuro.org](https://openneuro.org)
- ctcdecode library: [github.com/parlance/ctcdecode](https://github.com/parlance/ctcdecode)