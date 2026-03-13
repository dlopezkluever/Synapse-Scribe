 Phase 7A — Training Stability (smallest scope, fastest to implement)
  - Fix scheduler ordering bug
  - Add ReduceLROnPlateau to prevent collapse
  - Boost augmentation defaults (noise 0.01→0.1, channel dropout 10%→20%)
  - Update Colab notebook hyperparameters

  Phase 7B — Data Pipeline (medium scope)
  - Wire existing firing rate features into training (code exists, just not connected)
  - Session-aware train/val/test split (prevent temporal leakage)
  - Sentence trial oversampling (fix the 74/26% imbalance)

  Phase 7C — Model Downsampling (largest scope, most impactful)
  - Add 4x temporal downsampling to GRU, CNN-LSTM, and Transformer via stride-2 conv front-ends
  - Expose downsample_factor property on all models (CNN-Transformer already has 8x)
  - Adjust CTC input_lengths in trainer to match downsampled output


# Training Improvements — Post-GPU Run Analysis

## GPU Training Results (2026-03-13)

| Model | Epochs | Best CER | Outcome |
|-------|--------|----------|---------|
| GRU | 22 (early stop) | 0.9824 | Failed — single-char predictions |
| CNN-LSTM | 85 (early stop) | 0.2560 | Learned then collapsed |
| Transformer | 9+ (in progress) | 1.0000 | Stuck on blank outputs |
| CNN-Transformer | not yet run | — | Has 8x downsampling (best candidate) |

---

## Issues Found

### 1. LR Scheduler Bug (affects ALL models)
Every model prints: `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`. The first LR value is skipped. In trainer.py:234, scheduler.step() is called before optimizer.step() completes. → **Fix in Phase 7A**

### 2. GRU: Total failure (0.9824 CER = random)
Never gets past predicting a single character. Train loss drops (159 → 1.92) but val CER flat at ~0.98. Memorizing without generalizing. → **Root cause: no downsampling (Phase 7C) + raw signals (Phase 7B)**

### 3. CNN-LSTM: Breakthrough then catastrophic collapse
- Epochs 1-57: CER stuck at ~0.97
- Epoch 58: Sudden breakthrough to CER 0.50
- Epoch 65: Best CER = 0.2560
- Epochs 75-85: Complete collapse back to CER 0.99+

Classic overfitting. Cosine LR doesn't reduce fast enough after finding a good minimum. → **Fix in Phase 7A (ReduceLROnPlateau)**

### 4. Transformer: Stuck on blanks (19M params)
9 epochs in, outputting empty strings. CTC collapsing everything to blank. 19M params (larger than expected). → **Root cause: no downsampling (Phase 7C)**

### 5. Val loss very noisy for CNN-LSTM
Swings between 1.1 and 10.9 throughout training. High batch-level variance. → **Fix in Phase 7B (data balancing)**

---

## Root Causes (Priority Order)

| # | Issue | Severity | Phase |
|---|-------|----------|-------|
| 1 | No temporal downsampling in 3/4 models — CTC learns 99%+ blank ratio | CRITICAL | 7C |
| 2 | Training on raw signals, not firing rates (Pathway C exists but unused) | CRITICAL | 7B |
| 3 | 74% single-letter / 26% sentence trial imbalance, no stratification | HIGH | 7B |
| 4 | Weak augmentation (noise std=0.01 = 1% of signal variance) | HIGH | 7A |
| 5 | Random split, not session-aware — temporal leakage | HIGH | 7B |
| 6 | LR scheduler ordering bug | MEDIUM | 7A |
| 7 | No LR reduction on plateau — cosine schedule causes post-breakthrough collapse | MEDIUM | 7A |

---

## Implementation Phases

### Phase 7A: Training Stability Fixes — `._docs/phase-7a-training-stability.md`
Quick wins, no architectural changes. Fix scheduler bug, add ReduceLROnPlateau, boost augmentation defaults, update Colab notebook configs.
- Files: `trainer.py`, `config.py`, `07_colab_training.ipynb`

### Phase 7B: Data Pipeline Overhaul — `._docs/phase-7b-data-pipeline.md`
Change how data is prepared and fed to models. Wire firing rate features, session-aware splitting, trial type oversampling.
- Files: `dataset.py`, `loader.py`, `train.py`, `config.py`

### Phase 7C: Model Temporal Downsampling — `._docs/phase-7c-model-downsampling.md`
Add 4x temporal downsampling to GRU, CNN-LSTM, Transformer. Expose downsample_factor property. Adjust CTC input_lengths in trainer.
- Files: `gru_decoder.py`, `cnn_lstm.py`, `transformer.py`, `cnn_transformer.py`, `trainer.py`

**Execution order:** 7A → 7B → 7C (each builds on the last, each independently testable)
