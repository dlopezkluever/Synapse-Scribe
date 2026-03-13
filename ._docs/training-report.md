# GRU Decoder Training Report
**Date:** March 11, 2026
**Model:** GRU Decoder (Willett-style)
**Dataset:** Willett Handwriting BCI (4,126 trials)
**Hardware:** CPU only (no GPU available)

---

## 1. Objective

Train the GRU Decoder — the primary baseline model — on the real Willett handwriting BCI dataset, as described in Step 3 of our [training plan](./real-data-training-plan.md). The goal was to go from random weights to a model that can decode neural activity into character sequences using CTC (Connectionist Temporal Classification).

---

## 2. Dataset Analysis

Before training, we analyzed the Willett dataset to understand what we were working with.

### 2.1 Trial Composition

| Trial Type | Count | Signal Length | Label Length |
|------------|-------|---------------|--------------|
| Single letters | 3,042 | 201 timesteps (all identical) | 1 character |
| Sentences | 967 | 323–13,876 timesteps (median 3,842) | 3–116 characters (mean 43) |
| **Total** | **4,126** | | |

### 2.2 Key Data Properties

- **Signal format:** uint8 spike counts, 192 channels, range [0, 6], mean 0.24
- **Temporal ratio:** ~90 timesteps per character in sentence data
- **10 recording sessions** across different dates
- **Vocabulary:** {blank=0, a–z=1–26, space=27} = 28 classes

### 2.3 Critical Finding: Variable Sequence Lengths

The massive variation in signal lengths (201 to 13,876 timesteps) created a core challenge for training:

| t_max Setting | Trials That Fit | Coverage |
|---------------|-----------------|----------|
| 500 | 3,167 | 77% |
| 2,000 | 3,362 | 81% |
| 4,000 | 3,868 | 94% |
| 8,000 | 4,087 | 99% |

With `t_max=2000`, **764 sentence trials** (79% of sentences) would be truncated, losing over half their signal while keeping the full label. This signal/label mismatch is a known problem for CTC training.

---

## 3. Training Attempts

We made four training attempts, each informed by the failures and observations of the previous one.

### 3.1 Attempt 1: Full-Size GRU on All Data (Abandoned)

**Configuration:**
- Model: GRU(hidden=512, layers=3, params=4.4M)
- Data: All 4,126 trials, t_max=2000, batch_size=16
- No normalization

**What happened:** The first epoch never completed. With [16, 2000, 192] tensors flowing through a 3-layer GRU(512) on CPU, each batch took too long. After 10+ minutes with no epoch output, we killed the process.

**Lesson:** The full-size GRU is too slow for CPU training with 2000-timestep sequences. We needed either a smaller model or shorter sequences.

---

### 3.2 Attempt 2: Full-Size GRU, Single-Letter Trials Only (Stopped at Epoch 23)

**Rationale:** Single-letter trials are all exactly 201 timesteps — 10x shorter than sentence trials. This dramatically reduces per-batch compute while still testing whether the model can learn from the data.

**Configuration:**
- Model: GRU(hidden=512, layers=3, params=4.4M)
- Data: 3,159 single-letter trials only, t_max=250, batch_size=32
- No normalization
- Split: 2,527 train / 315 val / 317 test

**Results (ran for 23 epochs before manual stop, ~2 hours):**

| Epoch | Train Loss | Val Loss | Val CER | Notable |
|-------|-----------|----------|---------|---------|
| 1 | 430.67 | 5.94 | 0.968 | All blank predictions |
| 5 | 3.31 | 3.32 | 0.965 | Mode collapse: predicts 'm' for everything |
| 9 | 3.24 | 3.31 | 0.949 | First diverse predictions |
| 16 | 2.94 | 3.41 | **0.937** | Best CER; first correct prediction ('o'='o') |
| 23 | 2.66 | 3.63 | 0.949 | Severe overfitting; stopped manually |

**Observations:**
- The model broke out of the "all blank" phase by epoch 4
- It went through a "mode collapse" phase (epochs 5-6) predicting a single character for all inputs
- By epoch 16, it achieved its best CER of 0.937 (~6.3% accuracy, vs ~3.7% random)
- **Severe overfitting** developed: train loss 2.66 vs val loss 3.63 by epoch 23
- The model could never distinguish more than ~6% of letters correctly

**Why it struggled:**
1. **CTC is designed for sequences, not single-character classification.** For a 1-character label over 201 timesteps, CTC has to learn that 200 timesteps are blank and 1 has the character — an awkward formulation of what is fundamentally a classification problem.
2. **No data normalization.** The raw spike counts (range 0–6) were not standardized, making gradient optimization less effective.
3. **GaussianNoise augmentation too weak.** The augmentation added noise with std=0.01, negligible against data in range 0–6.

---

### 3.3 Attempt 3: Full-Size GRU, Sentences Only with Normalization (Abandoned)

**Rationale:** Sentences are the real use case for CTC. We added z-score normalization and filtered to sentence trials that fit within `t_max=4000`.

**Configuration:**
- Model: GRU(hidden=512, layers=3, params=4.4M)
- Data: 514 sentence trials (filtered to ≤4000 timesteps), **normalized**
- t_max=4000, batch_size=4
- Split: 411 train / 51 val / 52 test

**What happened:** Same problem as Attempt 1 — the first epoch never completed after 30+ minutes. Even with only 514 trials, the combination of [4, 4000, 192] tensors and the full-size GRU was too slow on CPU.

**Lesson:** On CPU, we cannot run the full-size GRU on long sequences at any batch size. We need a smaller model.

---

### 3.4 Attempt 4: Reduced GRU, All Data, with Normalization and Bucketed Batching (Completed)

**Rationale:** We applied every optimization we'd learned:
1. **Smaller model** — GRU(hidden=256, layers=2) = 846K params (vs 4.4M) for ~6x faster per-step compute
2. **Z-score normalization** — standardize per channel (mean=0.205, std=0.443)
3. **Dynamic padding** — pad to max length *within each batch*, not a fixed t_max
4. **Bucketed sampling** — group similar-length trials into batches so letter batches are [B, 201, 192] and sentence batches are [B, ~2000, 192]
5. **All data** — use both letters and sentences to maximize training signal

**Configuration:**
- Model: GRU(hidden=256, layers=2, params=846,108)
- Data: All 4,126 trials, t_max=2000, batch_size=16, **normalized**
- 764 sentence trials truncated from >2000 to 2000 timesteps
- Split: 3,300 train / 412 val / 414 test
- Optimizer: AdamW, lr=3e-4, cosine warmup schedule
- Early stopping: patience=20 epochs

**Full epoch-by-epoch results:**

| Epoch | Train Loss | Val Loss | Val CER | Time (s) | Phase |
|-------|-----------|----------|---------|----------|-------|
| 1 | 224.68 | 4.16 | 0.998 | 261 | All blank outputs |
| 2 | 4.16 | 3.98 | 0.998 | 367 | Still blank |
| 3 | 4.08 | 4.00 | 0.993 | 648 | First character: 'a' |
| 4 | 4.10 | 4.13 | 0.987 | 607 | Exploring: 'o' |
| 5 | 3.93 | 3.97 | 0.990 | 336 | Loss dropping fast |
| 6 | 3.71 | 3.74 | 0.988 | 311 | Train/val tracking together |
| 7 | 3.46 | 3.48 | 0.987 | 293 | **First correct prediction:** 'w'='w' |
| 8 | 3.03 | 3.38 | 0.988 | 353 | Rapid train loss drop |
| 9 | 2.88 | 3.43 | 0.986 | 278 | Predicting diverse chars for all samples |
| 10 | 2.76 | 3.47 | 0.986 | 490 | Overfitting gap developing |
| **11** | **2.65** | **3.54** | **0.985** | **496** | **Best CER checkpoint saved** |
| 12 | 2.58 | 3.75 | 0.986 | 599 | Val loss diverging |
| 13 | 2.53 | 3.64 | 0.987 | 555 | Stable wrong predictions |
| 14 | 2.43 | 3.74 | 0.985 | 689 | |
| 15 | 2.37 | 3.78 | 0.985 | 602 | |
| 16 | 2.28 | 3.86 | 0.987 | 647 | |
| 17 | 2.25 | 3.93 | 0.987 | 532 | |
| 18 | 2.18 | 3.98 | 0.988 | 590 | |
| 19 | 2.12 | 4.02 | 0.986 | 488 | |
| 20 | 2.05 | 4.08 | 0.986 | 623 | |
| 21 | 2.00 | 4.17 | 0.986 | 346 | |
| 22 | 1.95 | 4.42 | 0.987 | 219 | |
| 23 | 1.91 | 4.28 | 0.988 | 226 | |
| 24 | 1.88 | 4.39 | 0.988 | 230 | |
| 25 | 1.82 | 4.43 | 0.986 | 247 | |
| 26 | 1.78 | 4.45 | 0.986 | 268 | |
| 27 | 1.72 | 4.49 | 0.986 | 280 | |
| 28 | 1.70 | 4.52 | 0.985 | 292 | |
| 29 | 1.65 | 4.55 | 0.987 | 235 | |
| 30 | 1.58 | 4.65 | 0.985 | 235 | |
| **31** | **1.58** | **4.70** | **0.985** | **291** | **Early stopping triggered** |

**Final result:** Best val CER = **0.9848** at epoch 11, early stopping at epoch 31.
**Total training time:** ~3.5 hours (18:13 to 21:44).

---

## 4. Training Phases Observed

The model went through clearly identifiable learning phases:

### Phase 1: All Blank (Epochs 1–2)
The model output empty strings for every input. CTC loss was high. This is the expected initial state — outputting blank at every timestep yields a reasonable CTC loss because blanks are "free" in the CTC formulation.

### Phase 2: First Characters (Epochs 3–6)
The model began predicting single characters ('a', 'o', 'g'), but the same character for most inputs. Train and val losses were dropping together — no overfitting yet. This shows the model learning that *some* character is better than blank.

### Phase 3: Character Differentiation (Epochs 7–11)
The model started predicting *different* characters for different inputs. At epoch 7, it made its first correct prediction ('w' for 'w'). By epoch 9, all three sample predictions were non-blank and distinct. CER improved from 0.987 to 0.985.

### Phase 4: Overfitting Plateau (Epochs 12–31)
Train loss continued dropping (2.58 → 1.58) while val loss increased (3.75 → 4.70). CER flatlined at ~0.985 with no further improvement. The model memorized training data but could not generalize. Early stopping triggered after 20 consecutive epochs without CER improvement.

---

## 5. Infrastructure Improvements Made

To support training, we built several optimizations into the pipeline:

### 5.1 Dynamic Padding (src/data/dataset.py)
**Before:** Every trial was padded to a fixed `t_max` regardless of actual length. A 201-timestep letter trial padded to 2000 wasted 90% of compute.

**After:** The collate function pads to the *max length within each batch*. Letter batches are [B, 201, 192] and sentence batches are [B, ~2000, 192]. This reduced unnecessary computation by ~10x for letter-heavy batches.

### 5.2 Bucketed Sampling (src/data/dataset.py)
Trials are sorted by length and grouped into batches of similar-length sequences. This maximizes the benefit of dynamic padding — a batch of all-letter trials only processes 201 timesteps instead of 2000.

### 5.3 Z-Score Normalization (src/data/dataset.py)
Per-channel zero-mean, unit-variance normalization computed from the training set. Stats are shared with val/test sets to prevent data leakage. The raw data (uint8 spike counts, range 0–6, mean 0.24) is standardized to a distribution centered at 0 with std ~1.

### 5.4 Training Script Enhancements (scripts/train.py)
Added 6 new CLI flags for flexible training:
- `--trial-type {all,letters,sentences}` — filter by trial type
- `--filter-by-length` — drop trials exceeding t_max
- `--normalize` — enable z-score normalization
- `--hidden-size` — override GRU hidden dimension
- `--n-layers` — override number of recurrent layers
- `--proj-dim` — override input projection dimension

---

## 6. Analysis: Why CER Plateaued at 0.985

The model improved CER from 0.998 (random) to 0.985 (about 1.3 percentage points) then plateaued despite train loss continuing to drop. Several factors explain this:

### 6.1 Reduced Model Capacity
We used a GRU with 846K parameters (hidden=256, 2 layers) instead of the original 4.4M (hidden=512, 3 layers). This was necessary for CPU training speed but significantly limits the model's ability to learn complex neural-to-character mappings. The Willett paper used the full-size architecture with GPU training.

### 6.2 Truncated Sentence Data
764 out of 967 sentence trials (79%) were truncated from >2000 to 2000 timesteps. At ~90 timesteps per character, 2000 timesteps covers only ~22 characters of a sentence that averages 43 characters. CTC sees the full label but only partial neural activity, creating a fundamental mismatch.

### 6.3 CPU Training Speed Constraints
Even with all optimizations, epochs took 4–11 minutes. The model only ran for 31 epochs total. Deep learning models typically need 100+ epochs to converge. The Willett paper trained for longer with GPU acceleration.

### 6.4 Mixed Data Complexity
Training on both single-letter trials (1 character, 201 timesteps) and sentence trials (43 characters, 2000 timesteps) in the same batches creates a complex optimization landscape. The model must learn two very different temporal patterns simultaneously.

---

## 7. Comparison with Expected Results

From our [training plan](./real-data-training-plan.md), here's what we expected vs. what we achieved:

| Training Stage | Expected CER | Actual CER | Notes |
|---------------|-------------|-----------|-------|
| Epoch 1–10 | ~100% (blanks/repeats) | 99.8% → 98.6% | Matches expectation |
| Epoch 10–50 | 60–80% | 98.5% (plateaued) | Much worse than expected |
| Epoch 50–100 | 20–40% | N/A (stopped at 31) | Never reached |
| Epoch 100–200 | 5–15% | N/A | Never reached |

The plan's projections assumed the **full-size model on GPU** training for 200 epochs on properly-sized sequences. Our CPU constraints forced a smaller model with shorter sequences, explaining the gap.

---

## 8. Saved Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Best checkpoint | `outputs/checkpoints/GRUDecoder_best.pt` | Epoch 11, CER=0.9848 |
| Training config | Inline in training script | hidden=256, layers=2, lr=3e-4, t_max=2000 |

The checkpoint contains: model state dict, optimizer state dict, epoch number, val CER, global step count.

---

## 9. Recommendations for Better Results

In order of expected impact:

1. **Use GPU training (Google Colab or local GPU).** This is the single biggest improvement — it enables the full-size GRU (4.4M params), larger t_max (5000+), and 200+ epoch training. Expected to reach CER < 20%.

2. **Increase t_max to 5000+** to avoid truncating 79% of sentence trials. This requires GPU due to memory and compute requirements.

3. **Train on sentences only** with the full-size model. Single-letter trials add noise to the CTC training signal and are better suited for a classification head, not CTC.

4. **Increase augmentation strength.** The current GaussianNoise std of 0.01 is negligible against normalized data (std=1.0). Increase to 0.1–0.3 to help generalization.

5. **Add a language model.** The LM rescoring pipeline is already built (beam search + KenLM). Even a simple character n-gram model could reduce CER by 2–5% on top of any acoustic model improvement.

---

## 10. Conclusion

We successfully built and ran the full training pipeline on real Willett BCI data, from raw `.mat` files through preprocessing, dataset creation, and CTC training with the GRU decoder. The model demonstrated clear learning (progressing through blank → mode collapse → character differentiation phases) and achieved a best CER of 0.985 — a meaningful improvement over random (1.0).

However, CPU training imposed severe constraints that prevented reaching publication-quality results. The model was limited to 846K parameters (vs 4.4M designed), 2000-timestep sequences (vs 5000+ needed), and 31 epochs (vs 200 intended). The resulting overfitting and CER plateau at 0.985 are direct consequences of these constraints.

The pipeline is validated and ready. GPU training with the full-size model is the clear next step to achieve the <10% CER results demonstrated in the Willett paper.

---
---

# GPU Training Report: All Four Architectures
**Date:** March 13, 2026
**Hardware:** NVIDIA T4 GPU (Google Colab)
**Dataset:** Willett Handwriting BCI (3,691–3,868 trials after length filtering)
**New Feature:** Temporal downsampling (4x for GRU/CNN-LSTM/Transformer, 8x for CNN-Transformer)

---

## 11. Overview

Following the CPU baseline (Section 10), we trained all four model architectures on GPU with full-size configurations, z-score normalization, and the newly implemented temporal downsampling (Phase 7C). Temporal downsampling reduces CTC's blank ratio by compressing the time axis before decoding — 4x via strided Conv1d layers for GRU/CNN-LSTM/Transformer, and 8x via the CNN-Transformer's integrated MaxPool front-end.

### Results Summary

| Model | Params | Epochs (Early Stop) | Time | Best Val CER | Best Val WER | Status |
|-------|--------|---------------------|------|-------------|-------------|--------|
| CNN-LSTM | 10.7M | 173 (stop @ 153) | 111 min | **0.0424 (4.2%)** | ~0.14 | Excellent |
| CNN-Transformer | 13.3M | 89 (stop @ 69) | 37.7 min | 0.3516 (35.2%) | ~0.53 | Moderate |
| GRU Decoder | 4.9M | 28 (stop @ 8) | 14.2 min | 0.9817 (98.2%) | ~0.98 | Failed |
| Transformer | 20.7M | 21 (stop @ 1) | 12.3 min | 1.0000 (100%) | 1.00 | Failed |

**Total GPU time:** ~175 minutes (~2.9 hours).

---

## 12. CNN-LSTM: The Winner (4.2% CER)

### 12.1 Configuration
- **Model:** CNNLSTM (10.7M params) — 3-layer CNN (conv_channels=256, kernel=7, stride-2 on first 2 layers for 4x downsample) → 2-layer LSTM (hidden=512)
- **Data:** 3,868 trials (filtered to ≤5000 timesteps), z-score normalized
- **Training:** batch_size=12, lr=1e-4, dropout=0.5, mixed precision (fp16)
- **Split:** 3,094 train / 386 val / 388 test

### 12.2 Training Phases

The CNN-LSTM went through five distinct learning phases:

**Phase 1: All Blank (Epochs 1–6)**
CER ~99%. The model output empty strings. Standard CTC initialization behavior — predicting blank at every frame minimizes initial loss. Mode collapse to single characters ('u', 'n', 'f', 'w') emerged briefly.

**Phase 2: Single-Character Differentiation (Epochs 7–34)**
CER dropped slowly from 98% to 89%. The model learned to predict correct single-letter trials ('r'='r', 'k'='k') but still output single characters for sentences. First correct letter predictions at epoch 11 (CER 97.3%).

**Phase 3: Sentence Breakthrough (Epochs 35–55)**
The critical inflection point. At epoch 35, CER plummeted from 89% to 72% as the model began outputting multi-character sequences for sentences:
- Epoch 35: `'ta aes e fefsa'` for `'daisy leaned forward at once horrified and fascina'` — first sentence-like output
- Epoch 38: `'tary hansdswosaf nes moinsid ane fafrmd eo'` — word-level structure emerging
- Epoch 50: `'caisy hecned foonccrndi at once houmrifiicd anc'` — 22.4% CER
- Epoch 55: `'daisy learad foorwand at once hourrifisd and fasci'` — 16.5% CER

**Phase 4: Rapid Refinement (Epochs 55–107)**
CER dropped from 16.5% to ~5%. Character-level accuracy improved dramatically:
- Epoch 73: `'daisy leaned forward at once hourrifisd and fascin'` — 8.5% CER
- Epoch 94: `'daisy heaned forwardx at once horrified and fascin'` — 6.4% CER
- Epoch 107: `'daisy leaned forward at once horrified and fascina'` — exact match on sample sentence, 5.5% CER overall

**Phase 5: Fine-Tuning Plateau (Epochs 107–173)**
CER oscillated between 4.2%–5.3%. Remaining errors were minor: character doublings (`'forwarrd'`), substitutions (`'ard'` for `'and'`), and occasional truncations. Best checkpoint saved at epoch 153 with CER=0.0424.

### 12.3 Key Epoch Milestones

| Epoch | Train Loss | Val Loss | Val CER | Prediction Sample |
|-------|-----------|----------|---------|-------------------|
| 1 | 50.47 | 4.33 | 1.000 | `''` (blank) |
| 11 | 2.57 | 2.88 | 0.973 | `'r'` (single char) |
| 22 | 1.58 | 2.58 | 0.948 | `'t'` / `'r'` / `'k'` (correct letters) |
| 35 | 0.77 | 1.61 | 0.725 | `'ta aes e fefsa'` (first sentence fragments) |
| 50 | 0.27 | 0.76 | 0.224 | `'caisy hecned foonccrndi at once houmrifiicd anc'` |
| 73 | 0.09 | 0.56 | 0.085 | `'daisy leaned forward at once hourrifisd and fascin'` |
| 107 | 0.03 | 0.47 | 0.055 | `'daisy leaned forward at once horrified and fascina'` |
| 153 | 0.03 | 0.52 | **0.042** | `'daisy leaned forward at once horrified and fascina'` |

### 12.4 Why CNN-LSTM Worked

1. **CNN front-end provides local temporal features.** The strided convolutions extract local spike patterns before the LSTM sees them, acting as a learned feature extractor that reduces noise and highlights task-relevant neural dynamics.
2. **4x temporal downsampling is the sweet spot.** Reducing 5000 timesteps to 1250 frames gives CTC a manageable blank ratio while preserving enough temporal resolution for character-level alignment.
3. **High dropout (0.5) prevented overfitting.** Despite 10.7M parameters on only ~3,000 training trials, the model generalized well — val loss stabilized around 0.47 while train loss hit 0.01.
4. **LSTM captures long-range dependencies.** Unlike the pure CNN approach, the LSTM can model sequential dependencies across the sentence, critical for character ordering.

---

## 13. CNN-Transformer: Promising but Plateaued (35.2% CER)

### 13.1 Configuration
- **Model:** CNNTransformer (13.3M params) — 3-layer CNN with MaxPool(2) each (8x downsample) → 4-layer Transformer encoder (d_model=512, 8 heads)
- **Data:** 3,691 trials (filtered to ≤4096 timesteps), z-score normalized
- **Training:** batch_size=8, lr=1e-4, dropout=0.1, mixed precision

### 13.2 Training Progression

The CNN-Transformer showed steady improvement but stalled around epoch 69:
- **Epochs 1–8:** All blank output (CER=100%)
- **Epoch 10:** First character predictions (CER=88.5%)
- **Epoch 28–32:** Rapid drop from 77% to 60% CER as single-letter trials decoded correctly
- **Epoch 48:** CER=42.2%, single letters mostly correct
- **Epoch 69:** Best CER=35.2%, then plateaued and early-stopped at epoch 89

### 13.3 Issues

1. **8x downsampling is too aggressive.** Compressing 4096 timesteps to 512 frames loses fine-grained temporal detail needed for CTC alignment on long sentences. The CNN-LSTM's 4x downsampling preserves twice the temporal resolution.
2. **Overfitting with val loss oscillation.** Val loss bounced between 0.9–2.1 while train loss was ~0.05, indicating poor generalization. The model memorized training patterns but couldn't transfer to validation.
3. **CTC repetition collapse.** At epoch 52, the model briefly produced outputs like `'rxxxxxxxxxx'` (CER spiked to 97.7%), a classic CTC failure mode where the model gets stuck repeating a character. It recovered, but this suggests instability near the decision boundary.
4. **Low dropout (0.1) was insufficient** for the data size. The CNN-LSTM used 0.5 dropout, which may explain its better generalization.

---

## 14. GRU Decoder: Failed (98.2% CER)

### 14.1 Configuration
- **Model:** GRUDecoder (4.9M params) — Conv1d downsampler (4x) → 3-layer GRU (hidden=512)
- **Data:** 3,868 trials (filtered to ≤5000 timesteps), z-score normalized
- **Training:** batch_size=16, lr=1e-4, dropout=0.3, mixed precision

### 14.2 What Happened

The GRU never broke out of the single-character prediction phase across 28 epochs. While CER briefly improved from 100% to 98.2% (epoch 8), it never produced multi-character outputs for sentences. Train loss dropped steadily (70.99 → 2.26) but val loss diverged (4.22 → 4.84), indicating severe overfitting to trivial patterns.

### 14.3 Why It Failed

1. **Pure recurrent over long sequences.** Even with 4x downsampling, the GRU processes 1250-frame sequences without any local feature extraction. The Conv1d downsampler is too thin (2 layers) to substitute for a proper CNN front-end.
2. **Insufficient capacity for the task.** At 4.9M params, the GRU is the smallest model tested. The successful CNN-LSTM has 10.7M params with a more structured architecture.
3. **No local feature extraction.** The GRU sees raw (downsampled) neural channels and must learn both local spike patterns and sequential dependencies simultaneously — too much for a single recurrent module.

---

## 15. Transformer Decoder: Failed (100% CER)

### 15.1 Configuration
- **Model:** TransformerDecoder (20.7M params) — Conv1d downsampler (4x) → 6-layer Transformer encoder (d_model=512, 8 heads)
- **Data:** 3,691 trials (filtered to ≤4096 timesteps), z-score normalized
- **Training:** batch_size=8, lr=1e-4, dropout=0.1, mixed precision

### 15.2 What Happened

The Transformer never produced a single character in 21 epochs. CER remained at exactly 100% throughout. Train loss dropped (11.25 → 1.34) but this was entirely from the model learning to optimize blank token probabilities without ever emitting characters.

### 15.3 Why It Failed

1. **Self-attention on raw neural signals doesn't work at this scale.** With only ~3,000 training samples, 6 layers of full self-attention over 1024 frames (after 4x downsample) can't discover meaningful patterns. The model has no inductive bias for local temporal structure.
2. **Too many parameters, too little data.** 20.7M parameters on 2,952 training samples is a 7000:1 parameter-to-sample ratio — massively overparameterized with no mechanism to constrain learning.
3. **Needs a CNN embedding layer.** The successful CNN-Transformer (35.2% CER) works precisely because its CNN front-end provides structured local features. Raw attention over neural time series requires far more data or explicit local structure.

---

## 16. Architecture Comparison

| | CNN-LSTM | CNN-Transformer | GRU | Transformer |
|---|---------|----------------|-----|-------------|
| **Local features** | 3-layer CNN | 3-layer CNN+MaxPool | 2-layer Conv1d | 2-layer Conv1d |
| **Sequence model** | 2-layer LSTM | 4-layer Transformer | 3-layer GRU | 6-layer Transformer |
| **Downsample** | 4x (stride) | 8x (MaxPool) | 4x (stride) | 4x (stride) |
| **Dropout** | 0.5 | 0.1 | 0.3 | 0.1 |
| **Best CER** | **4.2%** | 35.2% | 98.2% | 100% |
| **Converged?** | Yes | Partial | No | No |

**Key insight:** The presence of a strong CNN front-end is the primary differentiator. Both models with substantial CNN layers (CNN-LSTM, CNN-Transformer) achieved meaningful CER reductions. The 4x vs 8x downsampling difference accounts for the gap between CNN-LSTM and CNN-Transformer.

---

## 17. Known Issues from This Session

1. **LR scheduler ordering bug.** All four models logged a PyTorch warning: `lr_scheduler.step()` was called before `optimizer.step()`, causing the first LR value to be skipped. Should swap the call order in `src/training/trainer.py:237`.
2. **CNN-Transformer 8x downsampling bottleneck.** The 8x reduction likely hurts CTC alignment. Running with `--no-downsample` or reducing to 4x could significantly improve results.
3. **GRU and Transformer need architectural changes.** The thin Conv1d downsampler is insufficient — these models need either a proper CNN front-end or significantly more training data to work.

---

## 18. Comparison: CPU vs GPU Training

| Metric | CPU (March 11) | GPU (March 13) |
|--------|---------------|----------------|
| **Hardware** | CPU only | T4 GPU (Colab) |
| **Best model** | GRU (846K params) | CNN-LSTM (10.7M params) |
| **Best CER** | 0.985 (98.5%) | **0.0424 (4.2%)** |
| **Epoch time** | 4–11 min | 25–40 sec |
| **Total time** | 3.5 hours (1 model) | 2.9 hours (4 models) |
| **Epochs trained** | 31 | 173 (best model) |
| **Improvement** | — | **23x better CER** |

GPU training delivered a 23x improvement in CER while training four models in less total time than the single CPU run.

---

## 19. Next Steps

1. **Run beam search + LM on CNN-LSTM checkpoint.** The LM rescoring pipeline is already built. Even a simple character n-gram model could push CER below 3%.
2. **Reduce CNN-Transformer downsampling to 4x.** Test whether matching the CNN-LSTM's temporal resolution closes the gap.
3. **Fix LR scheduler call order** in trainer.py (swap lines 237/243).
4. **Test set evaluation.** Run the CNN-LSTM best checkpoint on the held-out test set for final numbers.

---

## 20. Saved Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| CNN-LSTM best | `outputs/checkpoints/GPU-3-13/CNNLSTM_best.pt` | Epoch 153, CER=0.0424 |
| CNN-Transformer best | `outputs/checkpoints/GPU-3-13/CNNTransformer_best.pt` | Epoch 69, CER=0.3516 |
| GRU best | `outputs/checkpoints/GPU-3-13/GRUDecoder_best.pt` | Epoch 8, CER=0.9817 |
| Transformer best | `outputs/checkpoints/GPU-3-13/TransformerDecoder_best.pt` | Epoch 1, CER=1.0000 |
| Raw training logs | `._docs/GPU-Training-Summary-313.md` | Full epoch-by-epoch output |
| W&B dashboard | `wandb.ai/dlopezkluever-aiuteur/brain-text-decoder` | Interactive metrics |
