# 1. Add a New Stage Before Training: **Neural Recording Diagnostics**

## Where it should go

Between:

```
Phase 0: Project Setup
```

and

```
Phase 1: MVP
```

or as **Phase 0.5: Signal Diagnostics & QC**

Right now pipeline begins at:

```
data → model
```

Real BCI pipelines are:

```
data → signal quality diagnostics → preprocessing → model
```

## What should be added

A section like:

```
Phase 0.5: Neural Recording Diagnostics
```

Tasks would include:

• channel variance analysis
• bad electrode detection
• SNR estimation
• power spectrum analysis
• trial artifact detection
• correlation matrix across electrodes

Example new modules:

```
src/diagnostics/
    channel_quality.py
    spectral_analysis.py
    snr_analysis.py
    trial_qc.py
    report_generator.py
```

Outputs:

```
outputs/quality_reports/
```

Also add a notebook:

```
notebooks/02_signal_diagnostics.ipynb
```

 existing notebook:

```
02_preprocessing_qc.ipynb
```

should be **expanded** to include these diagnostics.

---

# 2. Modify Phase 1 Model Definition to Reflect **Willett Architecture**

current Model A:

```
Conv1D → BiLSTM
```

The Willett system is closer to:

```
feature projection
→ GRU stack
→ linear classifier
→ CTC
```

You do **not need to remove CNN-LSTM**, but  should introduce a **new primary baseline**.

## Where to modify

Section:

```
1.6 Model A — Baseline
```

## What should change conceptually

Instead of one baseline, define:

```
Model A: Willett-style GRU decoder
Model B: CNN-LSTM (existing)
```

The new baseline should include:

```
Linear projection (192 → 256)
→ 3-layer GRU (512 hidden)
→ Linear classifier
→ CTC
```

New file:

```
src/models/gru_decoder.py
```

Then update comparison table later to include this architecture.

---

# 3. Add **Neural Smoothing Preprocessing**

Real neural decoding pipelines almost always apply temporal smoothing.

Preprocessing section currently includes augmentation but **no smoothing step**.

## Where it should go

Before the model input pipeline.

Likely inside:

```
src/data/transforms.py
```

Add a transform like:

```
GaussianTemporalSmoothing
```

Conceptually:

```
spike counts
→ gaussian smoothing (20–40 ms)
→ normalized firing rates
```

Add a test for it.

---

# 4. Expand Phase 3 Visualization to Include **Latent Representation Analysis**

 already have this partially:

```
embedding_plots.py
t-SNE/UMAP
```

This is excellent. But to match the research idea we discussed,  should expand it.

## Where to modify

Section:

```
3.3 Visualization of Results
```

## Add analyses such as

### Neural trajectory visualization

Plot hidden state evolution during a word.

Example module:

```
src/visualization/trajectories.py
```

Example output:

```
latent_state_t
```

projected to 2D.

### Trial similarity matrix

Add:

```
neural similarity heatmap
```

using cosine similarity between embeddings.

Module:

```
src/analysis/similarity_matrix.py
```

### Electrode importance maps

Use gradient attribution.

New module:

```
src/analysis/saliency.py
```

---

# 5. Expand the Demo to Include **Neural Analysis Tools**

Demo currently focuses on decoding.

To reflect real BCI tooling, it should also visualize **signal structure**.

## Where to modify

Phase:

```
4.3 Streamlit Frontend — Signal Viewer
```

Add additional capabilities:

• channel quality indicators
• SNR visualizations
• electrode importance heatmap
• latent embedding viewer

Dshboard already has a t-SNE viewer, which is perfect. Just expand it.

---

# 6. Update the Evaluation Table to Include **Architecture Comparison**

Later in Phase 2  compare:

```
CNN+LSTM
Transformer
Hybrid CNN-Transformer
```

After adding the Willett decoder, the comparison table should include:

```
GRU Decoder (Willett-style)
CNN-LSTM
Transformer
Hybrid CNN-Transformer
```

This strengthens the project academically because it shows **comparison against a known BCI architecture**.

---

# 7. Add One New Advanced Feature: **Neural Latent Explorer**

In Phase 6 (advanced features) add something like:

```
6.6 Neural Representation Explorer
```

Tasks:

• extract hidden layer embeddings
• project using UMAP
• visualize per-character clusters
• show trajectory evolution across time

Connects project to modern **neural manifold analysis** research.

---

# Summary of Required Edits

 mainly need **five structural changes**.

### 1

Add new phase:

```
Signal Diagnostics
```

### 2

Add a **Willett GRU model** to Phase 1.

### 3

Add **temporal smoothing preprocessing**.

### 4

Expand Phase 3 analysis with:

```
trajectory plots
saliency maps
similarity matrices
```

### 5

Expand the dashboard with **analysis visualizations**, not just decoding.

---

---------------

---
Final Last Touch Fixes:

  ┌─────┬──────────────────────┬──────────────────────────────┐   
  │  #  │        Issue         │             Fix              │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ Pathway table had    │ Updated to GRU=A,            │   
  │ 1-2 │ old model names      │ CNN+LSTM=B, Transformer=C,   │   
  │     │ A/B/C                │ Hybrid=D with GRU's built-in │   
  │     │                      │  projection added            │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │ 3   │ Phase 2 goal said    │ Changed to "four models"     │   
  │     │ "three models"       │                              │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ Section 2.1          │ Changed to "Model B          │   
  │ 4   │ referenced "Model A  │ (CNN+LSTM)"                  │   
  │     │ (CNN+LSTM)"          │                              │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ Feature ablation     │ Changed to "Model D's" +     │   
  │ 5   │ referenced "Model    │ added "Model A's built-in    │   
  │     │ C's CNN front-end"   │ linear projection"           │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ PRD risks said       │ Changed to "Model A (GRU     │   
  │ 6   │ "Model A (CNN+LSTM)" │ Decoder) or Model B          │   
  │     │                      │ (CNN+LSTM)"                  │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ PRD CTC section said │                              │   
  │ 7   │  "see Module 7" for  │ Changed to "see Module 8"    │   
  │     │ LM                   │                              │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ PRD said "Two        │ Changed to "Three pathways"  │   
  │ 8   │ pathways"            │ + noted Models A and D have  │   
  │     │                      │ built-in input stages        │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ Notebook named 04b   │                              │   
  │ 9   │ in 3.3b but 05b in   │ Changed 3.3b to 05b to match │   
  │     │ 5.2                  │                              │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │ 10  │ GRU decoder missing  │ Added as row with built-in   │   
  │     │ from pathway table   │ linear projection            │   
  ├─────┼──────────────────────┼──────────────────────────────┤   
  │     │ Diagnostics vs       │ 1.2 now consumes diagnostics │   
  │ 11  │ preprocessing        │  output, with inline         │   
  │     │ overlap              │ fallback                     │   
  └─────┴──────────────────────┴──────────────────────────────┘   


  Phase 1 Implementation Complete

  Preprocessing (Tasks 1.1-1.3b):
  - src/preprocessing/filter.py — Bandpass filter (4th-order      
  Butterworth), notch filter (60/120 Hz), artifact rejection,     
  temporal downsampling, GaussianTemporalSmoothing
  - src/preprocessing/normalize.py — Per-channel z-score
  normalization with training-set stats, clip to [-5,5], bad      
  channel detection (inline fallback + diagnostics integration),  
  channel removal
  - src/preprocessing/segment.py — Trial segmentation with        
  onset/offset + padding, pad/truncate to fixed t_max

  Feature Extraction (Task 1.4):
  - src/features/firing_rate.py — Pathway C: 10ms non-overlapping 
  binning + square-root transform

  PyTorch Dataset (Task 1.5):
  - src/data/dataset.py — NeuralTrialDataset, vocabulary (28      
  classes), ctc_collate_fn, train/val/test split (80/10/10        
  stratified by subject), create_dataloaders()

  Models (Tasks 1.6-1.6b):
  - src/models/base.py — Abstract BaseDecoder base class
  - src/models/gru_decoder.py — Model A:
  Linear(192→256)+ReLU+Dropout → 3-layer GRU(512) → Linear →      
  logits
  - src/models/cnn_lstm.py — Model B: 3×Conv1D(256,k=7,BN+ReLU) → 
  BiLSTM(512,2-layer) → Linear → logits

  Training (Task 1.7):
  - src/training/ctc_loss.py — CTC loss wrapper with log-softmax +   time-first transpose
  - src/training/scheduler.py — Cosine annealing with linear      
  warmup
  - src/training/trainer.py — Full Trainer class: train loop,     
  validation with greedy CER, checkpointing, early stopping       
  (patience=20), mixed precision support

  Decoding (Task 1.8):
  - src/decoding/greedy.py — Greedy CTC decode: argmax → collapse 

  Phase 2: Enhanced Models & Decoding — Complete        

  New Files Created (6 source + 4 test files)

  Feature Extraction:
  - src/features/temporal_conv.py — Pathway A:
  Multi-scale 1D conv bank (kernels [3, 7, 15]),        
  BatchNorm+ReLU, optional max pooling
  - src/features/projection.py — Pathway B: Linear      
  projection to d_model + sinusoidal positional encoding

  Models:
  - src/models/transformer.py — Model C: Pathway B input   → 6-layer Transformer encoder (d_model=512, 8 heads, 
  FFN=2048) → Linear logits. Supports padding masks.    
  - src/models/cnn_transformer.py — Model D: 3-layer CNN   front-end with MaxPool(2) → 8x temporal reduction    
  (T=2000→250) → 4-layer Transformer → Linear logits    

  Decoding:
  - src/decoding/beam_search.py — Pure-Python CTC prefix   beam search with configurable beam width, top-k      
  output, optional LM shallow fusion. Returns
  Hypothesis(text, score, char_probs) objects.
  - src/decoding/lm_correction.py — KenLM integration   
  (KenLMScorer), DummyLMScorer fallback,
  rescore_hypotheses() with shallow fusion formula:     
  (1-α)*CTC + α*LM + β*len

  Tests (58 new):
  - tests/test_temporal_conv.py — 8 tests (shapes,      
  pooling, gradients, custom kernels)
  - tests/test_projection.py — 8 tests (PE varies across   positions, shapes, gradients)
  - tests/test_models.py — Updated with 14 new tests for   Transformer and CNN-Transformer (shapes, gradient    
  flow, padding mask, 2000→250 reduction)
  - tests/test_beam_search.py — 11 tests (CTC decoding, 
  repeated chars, spaces, batch, comparison with greedy)  - tests/test_lm_correction.py — 7 tests (dummy scorer,   rescoring, length bonus, loader fallback)

  Updated Files:
  - scripts/train.py — Added TransformerDecoder and     
  CNNTransformer to MODEL_REGISTRY with proper config   
  parameter passing


 Phase 3: Evaluation, Ablations & Multi-Dataset — Complete       

  3.1 Full Evaluation Suite

  - src/evaluation/ablations.py — Ablation runner
  (run_ablation_suite, run_single_evaluation), per-character      
  confusion matrix, per-character CER breakdown, per-subject      
  metrics, paired bootstrap significance testing (n=1000), export 
  to JSON + CSV

  3.2 Ablation Studies

  - Integrated into the ablation runner — supports model ×        
  decoding × augmentation combos, run_significance_tests() for    
  pairwise model comparison

  3.3 Visualization of Results

  - src/visualization/ctc_plots.py — CTC probability heatmap [time   × characters], per-character error bar charts, training curve  
  comparison (4 models on one figure), confusion matrix heatmap   
  - src/visualization/embedding_plots.py — t-SNE/PCA embedding    
  scatter plots colored by character class

  3.3b Neural Latent Representation Analysis

  - src/analysis/embeddings.py — Extract hidden-layer embeddings  
  from any model via forward hooks, save/load to .npz
  - src/analysis/trajectory_plots.py — Per-timestep hidden state  
  extraction, 2D PCA neural trajectories with time-coloring,      
  multi-trial trajectory overlays
  - src/analysis/similarity_matrix.py — Pairwise cosine
  similarity, character-grouped similarity heatmaps
  - src/analysis/saliency.py — Input×gradient and integrated      
  gradients attribution, per-electrode importance, electrode      
  heatmaps

  3.4 UCSF ECoG Dataset Integration

  - Extended src/data/loader.py with download_ucsf_ecog(),        
  load_ucsf_ecog_dataset(), preprocess_ecog() (70-150 Hz
  high-gamma bandpass + Hilbert transform)

  3.5 OpenNeuro Dataset Integration

  - Extended src/data/loader.py with download_openneuro(),        
  load_openneuro_dataset(), BIDS-format parsing (TSV events, JSON 
  sidecars, EDF files via MNE)

  Tests

  - 8 new test files, 105 new tests — all passing
  - 339 total tests across the full suite — zero failures, zero   
  regressions
  
  implement phase 4 of the 'c:/Users/Daniel Lopez/Desktop/Neet-a- 
thon/BCI-2/._docs/implementation-task-list.md'... run tests to    
verify your work once your done, making new ones where need be"