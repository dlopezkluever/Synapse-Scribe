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
