# Plan: Training on Real Data & Getting Meaningful Decoding Results
---
## Summary: 

  8 steps, roughly half require your action:

  1. YOU — Download Willett dataset from Dryad (~10GB, browser download)
  2. Run diagnostics (one command)
  3. Train GRU decoder first (python scripts/train.py --model gru_decoder — 2-8hrs on    
  CPU)
  4. YOU — Review: is loss dropping? Is CER improving? Sample predictions look like      
  English?
  5. Train the other 3 models (overnight, sequential)
  6. YOU — Find an English text corpus + install KenLM (or skip this — system works      
  without it)
  7. Run full evaluation across all model/decoding combos
  8. YOU — Launch the demo, explore real results

  The biggest bottleneck is Step 1 (the data download) and Step 3 (training time on CPU).   Everything else is either fast or hands-off. The doc includes a troubleshooting       
  section for common training failures (all-blank output, loss not decreasing,
  overfitting) and realistic CER expectations at each training stage.


---
> **Current state:** The full pipeline (preprocessing, 4 models, CTC training, decoding, evaluation, demo app) is built and tested — but all models have random weights. No real training has happened. This plan gets us from "working infrastructure" to "actual brain-to-text decoding."

---

## Overview

| Step | What | Who | Time Estimate |
|------|------|-----|---------------|
| 1 | Download Willett dataset | **YOU** | 10–30 min (depends on internet) |
| 2 | Verify data & run diagnostics | Automated | ~5 min |
| 3 | Train GRU decoder (baseline) | Automated | 2–8 hours (CPU) |
| 4 | Validate convergence & debug | **YOU** (review) | 15 min |
| 5 | Train remaining 3 models | Automated | 6–24 hours (CPU) |
| 6 | Build a character-level KenLM model | Mixed | 30 min |
| 7 | Full evaluation & benchmarking | Automated | 30 min |
| 8 | Demo with real results | **YOU** (run & explore) | 5 min |

---

## Step 1: Download the Willett Dataset

### **YOU** need to do this — it cannot be fully automated.

The Willett handwriting BCI dataset is hosted on Dryad. The download requires a browser.

1. Go to: **https://doi.org/10.5061/dryad.wh70rxwmv**
2. Click the download button. It's a ~10 GB zip file.
3. Save it to:
   ```
   data/willett_handwriting/raw/willett_handwriting.zip
   ```
4. Extract it. After extraction you should have something like:
   ```
   data/willett_handwriting/raw/extracted/
     Datasets/
       t5.2019.05.08/
         singleLetters.mat
         sentences.mat
         ...
   ```

**Alternatively**, run the helper script which will give you instructions:
```bash
bash scripts/download_data.sh --dataset willett
```

### What if the download is too large or slow?

You can start with just the `singleLetters.mat` file from one session if the full dataset is too big. The loader will work with whatever `.mat` files it finds. Even a single session (~200–400 trials) is enough to validate that training works.

---

## Step 2: Verify Data & Run Diagnostics

Once the data is in place, run the quality check to make sure everything parsed correctly:

```bash
python scripts/run_quality_check.py --dataset willett
```

This will:
- Parse the `.mat` files into standardized `.npy` + `.txt` format
- Flag bad channels (zero variance, excessive noise)
- Compute SNR per channel
- Generate a report in `outputs/quality_reports/`

**What to check (YOU review):**
- The trial index (`data/willett_handwriting/trial_index.csv`) should have rows. If it's empty, the `.mat` files weren't found or couldn't be parsed.
- The quality report should show most channels as "good" (>150 out of 192).
- If many channels are flagged, that's fine — the preprocessing pipeline will exclude them.

---

## Step 3: Train the GRU Decoder (Primary Baseline)

This is the most important training run. The GRU decoder replicates the architecture from the Willett paper and should produce the best results on this specific dataset.

```bash
python scripts/train.py \
  --model gru_decoder \
  --epochs 200 \
  --batch-size 16 \
  --lr 3e-4 \
  --checkpoint-dir ./outputs/checkpoints
```

**What happens:**
- Loads all parsed trials, splits 80/10/10 train/val/test
- Applies augmentation (time masking, channel dropout, Gaussian noise) on train set
- Trains with CTC loss, AdamW optimizer, cosine warmup scheduler
- Every epoch: computes val CER via greedy decode, saves best checkpoint
- Early stops after 20 epochs of no improvement
- Best checkpoint saved to: `outputs/checkpoints/GRUDecoder_best.pt`

**On CPU this will be slow** (~2–8 hours depending on dataset size and your machine). If you have a CUDA GPU, it will use mixed precision automatically and be 5–10x faster.

### While it trains, watch for these signs:

| Signal | Meaning |
|--------|---------|
| Loss decreasing steadily | Good — model is learning |
| Loss stuck high (~3.3 for 28 classes) | Bad — model outputs uniform distribution, not learning |
| Val CER decreasing | Good — model produces better text over time |
| Val CER = 1.0 every epoch | Bad — model outputs all blanks or garbage |
| Loss goes to 0 quickly | Suspicious — possible overfitting or label leak |

---

## Step 4: Validate Convergence

### **YOU** review the training output.

After training finishes (or after ~50 epochs if you're impatient), check:

1. **Did CTC loss converge?** It should drop from ~3.3 (random) to <1.0.

2. **What's the val CER?** Target benchmarks:
   - CER < 40% = minimum success (model learned something)
   - CER < 20% = decent (approaching useful)
   - CER < 10% = good (competitive with literature)
   - CER ~5% = excellent (matches Willett paper)

3. **Look at sample predictions** (printed during training):
   ```
   Reference: "hello world"
   Predicted: "helo wold"     ← This is promising!
   Predicted: "aaaaaaa"       ← This is bad (mode collapse)
   Predicted: ""              ← This is bad (all blanks)
   ```

### Common problems and fixes:

**All-blank output:**
- The model learns that outputting blank every timestep gives decent CTC loss because blanks are "free" in CTC.
- Fix: Check that `target_length < input_length` for all trials. CTC requires the output sequence to be longer than the target. If T_MAX is too small or labels are too long, increase `--t-max`.

**Loss doesn't decrease at all:**
- Learning rate may be wrong. Try `--lr 1e-3` or `--lr 1e-4`.
- Data may not be loading correctly. Check that the `.npy` files have shape `[T, 192]` and are not all zeros.

**Severe overfitting (train CER ~0%, val CER ~80%):**
- Too few trials. Try pooling multiple sessions.
- Increase dropout or augmentation strength in `config.yaml`.

---

## Step 5: Train Remaining Models

Once GRU decoder works, train the other three for comparison:

```bash
# CNN + LSTM
python scripts/train.py --model cnn_lstm --epochs 200

# Transformer (may need smaller batch size due to memory)
python scripts/train.py --model transformer --epochs 200 --batch-size 8

# Hybrid CNN-Transformer
python scripts/train.py --model cnn_transformer --epochs 200
```

These can run sequentially (overnight) or you can run them in parallel if you have enough RAM. Each saves its own checkpoint:
- `outputs/checkpoints/CNNLSTM_best.pt`
- `outputs/checkpoints/TransformerDecoder_best.pt`
- `outputs/checkpoints/CNNTransformer_best.pt`

**Expected relative performance (based on architecture analysis):**
1. GRU Decoder — likely best (designed for this data)
2. CNN+LSTM — competitive (bidirectional helps)
3. CNN-Transformer — decent but aggressive temporal reduction (8x) may hurt
4. Transformer — may overfit on small dataset without tuning

---

## Step 6: Build a Character-Level Language Model

The LM is what turns "helo wrld" into "hello world." This has two sub-steps:

### 6a: Get training text — **YOU** need to pick a source

You need a plain-text English corpus. Options:

- **Quick & easy**: Download a Wikipedia dump extract (~100MB text)
  - https://dumps.wikimedia.org/enwiki/ → get a recent `enwiki-*-pages-articles.xml.bz2`
  - Or use a pre-extracted version from HuggingFace datasets
- **Minimal**: Just use a large book from Project Gutenberg (free, instant download)
- **Best match**: If the Willett dataset has specific sentence prompts, extract those and augment with similar text

Save the text file to: `data/lm_corpus.txt` (one sentence per line, lowercase)

### 6b: Train KenLM — **YOU** need to install KenLM

KenLM requires a C++ build. This is the trickiest dependency.

```bash
# Option A: pip (may work on Windows with build tools)
pip install kenlm

# Option B: If pip fails, use conda
conda install -c conda-forge kenlm

# Option C: If both fail, skip LM for now — the system works without it
```

If KenLM installs successfully, train the model:

```bash
# Prepare character-level training data (one char per "word", space-separated)
python -c "
import sys
for line in open('data/lm_corpus.txt'):
    chars = ' '.join(list(line.strip().lower()))
    if chars:
        print(chars)
" > data/lm_chars.txt

# Train 5-gram character LM
lmplz -o 5 < data/lm_chars.txt > outputs/lm/char_5gram.arpa

# Optional: convert to binary for faster loading
build_binary outputs/lm/char_5gram.arpa outputs/lm/char_5gram.binary
```

**If KenLM doesn't install, skip this step.** The system uses a `DummyLMScorer` fallback that returns 0 for all inputs. You'll still get greedy and beam search results — just no LM rescoring boost.

---

## Step 7: Full Evaluation & Benchmarking

Run the evaluation script on each trained model:

```bash
# Greedy decoding
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/GRUDecoder_best.pt \
  --model gru_decoder

# Beam search (width 50)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/GRUDecoder_best.pt \
  --model gru_decoder \
  --beam-width 50

# Beam search + LM (if KenLM is available)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/GRUDecoder_best.pt \
  --model gru_decoder \
  --beam-width 50 \
  --use-lm \
  --lm-path outputs/lm/char_5gram.binary

# Repeat for all 4 models...
```

Results are saved to `outputs/results/` as CSV or JSON.

For the full ablation sweep across all models and decoding methods:

```bash
python scripts/run_ablations.py
```

---

## Step 8: Run the Demo with Real Results

Once you have trained checkpoints, the demo app will automatically pick them up:

```bash
# Terminal 1: API backend
uvicorn app.api:app --reload

# Terminal 2: Streamlit frontend
streamlit run app/dashboard.py
```

The API loads checkpoints from `outputs/checkpoints/` on startup. The Streamlit app will now show real decoded text instead of random output.

**YOU can now:**
- Upload actual `.npy` trial files and see decoded text
- Compare all 4 models side-by-side
- View CTC probability heatmaps that show actual character peaks
- Explore neural embeddings that cluster by character
- Run electrode importance to see which brain regions matter most

---

## Summary: What YOU Do vs What's Automated

| Task | Who |
|------|-----|
| Download Willett dataset from Dryad | **YOU** (browser download) |
| Review quality report output | **YOU** (quick check) |
| Run `scripts/train.py` | **YOU** (one command, then wait) |
| Review training convergence | **YOU** (check CER, sample outputs) |
| Decide to continue or debug | **YOU** |
| Find/download an English text corpus | **YOU** |
| Install KenLM (or skip) | **YOU** |
| Run `scripts/evaluate.py` | **YOU** (one command) |
| Start the demo app | **YOU** (two commands) |
| Everything else (preprocessing, training loop, checkpointing, decoding, evaluation metrics, visualization) | **Automated** |

---

## Realistic Expectations

With the Willett dataset and the GRU decoder:
- **Epoch 1–10**: Loss drops, output is still mostly blanks or repeated characters
- **Epoch 10–50**: Recognizable character fragments appear, CER ~60–80%
- **Epoch 50–100**: Most common characters decoded correctly, CER ~20–40%
- **Epoch 100–200**: Convergence, CER ~5–15% depending on session quality
- **+ Beam search**: CER drops another 1–3%
- **+ LM rescoring**: CER drops another 2–5%

The Willett paper reported **~5.3% CER** with their GRU + LM setup. Our architecture matches theirs, so that's the theoretical ceiling with proper training.
