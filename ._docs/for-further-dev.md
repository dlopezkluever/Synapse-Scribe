# Current Status & Further Development

## Where We Are

The pipeline is complete and performing well. The CNN-LSTM decodes handwritten neural activity at **0.43% CER** — meaning out of every ~230 characters, only 1 is wrong. Sentences like *"the job of waxing linoleum frequently peeves chintzy kids"* decode perfectly. The full stack (preprocessing, training, beam search + LM, API, dashboard, Docker, CI) is production-shaped.

### What's Strong

- CNN-LSTM is near state-of-the-art for this dataset (Willett 2021 reported ~5% CER with a much larger training set)
- Pure-Python LM rescoring works without heavy dependencies
- 520 tests passing, clean CI, Docker-ready
- Full interactive demo (FastAPI + Streamlit) with 5 pages

### What's Weak

- Only 1 of 4 architectures converged
- Single-subject, single-task (handwriting only)
- Character-level vocabulary (no punctuation, numbers, capitalization)
- Offline only — no real-time streaming

---

## A. Improve Existing Models

| # | Action | Expected Impact | Effort |
|---|---|---|---|
| 1 | **Fix CNN-Transformer: reduce to 4x downsample, increase dropout to 0.3-0.5** | Could close the gap with CNN-LSTM (35% to <10% CER) | Small — config change + retrain |
| 2 | **Fix LR scheduler bug** (`trainer.py:237` — `scheduler.step()` before `optimizer.step()`) | Minor CER improvement, eliminates wasted first LR step | Trivial |
| 3 | **Train a word-level or neural LM** — GPT-2 rescoring is already implemented, just needs `pip install transformers` on Colab | Could push CER below 0.3% by fixing remaining word-level errors like `clesure` to `closure` | Small |
| 4 | **Expand vocabulary** — add punctuation, digits, capitalization (28 to ~70 classes) | Needed for any real-world use | Medium |
| 5 | **Ensemble decoding** — average CNN-LSTM + CNN-Transformer logits before beam search | Common technique for 10-20% relative CER reduction | Small (once both models converge) |

---

## B. Build Bigger Things

These use the current system as a foundation for more ambitious projects.

### 1. Real-Time Streaming Decoder

Turn the batch model into a streaming pipeline that decodes neural activity as it arrives, character by character. The `src/streaming/` module already exists in skeleton form. This is the difference between "research tool" and "assistive device."

- Sliding window inference over the CNN-LSTM
- Incremental beam search with partial hypothesis display
- WebSocket API for live dashboard updates
- **Target:** <200ms latency from neural spike to character on screen

### 2. Cross-Subject Transfer Learning

Right now the model only works for Subject 1. A real BCI needs to work for new patients with minimal calibration.

- Pre-train on Subject 1's data (which we've done)
- Fine-tune on small amounts of data from a new subject (few-shot adaptation)
- This is how clinical BCIs actually deploy — the Willett paper used subject-specific models but transfer learning is the active research frontier

### 3. Speech BCI Pipeline

The UCSF ECoG data loader is already built. Extending from handwriting to speech decoding means:

- Swap character vocabulary for phoneme vocabulary
- Add a phoneme-to-word decoder (or use CTC directly on words)
- The architecture is the same — CNN front-end + sequence model + CTC
- This is the path toward a "silent speech" interface for locked-in patients

### 4. Multi-Modal Fusion

Combine handwriting + speech neural signals for higher accuracy:

- Dual-encoder architecture (one CNN-LSTM per modality)
- Late fusion before CTC head
- The user could think a word AND imagine writing it, and the system uses both signals

### 5. Edge Deployment / Embedded Inference

Shrink the model for on-device inference (implanted BCI can't depend on cloud):

- Quantize CNN-LSTM to INT8 (PyTorch quantization — model goes from 129MB to ~32MB)
- Export to ONNX for hardware-agnostic deployment
- Profile latency on Raspberry Pi / Jetson Nano as proxy for implantable processor
- **Target:** real-time inference under 50ms per frame on embedded hardware

### 6. Closed-Loop BCI Simulator

Build a full simulation of a closed-loop system where the decoded text feeds back to the user:

- Simulated neural input, decoder, text output, error correction UI
- Add a "correction" mode where the user can neurally select between beam search alternatives
- This demonstrates the full product concept, not just the ML model

### 7. Clinical Dashboard / Study Tool

Extend the existing Streamlit dashboard into a tool researchers could actually use in a clinical setting:

- Session-over-session performance tracking
- Electrode health monitoring (using the existing diagnostics modules)
- Adaptive model retraining when performance degrades
- Patient-facing simplified UI vs researcher-facing analytics UI

---

## Recommended Priority Order

If the goal is a portfolio/hackathon project that shows ambition:

1. **Fix CNN-Transformer** — quick win, shows you can debug and iterate
2. **Real-time streaming** — most impressive demo, biggest "wow factor"
3. **GPT-2 LM rescoring** — easy to add, meaningful improvement
4. **Cross-subject transfer** — shows ML depth, active research area
5. **Speech BCI extension** — shows the system generalizes beyond one task
