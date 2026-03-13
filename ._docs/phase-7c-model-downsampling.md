# Phase 7C: Model Temporal Downsampling

**Goal:** Add temporal downsampling to GRU, CNN-LSTM, and Transformer models so CTC doesn't have to learn 99%+ blank ratios. CNN-Transformer already has 8x downsampling via MaxPool and needs no changes.

**Files to modify:**
- `src/models/gru_decoder.py`
- `src/models/cnn_lstm.py`
- `src/models/transformer.py`
- `src/training/trainer.py` (input_length adjustment)

**Design principle:** Each model should expose a `downsample_factor` property (int) so the trainer can automatically adjust CTC `input_lengths`. CNN-Transformer already downsamples 8x — use the same pattern.

---

## Task 1: Add Downsampling to GRU Decoder

**File:** `src/models/gru_decoder.py`

**Current architecture:** Linear(192 → hidden) → GRU → Linear(hidden → 28). No temporal reduction — output is [B, T, 28].

**Fix:** Add a small CNN front-end before the GRU that reduces temporal resolution. This is a common pattern (Conv-GRU) and adds negligible parameters.

1. Add a `TemporalDownsampler` module (or inline it) before the GRU:
   ```python
   self.downsample = nn.Sequential(
       nn.Conv1d(in_channels, 256, kernel_size=5, stride=2, padding=2),
       nn.GELU(),
       nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
       nn.GELU(),
   )
   ```
   This gives 4x downsampling (stride=2 twice). A 5000-timestep input becomes 1250.

2. The Conv1d expects [B, C, T] but features come in as [B, T, C]. Transpose before conv, transpose back after:
   ```python
   x = x.transpose(1, 2)       # [B, T, C] → [B, C, T]
   x = self.downsample(x)      # [B, 256, T//4]
   x = x.transpose(1, 2)       # [B, T//4, 256]
   ```

3. Feed into existing GRU (adjust input size from `in_channels` to 256).

4. Add property:
   ```python
   @property
   def downsample_factor(self) -> int:
       return 4
   ```

5. Keep backward compatibility: add a `use_downsample: bool = True` constructor parameter. When False, use the original Linear projection (no downsampling, `downsample_factor = 1`). Default to True.

---

## Task 2: Add Downsampling to CNN-LSTM

**File:** `src/models/cnn_lstm.py`

**Current architecture:** 3 Conv1d layers with `padding=kernel_size//2` (same padding, no stride) → BiLSTM → Linear. Output is [B, T, 28] — no temporal reduction.

**Fix:** Change the existing conv layers to use stride instead of same-padding. This is minimally invasive — just change padding and add stride.

1. For the existing 3 Conv1d blocks, change:
   - Layer 1: stride=2 (instead of 1), adjust padding for the kernel
   - Layer 2: stride=2
   - Layer 3: stride=1 (keep this one as-is for fine-grained features)
   Total downsampling: 4x.

2. When modifying, keep the kernel_size=7 and adjust padding. With stride=2, use `padding=3` (kernel_size//2) which gives `T_out = ceil(T_in / 2)`.

3. Add property:
   ```python
   @property
   def downsample_factor(self) -> int:
       return 4
   ```

4. Keep backward compatibility: add `use_downsample: bool = True` constructor parameter. When False, use original same-padding (stride=1). Default to True.

---

## Task 3: Add Downsampling to Transformer

**File:** `src/models/transformer.py`

**Current architecture:** LinearProjection(192 → d_model) → PositionalEncoding → N Transformer layers → Linear(d_model → 28). No temporal reduction.

**Fix:** Add a CNN front-end (like CNN-Transformer uses) before the transformer layers. This is better than pooling after attention since it reduces the sequence length BEFORE the expensive O(T^2) attention.

1. Replace the simple `LinearProjection` with a `ConvEmbedding`:
   ```python
   self.conv_embed = nn.Sequential(
       nn.Conv1d(in_channels, d_model, kernel_size=5, stride=2, padding=2),
       nn.GELU(),
       nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
       nn.GELU(),
   )
   ```
   This gives 4x downsampling and also makes positional encoding more tractable (max_seq_len=4096 → effective 1024).

2. Handle transpose (same as GRU task).

3. Feed into existing positional encoding and transformer layers.

4. IMPORTANT: The positional encoding `max_seq_len` should be based on the DOWNSAMPLED length. If input is 4096 and downsample is 4x, max_seq_len=1024 is sufficient. Update the PE buffer size accordingly, or compute it as `max_seq_len // downsample_factor`.

5. Add property:
   ```python
   @property
   def downsample_factor(self) -> int:
       return 4
   ```

6. Keep backward compatibility with `use_downsample: bool = True` parameter.

---

## Task 4: Adjust CTC input_lengths in Trainer

**File:** `src/training/trainer.py`

**Problem:** The CTC loss receives `input_lengths` from the dataset, which are the raw signal lengths. If the model downsamples by 4x, the CTC input_lengths must be divided by the downsample factor.

**Fix:**

1. In `__init__`, detect the model's downsample factor:
   ```python
   self.downsample_factor = getattr(model, "downsample_factor", 1)
   ```

2. In `train_one_epoch()` and `validate()`, adjust input_lengths before passing to CTC loss:
   ```python
   input_lengths = batch["input_lengths"].to(self.device)
   # Adjust for model's temporal downsampling
   if self.downsample_factor > 1:
       input_lengths = input_lengths // self.downsample_factor
   ```

3. This MUST happen in both `train_one_epoch()` and `validate()` — anywhere `input_lengths` is used for CTC.

4. Log the downsample factor at training start:
   ```python
   logger.info("Temporal downsample factor: %d", self.downsample_factor)
   ```

---

## Task 5: Update CNN-Transformer downsample_factor

**File:** `src/models/cnn_transformer.py`

CNN-Transformer already has 3x MaxPool(2) = 8x downsampling, but likely doesn't expose a `downsample_factor` property yet.

1. Add the property (same pattern as other models):
   ```python
   @property
   def downsample_factor(self) -> int:
       return 8  # 3 MaxPool(2) layers
   ```

2. Verify: make sure the trainer's new input_length adjustment works with the existing CNN-Transformer architecture.

---

## Task 6: Update scripts/train.py Model Construction

**File:** `scripts/train.py`

The MODEL_REGISTRY in train.py constructs models. If the new `use_downsample` parameter defaults to True, no changes needed. But if we want a CLI override:

1. Add optional CLI flag: `--no-downsample` (action="store_true") to force `use_downsample=False` for ablation comparisons.
2. Pass to model constructor.

---

## Verification

- Run `pytest tests/` — tests that construct models may need updating if constructor signatures changed. Any test that creates a GRU/CNN-LSTM/Transformer directly will need `use_downsample=False` if the test was checking exact output shapes.
- Verify each model: create with dummy input [1, 1000, 192], confirm output shape is [1, 250, 28] (for 4x downsample) or [1, 125, 28] (for 8x downsample).
- Run one epoch on Colab with the updated models and confirm CTC loss doesn't error out (input_lengths must be >= target_lengths after adjustment).
- Edge case: after downsampling, if `input_length // 4 < target_length`, CTC will error. This can happen for very short sequences with long labels. The `CTCLossWrapper` already clamps, but log a warning if this occurs.
