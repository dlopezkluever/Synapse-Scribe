# Phase 7A: Training Loop & Stability Fixes

**Goal:** Fix known bugs and training instability issues. No architectural or data pipeline changes — just the trainer, scheduler, and augmentation defaults.

**Files to modify:**
- `src/training/trainer.py`
- `src/training/scheduler.py`
- `src/config.py`

---

## Task 1: Fix LR Scheduler Ordering

**File:** `src/training/trainer.py` — `train_one_epoch()` method

**Problem:** `self.scheduler.step()` (line 234) is called inside the batch loop BEFORE the first `optimizer.step()` completes. PyTorch warns about this — the first scheduled LR value gets skipped.

**Fix:** Move `self.scheduler.step()` to AFTER `self.optimizer.step()` (or after `self.scaler.step(self.optimizer)` in the AMP branch). Both the AMP and non-AMP code paths need this fix. The scheduler step should be the last thing in the batch loop, after the optimizer has stepped.

---

## Task 2: Add ReduceLROnPlateau

**File:** `src/training/trainer.py`

**Problem:** The cosine schedule keeps decaying LR on a fixed curve regardless of whether the model is improving. When CNN-LSTM found a good minimum at epoch 65, the cosine schedule kept pushing LR and the model overshot into collapse.

**Fix:** Add a `ReduceLROnPlateau` scheduler that monitors `val_cer` and reduces LR when it stagnates. This should work ALONGSIDE the existing cosine warmup, not replace it. Implementation:

1. In `__init__`, after creating `self.scheduler` (the cosine one), create a second scheduler:
   ```python
   self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
   )
   ```

2. In the `train()` method, after each epoch's validation, step the plateau scheduler with val_cer:
   ```python
   self.plateau_scheduler.step(val_cer)
   ```

3. The cosine scheduler stays per-step (in the batch loop), the plateau scheduler fires per-epoch (in the train loop). PyTorch handles multiple schedulers fine — they both modify the same optimizer param groups.

4. Log the actual LR after both schedulers have acted (already done — `self.optimizer.param_groups[0]["lr"]` reads the current value).

---

## Task 3: Boost Augmentation Defaults

**File:** `src/config.py`

**Problem:** Current augmentation is far too weak. `aug_gaussian_noise_std=0.01` adds only 1% noise relative to normalized data (std=1.0). Channel dropout at 10% is also conservative.

**Fix:** Update these defaults in the `Config` dataclass:
- `aug_gaussian_noise_std`: 0.01 → 0.1
- `aug_channel_dropout_rate`: 0.1 → 0.2

These are just default changes. The CLI still allows overriding them. No other code needs to change — the augmentation pipeline already reads these values from config.

---

## Task 4: Update Colab Notebook Hyperparameters

**File:** `notebooks/07_colab_training.ipynb`

**Problem:** The training configs in cell 12 need updated recommendations based on training results.

**Fix:** Update the `TRAINING_CONFIGS` dict:
- GRU: Lower LR to 1e-4 (was 3e-4), increase patience comment
- CNN-LSTM: Lower LR to 1e-4 (was 3e-4)
- Transformer: Increase warmup_steps comment, keep LR at 1e-4
- CNN-Transformer: Should be first in MODELS_TO_TRAIN list (highest priority, only model with downsampling)
- Reorder MODELS_TO_TRAIN to put `cnn_transformer` first

---

## Verification

After these changes:
- Run `pytest tests/` — all existing tests should pass (these are behavior-compatible changes)
- The scheduler warning should disappear in the next training run
- The plateau scheduler should prevent the collapse pattern seen in CNN-LSTM
