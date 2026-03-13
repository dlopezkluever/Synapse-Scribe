/# Phase 7B: Data Pipeline Overhaul

**Goal:** Fix how data is prepared, split, and fed to models. Three independent improvements: firing rate features, session-aware splitting, and trial type balancing.

**Files to modify:**
- `src/data/dataset.py`
- `src/data/loader.py`
- `scripts/train.py`
- `src/config.py`

**Key reference:** `src/features/firing_rate.py` — already implements `compute_firing_rate_features()` (bin + sqrt transform). This code is complete and tested but never wired into the training pipeline.

---

## Task 1: Wire Firing Rate Features into the Dataset

**Problem:** Models are trained on raw spike count signals. The Willett paper used binned firing rates with sqrt transform (Pathway C). The code for this exists in `src/features/firing_rate.py` but `NeuralTrialDataset.__getitem__()` never calls it — it just does `np.load(signal_path)` and optional z-score normalization.

**Fix — `src/data/dataset.py`:**

1. Add import at top: `from src.features.firing_rate import compute_firing_rate_features`

2. Add two new parameters to `NeuralTrialDataset.__init__()`:
   - `use_firing_rates: bool = False`
   - `bin_width_ms: float = 10.0`
   Store them as instance attributes.

3. In `__getitem__()`, after loading raw features (`np.load`) and BEFORE normalization, add:
   ```python
   if self.use_firing_rates:
       features = compute_firing_rate_features(features, self.bin_width_ms)
   ```
   This reduces temporal dimension by ~2.5x (at 250 Hz with 10ms bins: bin_size = 2, so T → T//2). The features are still [T', C] shaped, just shorter.

4. IMPORTANT: `actual_length` must be computed AFTER firing rate binning, not before. Move the `actual_length = min(features.shape[0], self.t_max)` line to after the firing rate step. Same for truncation.

5. In `_compute_channel_stats()`, also apply firing rate binning if `self.use_firing_rates` is True, so normalization stats match what the model sees.

**Fix — `src/data/dataset.py` `create_dataloaders()`:**

6. Add `use_firing_rates: bool = False` and `bin_width_ms: float = 10.0` parameters.
7. Pass them through to all three `NeuralTrialDataset()` constructors (train, val, test).

**Fix — `scripts/train.py`:**

8. Add CLI flag: `--use-firing-rates` (action="store_true")
9. Add CLI flag: `--bin-width-ms` (float, default=10.0)
10. Pass these to `create_dataloaders()`.

**Fix — `src/config.py`:**

11. Add `use_firing_rates: bool = False` to Config dataclass. The `bin_width_ms` field already exists (line 41).

---

## Task 2: Session-Aware Train/Val/Test Split

**Problem:** Current `split_trial_index()` in `dataset.py` shuffles randomly within each subject. Trials from the same recording session can land in both train and val sets, leaking session-specific neural patterns (electrode drift, brain state, etc.).

**Fix — `src/data/dataset.py`:**

1. The trial_index DataFrame has a `session` column (e.g., "ses-6", "ses-8", ..., "ses-16"). There are ~10 sessions for subject 1.

2. Add a new function `split_trial_index_by_session()` alongside the existing `split_trial_index()`:
   ```python
   def split_trial_index_by_session(
       trial_index: pd.DataFrame,
       split_ratios: list[float] | None = None,
       seed: int = 42,
   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   ```

3. Logic:
   - Get unique sessions, sorted (so it's deterministic)
   - Shuffle sessions with seed
   - Allocate sessions to train/val/test based on ratios (by number of trials, not number of sessions — some sessions have more trials)
   - Assign all trials from a session to the same split
   - Log which sessions went to which split

4. Do NOT remove or change the existing `split_trial_index()` — it's used by tests and other code. The new function is an alternative.

**Fix — `scripts/train.py`:**

5. Add CLI flag: `--session-split` (action="store_true")
6. When enabled, call `split_trial_index_by_session()` instead of `split_trial_index()`.
7. This means `create_dataloaders()` needs to accept an optional `split_fn` parameter, OR the split is done before calling `create_dataloaders()` and pre-split DataFrames are passed in. The simpler approach: do the split in `train.py` before creating dataloaders, and pass `split_ratios=None` to skip the internal split. Add a new `create_dataloaders_from_splits()` function that takes pre-split DataFrames instead of a single trial_index.

---

## Task 3: Trial Type Balancing

**Problem:** 74% of trials are single-letter (201 timesteps, 1 character), 26% are sentences. Models optimize for the letter pattern (predict one character) and struggle to generalize to sentences.

**Fix — `scripts/train.py`:**

The `--trial-type` flag already exists and can filter to "letters" or "sentences" only. But we also want an oversampling option.

1. Add CLI flag: `--oversample-sentences` (float, default=1.0). A value of 3.0 means sentence trials appear 3x in the training set (simple row duplication in the DataFrame before creating the dataset).

2. Implementation in train.py, after loading trial_index and before creating dataloaders:
   ```python
   if args.oversample_sentences > 1.0:
       # Identify sentence trials (n_timesteps > 300, or trial_type == "sentence")
       sentence_mask = trial_index["n_timesteps"] > 300
       sentence_df = trial_index[sentence_mask]
       n_copies = int(args.oversample_sentences) - 1
       oversampled = pd.concat([trial_index] + [sentence_df] * n_copies, ignore_index=True)
       trial_index = oversampled
   ```

3. This is applied BEFORE the train/val/test split, so only training data gets the duplicates (since val/test splits are later and smaller). Actually — better to apply AFTER the split, only to the train DataFrame. This requires using the pre-split approach from Task 2.

**Recommended approach for Tasks 2 + 3 together:**
- Do the split explicitly in `train.py` (either random or session-based)
- Apply oversampling only to the train DataFrame
- Pass pre-split DataFrames to a new `create_dataloaders_from_splits()` function

---

## Verification

- Run `pytest tests/` — existing tests should pass since all changes are additive (new flags default to off)
- Manually verify: run `python scripts/train.py --model gru_decoder --epochs 1 --use-firing-rates --normalize` and confirm it loads, bins features, and runs one epoch without errors
- Check that firing rate binning reduces the signal lengths in the logs (e.g., 201 → 80 for letters at 10ms bins with 250Hz)
