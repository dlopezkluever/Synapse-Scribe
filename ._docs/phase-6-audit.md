Phase 6 Audit — Complete Findings

  6.1 GPT-2 Re-ranking

  Core implementation: DONE. GPT2Scorer class, rescore_hypotheses(), load_lm_scorer(scorer_type="gpt2") all work correctly. 18 tests pass.

  Issues & Gaps:
  1. No lambda tuning script — Config has static gpt2_lambda=0.5 but no grid search over validation set as required
  2. No CLI integration — scripts/evaluate.py and scripts/run_ablations.py have no --scorer-type gpt2 flag. GPT-2 is only accessible via Python API, not CLI
  3. No KenLM vs GPT-2 comparison — No ablation table showing CER/WER delta between the two scorers
  4. run_ablations.py can't select GPT-2 — load_lm_scorer() is called without scorer_type parameter in ablation code

  ---
  6.2 Multi-Subject Training

  Core infrastructure: DONE. SubjectNormalization, SubjectAwareModel, SubjectAwareDataset, collate, evaluate_per_subject all work. 16 tests pass.

  Issues & Gaps:
  1. Trainer is incompatible — train_one_epoch() (line 219) and validate() (line 258) call self.model(features) with NO subject_ids. SubjectAwareModel requires subject_ids in forward(). Training would fail if attempted.  
  2. No --multi-subject CLI flag in scripts/train.py
  3. Redundant logic at multi_subject.py:188 — model(features) if subject_ids is None else model(features) — both branches are identical
  4. No cross-subject evaluation — Task requires "evaluate within-subject vs. cross-subject performance" but no leave-one-subject-out evaluation exists
  5. No results reporting — Task says "Report whether multi-subject training improves or degrades per-subject CER" — no mechanism to generate this report

  ---
  6.3 W&B Integration

  Core logging: DONE. _init_wandb, _log_wandb, _log_wandb_table, _finish_wandb integrated into Trainer. CLI flags work. 12 tests pass.

  Issues & Gaps:
  1. WER not logged — Requirement says "log loss, CER, WER, learning rate" but validate() only computes CER, not WER. No val/wer metric logged
  2. No artifact logging — Requirement says "Log training curves, confusion matrices, and embedding plots as W&B artifacts" — none of these are logged. The plotting functions exist in src/visualization/ but are never     
  called from the Trainer
  3. No sweep config — Requirement says "Add W&B sweep config for hyperparameter tuning" — no wandb_sweep.yaml or equivalent exists anywhere

  ---
  6.4 Cross-Session Generalization

  Core infrastructure: DONE. split_by_session, SessionNormalizer, SessionNormalizedDataset, evaluate_cross_session, cross_session_report all work. 16 tests pass.

  Issues & Gaps:
  1. BUG at cross_session.py:216 — batch["label_texts"] — the key is correct (confirmed: ctc_collate_fn at dataset.py:186 uses "label_texts" plural). So this is actually fine. Not a bug.
  2. No training script integration — No --train-sessions / --eval-sessions CLI flags in scripts/train.py. Users cannot run cross-session experiments from the command line
  3. No within-session vs cross-session comparison — cross_session_report() evaluates per-session CER but doesn't compute or display the within-session baseline for comparison
  4. No documentation of findings or limitations as required by the task

  ---
  6.5 Neural Manifold Analysis

  Core analysis module: DONE. All 10 functions (fit_pca, fit_umap, compute_cluster_metrics, compute_velocity_field, compute_multi_trial_dynamics, 5 plot functions) implemented. 28 tests pass.

  Issues & Gaps:
  1. Not integrated into Streamlit dashboard — Task requires "Integrate manifold analysis into the Streamlit Neural Representation Explorer as an advanced tab." The dashboard's Neural Representations page has 4 tabs but  
  none use manifold.py. PCA/t-SNE on that page is reimplemented independently
  2. No evaluation notebook documentation — Task requires "Document findings on neural manifold structure in the evaluation notebook" — not done
  3. Static, not animated — Task says "Create animated neural trajectory visualizations" but plot_neural_dynamics_3d() produces a static PNG with time-colored path. No matplotlib animation or interactive Plotly

  ---
  6.6 Real-Time Inference Prototype

  Core streaming pipeline: DONE. StreamingBuffer, StreamingDecoder (eager + stable modes), LatencyStats, simulate_streaming, terminal demo all work. LSL architecture documented. 34 tests pass.

  Issues & Gaps:
  1. 300ms latency target not met — Demo reports P95 ~1000ms+ with full GRU (192ch, 512 hidden). Test test_latency_under_300ms uses tiny model (4ch, 16 hidden) so it passes trivially but doesn't validate the actual       
  requirement. No documented mitigation strategy (smaller model, GPU, reduced window, quantization)
  2. is_stable misleading in eager mode — Line 301: is_stable = True always in eager mode, so terminal shows "stable" even though text may still be changing
  3. Minor: ANSI escape codes in realtime_demo.py may not render on Windows CMD (works in MINGW64/Git Bash)

  ---
  Summary: Cross-Cutting Themes

  ┌──────────────────────────────┬──────────┬────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │           Category           │ Severity │      Affects       │                                                               Description                                                                │
  ├──────────────────────────────┼──────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Trainer incompatibility      │ HIGH     │ 6.2, 6.4           │ Trainer doesn't pass subject_ids or support session-based splits. Multi-subject and cross-session training can't actually run end-to-end │
  ├──────────────────────────────┼──────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ No CLI/script integration    │ HIGH     │ 6.1, 6.2, 6.4      │ GPT-2 scorer, multi-subject, and cross-session modes have no CLI flags. Only usable via Python API                                       │
  ├──────────────────────────────┼──────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Missing comparison/reporting │ MEDIUM   │ 6.1, 6.2, 6.3, 6.4 │ Lambda tuning, within-vs-cross-subject CER, KenLM-vs-GPT2 delta, within-vs-cross-session CER — none produce actual comparison reports    │
  ├──────────────────────────────┼──────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Dashboard integration        │ MEDIUM   │ 6.5                │ Manifold analysis not wired into Streamlit                                                                                               │
  ├──────────────────────────────┼──────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ W&B incomplete               │ MEDIUM   │ 6.3                │ WER not logged, no artifacts, no sweep config                                                                                            │
  ├──────────────────────────────┼──────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Latency target               │ LOW      │ 6.6                │ 300ms target not met with production model; test masks this with tiny model                                                              │
  ├──────────────────────────────┼──────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Redundant logic              │ LOW      │ 6.2                │ multi_subject.py:188 identical branches                                                                                                  │
  └──────────────────────────────┴──────────┴────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Recommendations (prioritized)

  1. Fix Trainer to support subject_ids — Check for subject_ids in batch and pass to model if present. Unblocks both 6.2 and enables real multi-subject training
  2. Add CLI flags to scripts/train.py and scripts/evaluate.py for --multi-subject, --train-sessions/--eval-sessions, --scorer-type
  3. Add WER to validation loop in Trainer (simple — compute_wer already exists in metrics.py)
  4. Integrate manifold analysis into dashboard as a new tab
  5. Add W&B sweep config YAML
  6. Document latency trade-offs for 6.6 (model size vs latency, GPU recommendation)
  7. Fix multi_subject.py:188 redundant ternary
