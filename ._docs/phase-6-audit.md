
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
