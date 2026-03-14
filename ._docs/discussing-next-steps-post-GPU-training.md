Claude's analysis of the GPU training results:

  Training Summary (T4 GPU, Colab, 2026-03-13)

  ┌─────────────────┬────────┬──────────────────┬────────┬─────────────┬───────────┬───────────┐     
  │      Model      │ Params │      Epochs      │  Time  │  Best Val   │ Best Val  │  Status   │     
  │                 │        │                  │        │     CER     │    WER    │           │     
  ├─────────────────┼────────┼──────────────────┼────────┼─────────────┼───────────┼───────────┤     
  │ CNN-LSTM        │ 10.7M  │ 173 (early stop  │ 111    │ 0.0424      │ ~0.14     │ Excellent │     
  │                 │        │ @ 153)           │ min    │ (4.2%)      │           │           │     
  ├─────────────────┼────────┼──────────────────┼────────┼─────────────┼───────────┼───────────┤     
  │ CNN-Transformer │ 13.3M  │ 89 (early stop @ │ 37.7   │ 0.3516      │ ~0.53     │ Moderate  │     
  │                 │        │  69)             │ min    │ (35.2%)     │           │           │     
  ├─────────────────┼────────┼──────────────────┼────────┼─────────────┼───────────┼───────────┤     
  │ GRU Decoder     │ 4.9M   │ 28 (early stop @ │ 14.2   │ 0.9817      │ ~0.98     │ Failed    │     
  │                 │        │  8)              │ min    │ (98.2%)     │           │           │     
  ├─────────────────┼────────┼──────────────────┼────────┼─────────────┼───────────┼───────────┤     
  │ Transformer     │ 20.7M  │ 21 (early stop @ │ 12.3   │ 1.0000      │ 1.00      │ Failed    │     
  │                 │        │  1)              │ min    │ (100%)      │           │           │     
  └─────────────────┴────────┴──────────────────┴────────┴─────────────┴───────────┴───────────┘     

  Key Takeaways

  CNN-LSTM is the clear winner. 4.2% CER is a genuinely strong result -- it's decoding full sentences   like "daisy leaned forward at once horrified and fascina" nearly perfectly by epoch ~107. The     
  remaining errors are minor character substitutions ("forwarrd", "fascin" truncations). WER around  
  14% is very reasonable for greedy CTC decoding; beam search + LM would push this even lower.       

  CNN-Transformer shows promise but plateaued. It got down to 35% CER and was learning
  character-level patterns (single-letter refs decoded correctly), but it stalled. The 8x temporal   
  downsample (vs 4x for CNN-LSTM) may be too aggressive -- it's compressing the time axis more, which   likely hurts CTC alignment for longer sequences. The oscillating val_loss (bouncing between       
  0.9-2.1 while train_loss was ~0.05) signals overfitting with val generalization issues.

  GRU and pure Transformer completely failed. Both never got past predicting single characters for   
  full sentences. The GRU (with --no-downsample not used, t-max=5000) was dealing with very long     
  sequences through a recurrent-only architecture -- the CTC loss never collapsed. The Transformer   
  (20.7M params, 6 layers) never emitted a single character in 21 epochs; its train loss was still at   1.34 when it stopped. With no CNN front-end to provide local feature extraction, the raw attention   over long neural time series just doesn't work at this data scale.

  Observations & Suggestions

  1. LR scheduler warning: All models logged lr_scheduler.step() before optimizer.step() -- this     
  means the first LR value from the cosine schedule was skipped. Worth fixing in the trainer (swap   
  the order at line 237).
  2. CNN-LSTM's "hmmmm" spike didn't happen, but CNN-Transformer had one at epoch 79
  (pred='hmmmmmmmmmmmm', CER spiked to 1.57) -- a classic CTC repetition collapse. It recovered, but 
  this suggests the model is near an instability boundary.
  3. CNN-Transformer's 8x downsample is the likely bottleneck. The CNN-LSTM uses 4x, keeping more    
  temporal resolution for CTC. Running CNN-Transformer with --no-downsample or reducing to 4x could  
  significantly improve it.
  4. GRU needs a CNN front-end or much smaller t-max. Pure recurrent over 5000-step sequences with   
  4.9M params is just not enough capacity/structure. It never broke through the "predict one char"   
  phase.
  5. Transformer needs warmup or different architecture. 6 layers of self-attention on raw
  192-channel neural signals is too much for 2,952 training samples. It needs either a CNN embedding 
  (like the CNN-Transformer) or significantly more data.

  The CNN-LSTM checkpoint at 4.2% CER is your production model. Would you like to run evaluation with   beam search + LM on the test set, or address any of the issues above?

✻ Crunched for 1m 14s

❯ what is beam search

● Beam search is an alternative to greedy decoding for CTC models.

  Greedy decoding (what you used in training) picks the single highest-probability character at each 
  timestep, then collapses repeats and removes blanks. It's fast but can miss better overall
  sequences.

  Beam search keeps the top-K most probable candidate sequences (the "beam width") at each timestep, 
  expanding and pruning as it goes. This explores more of the search space and often finds sequences 
  with higher total probability that greedy misses.

  With a language model (LM), beam search gets even better -- it scores candidates not just by the   
  CTC acoustic probability but also by how likely the text is in English. So if the model is torn    
  between "horrified" and "horrrified", the LM pushes it toward the real word.

  In your project, this is already implemented:
  - src/decoding/beam_search.py -- beam search decoder
  - KenLM integration for LM rescoring

  Given your CNN-LSTM's 4.2% CER with greedy decoding, beam search + LM could realistically push that   down to ~2-3% CER and significantly improve WER (from ~14% to potentially single digits), since   
  many of the remaining errors are minor misspellings that a language model would correct.

❯ okay will  this require a whole nother round of testing?
  ⎿  ⧉ Selected 1 lines from ._docs\GPU-Training-Summary-313.md in Cursor



● No. Beam search is just a decoding strategy applied at inference time -- it doesn't change the     
  model weights at all. You use the same trained checkpoint (CNNLSTM_best.pt) and just swap out how  
  you decode the CTC output.

  You'd run something like:

  python scripts/evaluate.py --model cnn_lstm --checkpoint
  outputs/checkpoints/GPU-3-13/CNNLSTM_best.pt --beam-width 10 --lm-path <path-to-kenlm-model>       

  It takes seconds/minutes to run on the test set, not hours of GPU training. The model is already   
  trained -- beam search just does a smarter job reading its output probabilities.