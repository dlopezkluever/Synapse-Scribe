Training Summary (T4 GPU, Colab, 2026-03-13)

┌─────────────────┬────────┬────────────────────────┬──────────┬────────────────┬──────────────┬───────────┐
│      Model      │ Params │         Epochs         │   Time   │  Best Val CER  │ Best Val WER │  Status   │
├─────────────────┼────────┼────────────────────────┼──────────┼────────────────┼──────────────┼───────────┤
│ CNN-LSTM        │ 10.7M  │ 173 (early stop @ 153) │ 111 min  │ 0.0424 (4.2%)  │ ~0.14        │ Excellent │
├─────────────────┼────────┼────────────────────────┼──────────┼────────────────┼──────────────┼───────────┤
│ CNN-Transformer │ 13.3M  │ 89 (early stop @ 69)   │ 37.7 min │ 0.3516 (35.2%) │ ~0.53        │ Moderate  │
├─────────────────┼────────┼────────────────────────┼──────────┼────────────────┼──────────────┼───────────┤
│ GRU Decoder     │ 4.9M   │ 28 (early stop @ 8)    │ 14.2 min │ 0.9817 (98.2%) │ ~0.98        │ Failed    │
├─────────────────┼────────┼────────────────────────┼──────────┼────────────────┼──────────────┼───────────┤
│ Transformer     │ 20.7M  │ 21 (early stop @ 1)    │ 12.3 min │ 1.0000 (100%)  │ 1.00         │ Failed    │
└─────────────────┴────────┴────────────────────────┴──────────┴────────────────┴──────────────┴───────────┘

Key Takeaways

CNN-LSTM is the clear winner. 4.2% CER is a genuinely strong result -- it's decoding full sentences like "daisy leaned       
forward at once horrified and fascina" nearly perfectly by epoch ~107. The remaining errors are minor character substitutions   ("forwarrd", "fascin" truncations). WER around 14% is very reasonable for greedy CTC decoding; beam search + LM would push  
this even lower.

CNN-Transformer shows promise but plateaued. It got down to 35% CER and was learning character-level patterns (single-letter 
refs decoded correctly), but it stalled. The 8x temporal downsample (vs 4x for CNN-LSTM) may be too aggressive -- it's       
compressing the time axis more, which likely hurts CTC alignment for longer sequences. The oscillating val_loss (bouncing    
between 0.9-2.1 while train_loss was ~0.05) signals overfitting with val generalization issues.

GRU and pure Transformer completely failed. Both never got past predicting single characters for full sentences. The GRU     
(with --no-downsample not used, t-max=5000) was dealing with very long sequences through a recurrent-only architecture -- the   CTC loss never collapsed. The Transformer (20.7M params, 6 layers) never emitted a single character in 21 epochs; its train 
loss was still at 1.34 when it stopped. With no CNN front-end to provide local feature extraction, the raw attention over    
long neural time series just doesn't work at this data scale.

Observations & Suggestions

1. LR scheduler warning: All models logged lr_scheduler.step() before optimizer.step() -- this means the first LR value from 
the cosine schedule was skipped. Worth fixing in the trainer (swap the order at line 237).
2. CNN-LSTM's "hmmmm" spike didn't happen, but CNN-Transformer had one at epoch 79 (pred='hmmmmmmmmmmmm', CER spiked to 1.57)   -- a classic CTC repetition collapse. It recovered, but this suggests the model is near an instability boundary.
3. CNN-Transformer's 8x downsample is the likely bottleneck. The CNN-LSTM uses 4x, keeping more temporal resolution for CTC. 
Running CNN-Transformer with --no-downsample or reducing to 4x could significantly improve it.
4. GRU needs a CNN front-end or much smaller t-max. Pure recurrent over 5000-step sequences with 4.9M params is just not     
enough capacity/structure. It never broke through the "predict one char" phase.
5. Transformer needs warmup or different architecture. 6 layers of self-attention on raw 192-channel neural signals is too   
much for 2,952 training samples. It needs either a CNN embedding (like the CNN-Transformer) or significantly more data.

The CNN-LSTM checkpoint at 4.2% CER is your production model. Would you like to run evaluation with beam search + LM on the  
test set, or address any of the issues above?

======================================================================
TRAINING: cnn_transformer
======================================================================
Command: python scripts/train.py --model cnn_transformer --epochs 200 --batch-size 8 --lr 0.0001 --t-max 4096 --normalize --filter-by-length --wandb --wandb-project brain-text-decoder --wandb-run-name cnn_transformer_gpu_v1 --wandb-tags hybrid cnn-transformer gpu full-size --n-layers 4 --dropout 0.1

08:44:36 __main__ INFO: Loading dataset...
08:44:36 src.data.loader INFO: Loading cached trial index from data/willett_handwriting/trial_index.csv
08:44:37 __main__ INFO: Loaded 4126 trials
08:44:37 __main__ INFO: Length filter (t_max=4096): 4126 -> 3691 trials
08:44:37 __main__ INFO: Signal lengths: min=201, max=4094, mean=522, median=201
08:44:37 src.data.dataset INFO: Split: 2952 train, 369 val, 370 test
08:44:37 src.data.dataset INFO: Computed channel stats from 200 trials: mean=0.2085, std=0.4468
08:44:37 __main__ INFO: Model: cnn_transformer (13298460 params)
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /root/.netrc.
wandb: Currently logged in as: dlopezkluever (dlopezkluever-aiuteur) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
wandb: setting up run a2kqln33
wandb: Tracking run with wandb version 0.25.0
wandb: Run data is saved locally in /content/BCI-2/wandb/run-20260313_084444-a2kqln33
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run cnn_transformer_gpu_v1
wandb: ⭐️ View project at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: 🚀 View run at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/a2kqln33
08:44:46 src.training.trainer INFO: W&B run initialized: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/a2kqln33
08:44:46 src.training.trainer INFO: Starting training: model=CNNTransformer, device=cuda, epochs=200, amp=True
08:44:46 src.training.trainer INFO: Parameters: 13298460
08:44:46 src.training.trainer INFO: Temporal downsample factor: 8
/content/BCI-2/src/training/trainer.py:237: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  self.scheduler.step()
08:45:16 src.training.trainer INFO: Epoch 1/200 — train_loss=8.9181, val_loss=5.2229, val_cer=1.0000, val_wer=1.0000, lr=7.38e-05, time=29.9s
08:45:16 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:45:16 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:45:16 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:45:18 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=1.0000)
08:45:43 src.training.trainer INFO: Epoch 2/200 — train_loss=3.6845, val_loss=4.9522, val_cer=1.0000, val_wer=1.0000, lr=1.00e-04, time=24.9s
08:45:43 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:45:43 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:45:43 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:46:07 src.training.trainer INFO: Epoch 3/200 — train_loss=2.4809, val_loss=3.3050, val_cer=1.0000, val_wer=1.0000, lr=1.00e-04, time=23.9s
08:46:07 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:46:07 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:46:07 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:46:32 src.training.trainer INFO: Epoch 4/200 — train_loss=2.0221, val_loss=3.1996, val_cer=1.0000, val_wer=1.0000, lr=1.00e-04, time=24.8s
08:46:32 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:46:32 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:46:32 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:46:57 src.training.trainer INFO: Epoch 5/200 — train_loss=1.8296, val_loss=2.5178, val_cer=1.0000, val_wer=1.0000, lr=9.99e-05, time=24.6s
08:46:57 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:46:57 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:46:57 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:47:21 src.training.trainer INFO: Epoch 6/200 — train_loss=1.6731, val_loss=2.5582, val_cer=1.0000, val_wer=1.0000, lr=9.99e-05, time=23.8s
08:47:21 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:47:21 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:47:21 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:47:47 src.training.trainer INFO: Epoch 7/200 — train_loss=1.5860, val_loss=2.2702, val_cer=1.0000, val_wer=1.0000, lr=4.99e-05, time=25.4s
08:47:47 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:47:47 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:47:47 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:48:12 src.training.trainer INFO: Epoch 8/200 — train_loss=1.5204, val_loss=2.1380, val_cer=0.9938, val_wer=0.9811, lr=9.97e-05, time=25.1s
08:48:12 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:48:12 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:48:12 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:48:13 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.9938)
08:48:37 src.training.trainer INFO: Epoch 9/200 — train_loss=0.9553, val_loss=2.1045, val_cer=0.9370, val_wer=0.8142, lr=9.96e-05, time=24.1s
08:48:37 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:48:37 src.training.trainer INFO:   [1] pred=''                             ref='x'
08:48:37 src.training.trainer INFO:   [2] pred=''                             ref='r'
08:48:38 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.9370)
08:49:02 src.training.trainer INFO: Epoch 10/200 — train_loss=0.7025, val_loss=2.0469, val_cer=0.8849, val_wer=0.6992, lr=9.95e-05, time=23.8s
08:49:02 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:49:02 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:49:02 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:49:02 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.8849)
08:49:27 src.training.trainer INFO: Epoch 11/200 — train_loss=0.6728, val_loss=1.8454, val_cer=0.8865, val_wer=0.7134, lr=9.94e-05, time=24.9s
08:49:27 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:49:27 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:49:27 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:49:52 src.training.trainer INFO: Epoch 12/200 — train_loss=0.6651, val_loss=1.4471, val_cer=0.8661, val_wer=0.6567, lr=9.93e-05, time=24.7s
08:49:52 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:49:52 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:49:52 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:49:52 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.8661)
08:50:16 src.training.trainer INFO: Epoch 13/200 — train_loss=0.6127, val_loss=1.4938, val_cer=0.8823, val_wer=0.6929, lr=9.92e-05, time=23.9s
08:50:16 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:50:16 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:50:16 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:50:41 src.training.trainer INFO: Epoch 14/200 — train_loss=0.5870, val_loss=1.5716, val_cer=0.8750, val_wer=0.6772, lr=9.90e-05, time=24.6s
08:50:41 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:50:41 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:50:41 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:51:07 src.training.trainer INFO: Epoch 15/200 — train_loss=0.5820, val_loss=1.6202, val_cer=0.8745, val_wer=0.6819, lr=9.88e-05, time=25.0s
08:51:07 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:51:07 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:51:07 src.training.trainer INFO:   [2] pred='i'                            ref='r'
08:51:31 src.training.trainer INFO: Epoch 16/200 — train_loss=0.5625, val_loss=1.3784, val_cer=0.8599, val_wer=0.6252, lr=9.87e-05, time=24.1s
08:51:31 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:51:31 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:51:31 src.training.trainer INFO:   [2] pred='i'                            ref='r'
08:51:32 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.8599)
08:51:56 src.training.trainer INFO: Epoch 17/200 — train_loss=0.5567, val_loss=1.7204, val_cer=0.8630, val_wer=0.6535, lr=9.85e-05, time=23.9s
08:51:56 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:51:56 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:51:56 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:52:21 src.training.trainer INFO: Epoch 18/200 — train_loss=0.5555, val_loss=1.6392, val_cer=0.8630, val_wer=0.6425, lr=9.83e-05, time=24.9s
08:52:21 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:52:21 src.training.trainer INFO:   [1] pred='a'                            ref='x'
08:52:21 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:52:46 src.training.trainer INFO: Epoch 19/200 — train_loss=0.5281, val_loss=1.6713, val_cer=0.8599, val_wer=0.6346, lr=9.81e-05, time=24.7s
08:52:46 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:52:46 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:52:46 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:53:10 src.training.trainer INFO: Epoch 20/200 — train_loss=0.5572, val_loss=1.4992, val_cer=0.8521, val_wer=0.6173, lr=9.78e-05, time=23.7s
08:53:10 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:53:10 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:53:10 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:53:11 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.8521)
08:53:35 src.training.trainer INFO: Epoch 21/200 — train_loss=0.5139, val_loss=1.4752, val_cer=0.8500, val_wer=0.6205, lr=9.76e-05, time=24.8s
08:53:35 src.training.trainer INFO:   [0] pred=''                             ref='h'
08:53:35 src.training.trainer INFO:   [1] pred='i'                            ref='x'
08:53:35 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:53:36 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.8500)
08:54:01 src.training.trainer INFO: Epoch 22/200 — train_loss=0.5120, val_loss=1.6650, val_cer=0.8625, val_wer=0.6362, lr=9.74e-05, time=24.9s
08:54:01 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:54:01 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:54:01 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:54:26 src.training.trainer INFO: Epoch 23/200 — train_loss=0.5227, val_loss=1.5035, val_cer=0.8526, val_wer=0.6142, lr=9.71e-05, time=24.2s
08:54:26 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:54:26 src.training.trainer INFO:   [1] pred='th'                           ref='x'
08:54:26 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:54:50 src.training.trainer INFO: Epoch 24/200 — train_loss=0.5608, val_loss=1.7748, val_cer=0.8609, val_wer=0.6220, lr=9.68e-05, time=24.0s
08:54:50 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:54:50 src.training.trainer INFO:   [1] pred='thx'                          ref='x'
08:54:50 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:55:15 src.training.trainer INFO: Epoch 25/200 — train_loss=0.5104, val_loss=2.1144, val_cer=0.8745, val_wer=0.6709, lr=9.65e-05, time=25.1s
08:55:15 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:55:15 src.training.trainer INFO:   [1] pred='thx'                          ref='x'
08:55:15 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:55:40 src.training.trainer INFO: Epoch 26/200 — train_loss=0.5035, val_loss=1.2683, val_cer=0.8438, val_wer=0.6000, lr=9.62e-05, time=24.8s
08:55:40 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:55:40 src.training.trainer INFO:   [1] pred='hx'                           ref='x'
08:55:40 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:55:41 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.8438)
08:56:05 src.training.trainer INFO: Epoch 27/200 — train_loss=0.4596, val_loss=1.8035, val_cer=0.8448, val_wer=0.6567, lr=9.59e-05, time=23.9s
08:56:05 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:56:05 src.training.trainer INFO:   [1] pred='xhe'                          ref='x'
08:56:05 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:56:30 src.training.trainer INFO: Epoch 28/200 — train_loss=0.4630, val_loss=1.3008, val_cer=0.7734, val_wer=0.6157, lr=9.56e-05, time=24.8s
08:56:30 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:56:30 src.training.trainer INFO:   [1] pred='x'                            ref='x'
08:56:30 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:56:31 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.7734)
08:56:56 src.training.trainer INFO: Epoch 29/200 — train_loss=0.4276, val_loss=1.7135, val_cer=0.7026, val_wer=0.6378, lr=9.53e-05, time=24.9s
08:56:56 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:56:56 src.training.trainer INFO:   [1] pred='tx'                           ref='x'
08:56:56 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:56:56 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.7026)
08:57:21 src.training.trainer INFO: Epoch 30/200 — train_loss=0.4714, val_loss=1.5787, val_cer=0.8021, val_wer=0.6567, lr=9.50e-05, time=24.7s
08:57:21 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:57:21 src.training.trainer INFO:   [1] pred='th'                           ref='x'
08:57:21 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:57:45 src.training.trainer INFO: Epoch 31/200 — train_loss=0.4093, val_loss=1.5348, val_cer=0.6974, val_wer=0.6299, lr=9.46e-05, time=23.6s
08:57:45 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:57:45 src.training.trainer INFO:   [1] pred='i'                            ref='x'
08:57:45 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:57:45 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.6974)
08:58:11 src.training.trainer INFO: Epoch 32/200 — train_loss=0.3700, val_loss=1.1252, val_cer=0.5995, val_wer=0.6000, lr=9.42e-05, time=25.1s
08:58:11 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:58:11 src.training.trainer INFO:   [1] pred='x'                            ref='x'
08:58:11 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:58:11 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.5995)
08:58:36 src.training.trainer INFO: Epoch 33/200 — train_loss=0.3390, val_loss=1.0402, val_cer=0.7365, val_wer=0.5969, lr=9.39e-05, time=24.9s
08:58:36 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:58:36 src.training.trainer INFO:   [1] pred='t'                            ref='x'
08:58:36 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:59:00 src.training.trainer INFO: Epoch 34/200 — train_loss=0.3642, val_loss=1.2345, val_cer=0.6568, val_wer=0.6110, lr=9.35e-05, time=23.8s
08:59:00 src.training.trainer INFO:   [0] pred='r'                            ref='h'
08:59:00 src.training.trainer INFO:   [1] pred='x'                            ref='x'
08:59:00 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:59:25 src.training.trainer INFO: Epoch 35/200 — train_loss=0.3499, val_loss=1.7967, val_cer=0.6344, val_wer=0.6362, lr=9.31e-05, time=24.3s
08:59:25 src.training.trainer INFO:   [0] pred='h'                            ref='h'
08:59:25 src.training.trainer INFO:   [1] pred='x'                            ref='x'
08:59:25 src.training.trainer INFO:   [2] pred='r'                            ref='r'
08:59:50 src.training.trainer INFO: Epoch 36/200 — train_loss=0.3363, val_loss=1.4849, val_cer=0.6375, val_wer=0.6299, lr=9.27e-05, time=24.9s
08:59:50 src.training.trainer INFO:   [0] pred='hr'                           ref='h'
08:59:50 src.training.trainer INFO:   [1] pred='x'                            ref='x'
08:59:50 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:00:15 src.training.trainer INFO: Epoch 37/200 — train_loss=0.3453, val_loss=1.1436, val_cer=0.6010, val_wer=0.6016, lr=9.23e-05, time=24.6s
09:00:15 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:00:15 src.training.trainer INFO:   [1] pred='xx'                           ref='x'
09:00:15 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:00:39 src.training.trainer INFO: Epoch 38/200 — train_loss=0.3126, val_loss=1.4121, val_cer=0.5906, val_wer=0.6299, lr=9.18e-05, time=23.6s
09:00:39 src.training.trainer INFO:   [0] pred='hr'                           ref='h'
09:00:39 src.training.trainer INFO:   [1] pred='h'                            ref='x'
09:00:39 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:00:40 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.5906)
09:01:05 src.training.trainer INFO: Epoch 39/200 — train_loss=0.2965, val_loss=0.9121, val_cer=0.5208, val_wer=0.5764, lr=9.14e-05, time=25.0s
09:01:05 src.training.trainer INFO:   [0] pred='r'                            ref='h'
09:01:05 src.training.trainer INFO:   [1] pred='tx'                           ref='x'
09:01:05 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:01:05 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.5208)
09:01:30 src.training.trainer INFO: Epoch 40/200 — train_loss=0.2964, val_loss=1.2339, val_cer=0.5000, val_wer=0.5843, lr=9.09e-05, time=25.1s
09:01:30 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:01:30 src.training.trainer INFO:   [1] pred='tf'                           ref='x'
09:01:30 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:01:31 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.5000)
09:01:55 src.training.trainer INFO: Epoch 41/200 — train_loss=0.2494, val_loss=0.9832, val_cer=0.4833, val_wer=0.5606, lr=9.05e-05, time=24.2s
09:01:55 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:01:55 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:01:55 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:01:56 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.4833)
09:02:20 src.training.trainer INFO: Epoch 42/200 — train_loss=0.2465, val_loss=1.0480, val_cer=0.4714, val_wer=0.5701, lr=9.00e-05, time=24.4s
09:02:20 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:02:20 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:02:20 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:02:21 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.4714)
09:02:46 src.training.trainer INFO: Epoch 43/200 — train_loss=0.1909, val_loss=1.1826, val_cer=0.4604, val_wer=0.5606, lr=8.95e-05, time=25.4s
09:02:46 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:02:46 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:02:46 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:02:47 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.4604)
09:03:12 src.training.trainer INFO: Epoch 44/200 — train_loss=0.2161, val_loss=1.6325, val_cer=0.4760, val_wer=0.5937, lr=8.91e-05, time=25.3s
09:03:12 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:03:12 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:03:12 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:03:37 src.training.trainer INFO: Epoch 45/200 — train_loss=0.2964, val_loss=0.9161, val_cer=0.4766, val_wer=0.5622, lr=8.86e-05, time=24.5s
09:03:37 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:03:37 src.training.trainer INFO:   [1] pred='xx'                           ref='x'
09:03:37 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:04:02 src.training.trainer INFO: Epoch 46/200 — train_loss=0.2238, val_loss=1.5021, val_cer=0.4760, val_wer=0.5906, lr=8.80e-05, time=24.6s
09:04:02 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:04:02 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:04:02 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:04:27 src.training.trainer INFO: Epoch 47/200 — train_loss=0.2064, val_loss=0.9702, val_cer=0.4688, val_wer=0.6236, lr=8.75e-05, time=25.1s
09:04:27 src.training.trainer INFO:   [0] pred='hx'                           ref='h'
09:04:27 src.training.trainer INFO:   [1] pred='xx'                           ref='x'
09:04:27 src.training.trainer INFO:   [2] pred='rxx'                          ref='r'
09:04:53 src.training.trainer INFO: Epoch 48/200 — train_loss=0.2132, val_loss=1.2325, val_cer=0.4219, val_wer=0.5559, lr=8.70e-05, time=25.2s
09:04:53 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:04:53 src.training.trainer INFO:   [1] pred='xx'                           ref='x'
09:04:53 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:04:54 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.4219)
09:05:18 src.training.trainer INFO: Epoch 49/200 — train_loss=0.1778, val_loss=0.9969, val_cer=0.4510, val_wer=0.5921, lr=8.65e-05, time=24.7s
09:05:18 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:05:18 src.training.trainer INFO:   [1] pred='xx'                           ref='x'
09:05:18 src.training.trainer INFO:   [2] pred='rx'                           ref='r'
09:05:43 src.training.trainer INFO: Epoch 50/200 — train_loss=0.1501, val_loss=0.9025, val_cer=0.4859, val_wer=0.6457, lr=8.59e-05, time=24.4s
09:05:43 src.training.trainer INFO:   [0] pred='hxx'                          ref='h'
09:05:43 src.training.trainer INFO:   [1] pred='xxx'                          ref='x'
09:05:43 src.training.trainer INFO:   [2] pred='rxxxx'                        ref='r'
09:06:09 src.training.trainer INFO: Epoch 51/200 — train_loss=0.1747, val_loss=1.0914, val_cer=0.4271, val_wer=0.5512, lr=8.54e-05, time=25.6s
09:06:09 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:06:09 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:06:09 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:06:35 src.training.trainer INFO: Epoch 52/200 — train_loss=0.2006, val_loss=1.0061, val_cer=0.9771, val_wer=0.7921, lr=8.48e-05, time=25.4s
09:06:35 src.training.trainer INFO:   [0] pred='hx'                           ref='h'
09:06:35 src.training.trainer INFO:   [1] pred='xx'                           ref='x'
09:06:35 src.training.trainer INFO:   [2] pred='rxxxxxxxxxx'                  ref='r'
09:07:00 src.training.trainer INFO: Epoch 53/200 — train_loss=0.1962, val_loss=1.2394, val_cer=0.4245, val_wer=0.5402, lr=8.42e-05, time=24.7s
09:07:00 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:07:00 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:07:00 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:07:25 src.training.trainer INFO: Epoch 54/200 — train_loss=0.1300, val_loss=1.2305, val_cer=0.4245, val_wer=0.5669, lr=4.18e-05, time=24.7s
09:07:25 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:07:25 src.training.trainer INFO:   [1] pred='xh'                           ref='x'
09:07:25 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:07:50 src.training.trainer INFO: Epoch 55/200 — train_loss=0.1680, val_loss=1.3155, val_cer=0.4042, val_wer=0.5543, lr=8.31e-05, time=25.1s
09:07:50 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:07:50 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:07:50 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:07:51 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.4042)
09:08:16 src.training.trainer INFO: Epoch 56/200 — train_loss=0.1457, val_loss=1.1007, val_cer=0.4443, val_wer=0.5480, lr=8.25e-05, time=25.1s
09:08:16 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:08:16 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:08:16 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:08:40 src.training.trainer INFO: Epoch 57/200 — train_loss=0.1379, val_loss=1.0680, val_cer=0.4677, val_wer=0.5827, lr=8.19e-05, time=23.8s
09:08:40 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:08:40 src.training.trainer INFO:   [1] pred='th'                           ref='x'
09:08:40 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:09:05 src.training.trainer INFO: Epoch 58/200 — train_loss=0.1362, val_loss=1.1199, val_cer=0.4630, val_wer=0.5559, lr=8.12e-05, time=25.2s
09:09:05 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:09:05 src.training.trainer INFO:   [1] pred='t'                            ref='x'
09:09:05 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:09:30 src.training.trainer INFO: Epoch 59/200 — train_loss=0.1607, val_loss=1.8985, val_cer=0.4281, val_wer=0.5969, lr=8.06e-05, time=25.1s
09:09:30 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:09:30 src.training.trainer INFO:   [1] pred='th'                           ref='x'
09:09:30 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:09:55 src.training.trainer INFO: Epoch 60/200 — train_loss=0.1468, val_loss=1.4794, val_cer=0.4234, val_wer=0.5591, lr=8.00e-05, time=24.0s
09:09:55 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:09:55 src.training.trainer INFO:   [1] pred='th'                           ref='x'
09:09:55 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:10:20 src.training.trainer INFO: Epoch 61/200 — train_loss=0.1460, val_loss=1.4069, val_cer=0.4047, val_wer=0.5512, lr=3.97e-05, time=24.6s
09:10:20 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:10:20 src.training.trainer INFO:   [1] pred='th'                           ref='x'
09:10:20 src.training.trainer INFO:   [2] pred='rr'                           ref='r'
09:10:45 src.training.trainer INFO: Epoch 62/200 — train_loss=0.1586, val_loss=1.2698, val_cer=0.4036, val_wer=0.5606, lr=7.87e-05, time=25.3s
09:10:45 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:10:45 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:10:45 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:10:46 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.4036)
09:11:11 src.training.trainer INFO: Epoch 63/200 — train_loss=0.1207, val_loss=1.2839, val_cer=0.4016, val_wer=0.5449, lr=7.81e-05, time=25.0s
09:11:11 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:11:11 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:11:11 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:11:11 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.4016)
09:11:35 src.training.trainer INFO: Epoch 64/200 — train_loss=0.1067, val_loss=1.9548, val_cer=0.4307, val_wer=0.5780, lr=7.74e-05, time=23.9s
09:11:35 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:11:35 src.training.trainer INFO:   [1] pred='txh'                          ref='x'
09:11:35 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:12:00 src.training.trainer INFO: Epoch 65/200 — train_loss=0.0983, val_loss=1.3787, val_cer=0.3927, val_wer=0.5449, lr=7.67e-05, time=24.7s
09:12:00 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:12:00 src.training.trainer INFO:   [1] pred='th'                           ref='x'
09:12:00 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:12:01 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.3927)
09:12:26 src.training.trainer INFO: Epoch 66/200 — train_loss=0.1043, val_loss=1.7032, val_cer=0.4307, val_wer=0.5669, lr=7.61e-05, time=25.3s
09:12:26 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:12:26 src.training.trainer INFO:   [1] pred='th'                           ref='x'
09:12:26 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:12:51 src.training.trainer INFO: Epoch 67/200 — train_loss=0.1199, val_loss=1.1817, val_cer=0.3974, val_wer=0.5213, lr=7.54e-05, time=24.3s
09:12:51 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:12:51 src.training.trainer INFO:   [1] pred='tx'                           ref='x'
09:12:51 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:13:15 src.training.trainer INFO: Epoch 68/200 — train_loss=0.1027, val_loss=1.1704, val_cer=0.3745, val_wer=0.5102, lr=7.47e-05, time=24.1s
09:13:15 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:13:15 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:13:15 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:13:16 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.3745)
09:13:41 src.training.trainer INFO: Epoch 69/200 — train_loss=0.0857, val_loss=1.2042, val_cer=0.3516, val_wer=0.5260, lr=7.40e-05, time=25.0s
09:13:41 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:13:41 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:13:41 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:13:41 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNTransformer_best.pt (val_cer=0.3516)
09:14:07 src.training.trainer INFO: Epoch 70/200 — train_loss=0.0834, val_loss=1.1404, val_cer=0.3703, val_wer=0.5260, lr=7.33e-05, time=25.3s
09:14:07 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:14:07 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:14:07 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:14:31 src.training.trainer INFO: Epoch 71/200 — train_loss=0.0709, val_loss=1.8423, val_cer=0.4005, val_wer=0.5606, lr=7.26e-05, time=23.9s
09:14:31 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:14:31 src.training.trainer INFO:   [1] pred='bh'                           ref='x'
09:14:31 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:14:56 src.training.trainer INFO: Epoch 72/200 — train_loss=0.1033, val_loss=1.3686, val_cer=0.3818, val_wer=0.5323, lr=7.19e-05, time=24.5s
09:14:56 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:14:56 src.training.trainer INFO:   [1] pred='a'                            ref='x'
09:14:56 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:15:21 src.training.trainer INFO: Epoch 73/200 — train_loss=0.0830, val_loss=1.8536, val_cer=0.3927, val_wer=0.5449, lr=7.12e-05, time=25.2s
09:15:21 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:15:21 src.training.trainer INFO:   [1] pred='h'                            ref='x'
09:15:21 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:15:46 src.training.trainer INFO: Epoch 74/200 — train_loss=0.0970, val_loss=1.8781, val_cer=0.4208, val_wer=0.5969, lr=7.05e-05, time=24.7s
09:15:46 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:15:46 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:15:46 src.training.trainer INFO:   [2] pred='ra'                           ref='r'
09:16:11 src.training.trainer INFO: Epoch 75/200 — train_loss=0.0601, val_loss=1.3713, val_cer=0.3865, val_wer=0.5370, lr=3.49e-05, time=24.3s
09:16:11 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:16:11 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:16:11 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:16:36 src.training.trainer INFO: Epoch 76/200 — train_loss=0.0733, val_loss=1.9074, val_cer=0.4146, val_wer=0.5654, lr=6.90e-05, time=25.2s
09:16:36 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:16:36 src.training.trainer INFO:   [1] pred='xh'                           ref='x'
09:16:36 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:17:02 src.training.trainer INFO: Epoch 77/200 — train_loss=0.0954, val_loss=2.0602, val_cer=0.4010, val_wer=0.5858, lr=6.83e-05, time=25.3s
09:17:02 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:17:02 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:17:02 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:17:26 src.training.trainer INFO: Epoch 78/200 — train_loss=0.0806, val_loss=1.4507, val_cer=0.3880, val_wer=0.5386, lr=6.76e-05, time=24.0s
09:17:26 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:17:26 src.training.trainer INFO:   [1] pred='xh'                           ref='x'
09:17:26 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:17:51 src.training.trainer INFO: Epoch 79/200 — train_loss=0.0888, val_loss=2.1152, val_cer=1.5734, val_wer=0.8268, lr=6.68e-05, time=24.3s
09:17:51 src.training.trainer INFO:   [0] pred='hmmmmmmmmmmmm'                ref='h'
09:17:51 src.training.trainer INFO:   [1] pred='hmmmmmmmmmmmmmm'              ref='x'
09:17:51 src.training.trainer INFO:   [2] pred='rmmmmmmmmmmmm'                ref='r'
09:18:16 src.training.trainer INFO: Epoch 80/200 — train_loss=0.0897, val_loss=1.2188, val_cer=0.3771, val_wer=0.5370, lr=6.61e-05, time=25.0s
09:18:16 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:18:16 src.training.trainer INFO:   [1] pred='xh'                           ref='x'
09:18:16 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:18:41 src.training.trainer INFO: Epoch 81/200 — train_loss=0.0683, val_loss=1.1906, val_cer=0.3760, val_wer=0.5276, lr=3.27e-05, time=24.9s
09:18:41 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:18:41 src.training.trainer INFO:   [1] pred='xh'                           ref='x'
09:18:41 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:19:06 src.training.trainer INFO: Epoch 82/200 — train_loss=0.0877, val_loss=1.4926, val_cer=0.4010, val_wer=0.5685, lr=6.46e-05, time=24.3s
09:19:06 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:19:06 src.training.trainer INFO:   [1] pred='xh'                           ref='x'
09:19:06 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:19:32 src.training.trainer INFO: Epoch 83/200 — train_loss=0.0592, val_loss=1.1717, val_cer=0.3807, val_wer=0.5638, lr=6.38e-05, time=25.4s
09:19:32 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:19:32 src.training.trainer INFO:   [1] pred='xh'                           ref='x'
09:19:32 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:19:57 src.training.trainer INFO: Epoch 84/200 — train_loss=0.0729, val_loss=1.3297, val_cer=0.3875, val_wer=0.5323, lr=6.30e-05, time=25.4s
09:19:57 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:19:57 src.training.trainer INFO:   [1] pred='h'                            ref='x'
09:19:57 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:20:22 src.training.trainer INFO: Epoch 85/200 — train_loss=0.0515, val_loss=1.1977, val_cer=0.3745, val_wer=0.5339, lr=6.23e-05, time=24.8s
09:20:22 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:20:22 src.training.trainer INFO:   [1] pred='h'                            ref='x'
09:20:22 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:20:47 src.training.trainer INFO: Epoch 86/200 — train_loss=0.0584, val_loss=1.5190, val_cer=0.3901, val_wer=0.5559, lr=6.15e-05, time=24.3s
09:20:47 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:20:47 src.training.trainer INFO:   [1] pred='h'                            ref='x'
09:20:47 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:21:13 src.training.trainer INFO: Epoch 87/200 — train_loss=0.0856, val_loss=1.2579, val_cer=0.3964, val_wer=0.5559, lr=3.04e-05, time=25.5s
09:21:13 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:21:13 src.training.trainer INFO:   [1] pred='x'                            ref='x'
09:21:13 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:21:39 src.training.trainer INFO: Epoch 88/200 — train_loss=0.0527, val_loss=1.1569, val_cer=0.3802, val_wer=0.5354, lr=6.00e-05, time=25.5s
09:21:39 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:21:39 src.training.trainer INFO:   [1] pred='z'                            ref='x'
09:21:39 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:22:04 src.training.trainer INFO: Epoch 89/200 — train_loss=0.0522, val_loss=1.8607, val_cer=0.3870, val_wer=0.5669, lr=5.92e-05, time=24.9s
09:22:04 src.training.trainer INFO:   [0] pred='h'                            ref='h'
09:22:04 src.training.trainer INFO:   [1] pred='zh'                           ref='x'
09:22:04 src.training.trainer INFO:   [2] pred='r'                            ref='r'
09:22:05 src.training.trainer INFO: Early stopping: no improvement for 20 epochs (best_cer=0.3516)
09:22:05 src.training.trainer INFO: Training complete. Best val CER: 0.3516
wandb: uploading artifact run-a2kqln33-decoded_samples; updating run metadata
wandb: uploading artifact run-a2kqln33-decoded_samples
wandb: uploading output.log; uploading wandb-summary.json; uploading config.yaml
wandb: uploading config.yaml
wandb: uploading history steps 87-88, summary, console lines 384-389
wandb: 
wandb: Run history:
wandb:          best/epoch ▁▂▂▂▂▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▇▇▇███
wandb:        best/val_cer ██▇▇▇▆▆▆▆▆▅▅▄▄▃▃▂▂▂▂▂▂▂▁▁▁
wandb:        epoch_time_s ▄▇▃▂▂▅▅▃▆▅▆▁▆▁▄▇▇▄▆▅█▇▅▅▆▇▂▇▃▇▄▅▇▇▂▆▅▇▅▆
wandb:  final/best_val_cer ▁
wandb:  final/total_epochs ▁
wandb:         train/epoch ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▇▇▇███
wandb:   train/global_step ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▅▆▆▆▇▇▇▇▇███
wandb: train/learning_rate ████████████▇▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▅▅▅▅▁▅▁▄▄▄
wandb:          train/loss █▄▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:             val/cer ████▇▇▇▆▆▆▆▆▇▅▄▃▂▂▂▂▂▂█▁▂▂▁▁▁▂▂▁▁▁▁▁▁▁▁▁
wandb:                  +2 ...
wandb: 
wandb: Run summary:
wandb:          best/epoch 69
wandb:        best/val_cer 0.35156
wandb:        epoch_time_s 24.91894
wandb:  final/best_val_cer 0.35156
wandb:  final/total_epochs 89
wandb:         train/epoch 89
wandb:   train/global_step 32841
wandb: train/learning_rate 6e-05
wandb:          train/loss 0.05224
wandb:             val/cer 0.38698
wandb:                  +2 ...
wandb: 
wandb: 🚀 View run cnn_transformer_gpu_v1 at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/a2kqln33
wandb: ⭐️ View project at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: Synced 5 W&B file(s), 89 media file(s), 178 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260313_084444-a2kqln33/logs
09:22:09 __main__ INFO: Best validation CER: 0.3516

cnn_transformer finished in 37.7 min (exit code 0)

======================================================================
TRAINING: gru_decoder
======================================================================
Command: python scripts/train.py --model gru_decoder --epochs 200 --batch-size 16 --lr 0.0001 --t-max 5000 --normalize --filter-by-length --wandb --wandb-project brain-text-decoder --wandb-run-name gru_decoder_gpu_v1 --wandb-tags baseline gru gpu full-size --hidden-size 512 --n-layers 3 --dropout 0.3

09:22:15 __main__ INFO: Loading dataset...
09:22:15 src.data.loader INFO: Loading cached trial index from data/willett_handwriting/trial_index.csv
09:22:15 __main__ INFO: Loaded 4126 trials
09:22:15 __main__ INFO: Length filter (t_max=5000): 4126 -> 3868 trials
09:22:15 __main__ INFO: Signal lengths: min=201, max=5000, mean=705, median=201
09:22:15 src.data.dataset INFO: Split: 3094 train, 386 val, 388 test
09:22:16 src.data.dataset INFO: Computed channel stats from 200 trials: mean=0.2143, std=0.4529
09:22:16 __main__ INFO: Model: gru_decoder (4922908 params)
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /root/.netrc.
wandb: Currently logged in as: dlopezkluever (dlopezkluever-aiuteur) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
wandb: setting up run p6guyu6f
wandb: Tracking run with wandb version 0.25.0
wandb: Run data is saved locally in /content/BCI-2/wandb/run-20260313_092219-p6guyu6f
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run gru_decoder_gpu_v1
wandb: ⭐️ View project at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: 🚀 View run at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/p6guyu6f
09:22:21 src.training.trainer INFO: W&B run initialized: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/p6guyu6f
09:22:21 src.training.trainer INFO: Starting training: model=GRUDecoder, device=cuda, epochs=200, amp=True
09:22:21 src.training.trainer INFO: Parameters: 4922908
09:22:21 src.training.trainer INFO: Temporal downsample factor: 4
/content/BCI-2/src/training/trainer.py:237: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  self.scheduler.step()
09:22:53 src.training.trainer INFO: Epoch 1/200 — train_loss=70.9886, val_loss=4.2182, val_cer=1.0000, val_wer=1.0000, lr=3.88e-05, time=32.2s
09:22:53 src.training.trainer INFO:   [0] pred=''                             ref='daisy leaned forward at once horrified and fascina'
09:22:53 src.training.trainer INFO:   [1] pred=''                             ref='r'
09:22:53 src.training.trainer INFO:   [2] pred=''                             ref='k'
09:22:55 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/GRUDecoder_best.pt (val_cer=1.0000)
09:23:26 src.training.trainer INFO: Epoch 2/200 — train_loss=4.0716, val_loss=3.9301, val_cer=1.0000, val_wer=1.0000, lr=7.76e-05, time=31.0s
09:23:26 src.training.trainer INFO:   [0] pred=''                             ref='daisy leaned forward at once horrified and fascina'
09:23:26 src.training.trainer INFO:   [1] pred=''                             ref='r'
09:23:26 src.training.trainer INFO:   [2] pred=''                             ref='k'
09:23:57 src.training.trainer INFO: Epoch 3/200 — train_loss=3.9177, val_loss=3.7582, val_cer=1.0000, val_wer=1.0000, lr=1.00e-04, time=30.9s
09:23:57 src.training.trainer INFO:   [0] pred=''                             ref='daisy leaned forward at once horrified and fascina'
09:23:57 src.training.trainer INFO:   [1] pred=''                             ref='r'
09:23:57 src.training.trainer INFO:   [2] pred=''                             ref='k'
09:24:26 src.training.trainer INFO: Epoch 4/200 — train_loss=3.7946, val_loss=3.6037, val_cer=0.9938, val_wer=1.0000, lr=1.00e-04, time=28.6s
09:24:26 src.training.trainer INFO:   [0] pred=''                             ref='daisy leaned forward at once horrified and fascina'
09:24:26 src.training.trainer INFO:   [1] pred=''                             ref='r'
09:24:26 src.training.trainer INFO:   [2] pred=''                             ref='k'
09:24:27 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/GRUDecoder_best.pt (val_cer=0.9938)
09:24:58 src.training.trainer INFO: Epoch 5/200 — train_loss=3.3695, val_loss=3.3325, val_cer=0.9855, val_wer=1.0000, lr=1.00e-04, time=31.2s
09:24:58 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:24:58 src.training.trainer INFO:   [1] pred='t'                            ref='r'
09:24:58 src.training.trainer INFO:   [2] pred='t'                            ref='k'
09:24:58 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/GRUDecoder_best.pt (val_cer=0.9855)
09:25:28 src.training.trainer INFO: Epoch 6/200 — train_loss=3.1678, val_loss=3.3824, val_cer=0.9828, val_wer=1.0038, lr=9.99e-05, time=29.6s
09:25:28 src.training.trainer INFO:   [0] pred='e'                            ref='daisy leaned forward at once horrified and fascina'
09:25:28 src.training.trainer INFO:   [1] pred='s'                            ref='r'
09:25:28 src.training.trainer INFO:   [2] pred='e'                            ref='k'
09:25:28 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/GRUDecoder_best.pt (val_cer=0.9828)
09:25:57 src.training.trainer INFO: Epoch 7/200 — train_loss=3.0966, val_loss=3.4761, val_cer=0.9841, val_wer=0.9987, lr=9.99e-05, time=28.9s
09:25:57 src.training.trainer INFO:   [0] pred='i'                            ref='daisy leaned forward at once horrified and fascina'
09:25:57 src.training.trainer INFO:   [1] pred='s'                            ref='r'
09:25:57 src.training.trainer INFO:   [2] pred='i'                            ref='k'
09:26:26 src.training.trainer INFO: Epoch 8/200 — train_loss=3.0410, val_loss=3.5270, val_cer=0.9817, val_wer=0.9950, lr=9.98e-05, time=28.5s
09:26:26 src.training.trainer INFO:   [0] pred='n'                            ref='daisy leaned forward at once horrified and fascina'
09:26:26 src.training.trainer INFO:   [1] pred='e'                            ref='r'
09:26:26 src.training.trainer INFO:   [2] pred='n'                            ref='k'
09:26:26 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/GRUDecoder_best.pt (val_cer=0.9817)
09:26:57 src.training.trainer INFO: Epoch 9/200 — train_loss=2.9824, val_loss=3.5469, val_cer=0.9931, val_wer=1.0063, lr=9.97e-05, time=30.5s
09:26:57 src.training.trainer INFO:   [0] pred='u'                            ref='daisy leaned forward at once horrified and fascina'
09:26:57 src.training.trainer INFO:   [1] pred='f'                            ref='r'
09:26:57 src.training.trainer INFO:   [2] pred='q'                            ref='k'
09:27:25 src.training.trainer INFO: Epoch 10/200 — train_loss=2.9200, val_loss=3.6723, val_cer=0.9852, val_wer=0.9937, lr=9.97e-05, time=28.2s
09:27:25 src.training.trainer INFO:   [0] pred='u'                            ref='daisy leaned forward at once horrified and fascina'
09:27:25 src.training.trainer INFO:   [1] pred='i'                            ref='r'
09:27:25 src.training.trainer INFO:   [2] pred='i'                            ref='k'
09:27:53 src.training.trainer INFO: Epoch 11/200 — train_loss=2.8732, val_loss=3.7769, val_cer=0.9817, val_wer=0.9950, lr=9.96e-05, time=27.2s
09:27:53 src.training.trainer INFO:   [0] pred='e'                            ref='daisy leaned forward at once horrified and fascina'
09:27:53 src.training.trainer INFO:   [1] pred='e'                            ref='r'
09:27:53 src.training.trainer INFO:   [2] pred='a'                            ref='k'
09:28:22 src.training.trainer INFO: Epoch 12/200 — train_loss=2.8059, val_loss=3.8454, val_cer=0.9838, val_wer=0.9950, lr=9.94e-05, time=29.3s
09:28:22 src.training.trainer INFO:   [0] pred='u'                            ref='daisy leaned forward at once horrified and fascina'
09:28:22 src.training.trainer INFO:   [1] pred='f'                            ref='r'
09:28:22 src.training.trainer INFO:   [2] pred='h'                            ref='k'
09:28:51 src.training.trainer INFO: Epoch 13/200 — train_loss=2.7582, val_loss=3.9645, val_cer=0.9824, val_wer=0.9887, lr=9.93e-05, time=28.3s
09:28:51 src.training.trainer INFO:   [0] pred='c'                            ref='daisy leaned forward at once horrified and fascina'
09:28:51 src.training.trainer INFO:   [1] pred='i'                            ref='r'
09:28:51 src.training.trainer INFO:   [2] pred='r'                            ref='k'
09:29:22 src.training.trainer INFO: Epoch 14/200 — train_loss=2.6913, val_loss=4.0972, val_cer=0.9869, val_wer=1.0000, lr=4.96e-05, time=30.5s
09:29:22 src.training.trainer INFO:   [0] pred='g'                            ref='daisy leaned forward at once horrified and fascina'
09:29:22 src.training.trainer INFO:   [1] pred='g'                            ref='r'
09:29:22 src.training.trainer INFO:   [2] pred='q'                            ref='k'
09:29:50 src.training.trainer INFO: Epoch 15/200 — train_loss=2.6586, val_loss=4.1111, val_cer=0.9831, val_wer=0.9937, lr=9.90e-05, time=28.2s
09:29:50 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:29:50 src.training.trainer INFO:   [1] pred='t'                            ref='r'
09:29:50 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:30:19 src.training.trainer INFO: Epoch 16/200 — train_loss=2.6106, val_loss=4.1663, val_cer=0.9848, val_wer=0.9987, lr=9.89e-05, time=28.0s
09:30:19 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:30:19 src.training.trainer INFO:   [1] pred='n'                            ref='r'
09:30:19 src.training.trainer INFO:   [2] pred='w'                            ref='k'
09:30:47 src.training.trainer INFO: Epoch 17/200 — train_loss=2.5729, val_loss=4.2238, val_cer=0.9824, val_wer=0.9899, lr=9.87e-05, time=28.2s
09:30:47 src.training.trainer INFO:   [0] pred='g'                            ref='daisy leaned forward at once horrified and fascina'
09:30:47 src.training.trainer INFO:   [1] pred='g'                            ref='r'
09:30:47 src.training.trainer INFO:   [2] pred='w'                            ref='k'
09:31:18 src.training.trainer INFO: Epoch 18/200 — train_loss=2.5233, val_loss=4.3423, val_cer=0.9869, val_wer=1.0025, lr=9.85e-05, time=31.1s
09:31:18 src.training.trainer INFO:   [0] pred='e'                            ref='daisy leaned forward at once horrified and fascina'
09:31:18 src.training.trainer INFO:   [1] pred='f'                            ref='r'
09:31:18 src.training.trainer INFO:   [2] pred='l'                            ref='k'
09:31:48 src.training.trainer INFO: Epoch 19/200 — train_loss=2.4904, val_loss=4.3857, val_cer=0.9834, val_wer=0.9912, lr=9.83e-05, time=28.9s
09:31:48 src.training.trainer INFO:   [0] pred='f'                            ref='daisy leaned forward at once horrified and fascina'
09:31:48 src.training.trainer INFO:   [1] pred='f'                            ref='r'
09:31:48 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:32:18 src.training.trainer INFO: Epoch 20/200 — train_loss=2.4617, val_loss=4.4630, val_cer=0.9834, val_wer=1.0000, lr=4.90e-05, time=30.5s
09:32:18 src.training.trainer INFO:   [0] pred='f'                            ref='daisy leaned forward at once horrified and fascina'
09:32:18 src.training.trainer INFO:   [1] pred='t'                            ref='r'
09:32:18 src.training.trainer INFO:   [2] pred='w'                            ref='k'
09:32:49 src.training.trainer INFO: Epoch 21/200 — train_loss=2.4124, val_loss=4.5838, val_cer=0.9879, val_wer=1.0025, lr=9.79e-05, time=30.7s
09:32:49 src.training.trainer INFO:   [0] pred='f'                            ref='daisy leaned forward at once horrified and fascina'
09:32:49 src.training.trainer INFO:   [1] pred='f'                            ref='r'
09:32:49 src.training.trainer INFO:   [2] pred='a'                            ref='k'
09:33:17 src.training.trainer INFO: Epoch 22/200 — train_loss=2.3972, val_loss=4.5232, val_cer=0.9841, val_wer=0.9987, lr=9.76e-05, time=27.7s
09:33:17 src.training.trainer INFO:   [0] pred='n'                            ref='daisy leaned forward at once horrified and fascina'
09:33:17 src.training.trainer INFO:   [1] pred='g'                            ref='r'
09:33:17 src.training.trainer INFO:   [2] pred='a'                            ref='k'
09:33:47 src.training.trainer INFO: Epoch 23/200 — train_loss=2.3860, val_loss=4.5516, val_cer=0.9834, val_wer=0.9937, lr=9.74e-05, time=29.7s
09:33:47 src.training.trainer INFO:   [0] pred='g'                            ref='daisy leaned forward at once horrified and fascina'
09:33:47 src.training.trainer INFO:   [1] pred='n'                            ref='r'
09:33:47 src.training.trainer INFO:   [2] pred='w'                            ref='k'
09:34:16 src.training.trainer INFO: Epoch 24/200 — train_loss=2.3629, val_loss=4.6822, val_cer=0.9890, val_wer=1.0013, lr=9.71e-05, time=28.2s
09:34:16 src.training.trainer INFO:   [0] pred='g'                            ref='daisy leaned forward at once horrified and fascina'
09:34:16 src.training.trainer INFO:   [1] pred='g'                            ref='r'
09:34:16 src.training.trainer INFO:   [2] pred='a'                            ref='k'
09:34:46 src.training.trainer INFO: Epoch 25/200 — train_loss=2.3152, val_loss=4.7055, val_cer=0.9848, val_wer=1.0076, lr=9.69e-05, time=29.3s
09:34:46 src.training.trainer INFO:   [0] pred='f'                            ref='daisy leaned forward at once horrified and fascina'
09:34:46 src.training.trainer INFO:   [1] pred='n'                            ref='r'
09:34:46 src.training.trainer INFO:   [2] pred='a'                            ref='k'
09:35:17 src.training.trainer INFO: Epoch 26/200 — train_loss=2.3162, val_loss=4.7146, val_cer=0.9821, val_wer=1.0000, lr=4.83e-05, time=30.9s
09:35:17 src.training.trainer INFO:   [0] pred='f'                            ref='daisy leaned forward at once horrified and fascina'
09:35:17 src.training.trainer INFO:   [1] pred='n'                            ref='r'
09:35:17 src.training.trainer INFO:   [2] pred='n'                            ref='k'
09:35:47 src.training.trainer INFO: Epoch 27/200 — train_loss=2.2695, val_loss=4.8220, val_cer=0.9883, val_wer=1.0076, lr=9.63e-05, time=29.6s
09:35:47 src.training.trainer INFO:   [0] pred='z'                            ref='daisy leaned forward at once horrified and fascina'
09:35:47 src.training.trainer INFO:   [1] pred='f'                            ref='r'
09:35:47 src.training.trainer INFO:   [2] pred='z'                            ref='k'
09:36:18 src.training.trainer INFO: Epoch 28/200 — train_loss=2.2570, val_loss=4.8449, val_cer=0.9848, val_wer=1.0000, lr=9.60e-05, time=30.9s
09:36:18 src.training.trainer INFO:   [0] pred='f'                            ref='daisy leaned forward at once horrified and fascina'
09:36:18 src.training.trainer INFO:   [1] pred='i'                            ref='r'
09:36:18 src.training.trainer INFO:   [2] pred='i'                            ref='k'
09:36:18 src.training.trainer INFO: Early stopping: no improvement for 20 epochs (best_cer=0.9817)
09:36:18 src.training.trainer INFO: Training complete. Best val CER: 0.9817
wandb: uploading artifact run-p6guyu6f-decoded_samples; updating run metadata
wandb: uploading artifact run-p6guyu6f-decoded_samples
wandb: uploading history steps 26-27, summary, console lines 119-124; uploading output.log; uploading wandb-summary.json; uploading config.yaml
wandb: uploading history steps 26-27, summary, console lines 119-124
wandb: uploading data
wandb: 
wandb: Run history:
wandb:          best/epoch ▁▄▅▆█
wandb:        best/val_cer █▆▂▁▁
wandb:        epoch_time_s █▆▆▃▇▄▃▃▆▂▁▄▃▆▂▂▂▆▃▆▆▂▅▂▄▆▄▆
wandb:  final/best_val_cer ▁
wandb:  final/total_epochs ▁
wandb:         train/epoch ▁▁▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▇▇▇▇██
wandb:   train/global_step ▁▁▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▇▇▇▇██
wandb: train/learning_rate ▁▅███████████▂█████▂█████▂██
wandb:          train/loss █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:             val/cer ███▆▂▁▂▁▅▂▁▂▁▃▂▂▁▃▂▂▃▂▂▄▂▁▄▂
wandb:                  +2 ...
wandb: 
wandb: Run summary:
wandb:          best/epoch 8
wandb:        best/val_cer 0.98172
wandb:        epoch_time_s 30.87913
wandb:  final/best_val_cer 0.98172
wandb:  final/total_epochs 28
wandb:         train/epoch 28
wandb:   train/global_step 5432
wandb: train/learning_rate 0.0001
wandb:          train/loss 2.25703
wandb:             val/cer 0.98482
wandb:                  +2 ...
wandb: 
wandb: 🚀 View run gru_decoder_gpu_v1 at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/p6guyu6f
wandb: ⭐️ View project at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: Synced 5 W&B file(s), 28 media file(s), 56 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260313_092219-p6guyu6f/logs
09:36:26 __main__ INFO: Best validation CER: 0.9817

gru_decoder finished in 14.2 min (exit code 0)

======================================================================
TRAINING: cnn_lstm
======================================================================
Command: python scripts/train.py --model cnn_lstm --epochs 200 --batch-size 12 --lr 0.0001 --t-max 5000 --normalize --filter-by-length --wandb --wandb-project brain-text-decoder --wandb-run-name cnn_lstm_gpu_v1 --wandb-tags cnn-lstm gpu full-size --hidden-size 512 --n-layers 2 --dropout 0.5

09:36:30 __main__ INFO: Loading dataset...
09:36:30 src.data.loader INFO: Loading cached trial index from data/willett_handwriting/trial_index.csv
09:36:30 __main__ INFO: Loaded 4126 trials
09:36:30 __main__ INFO: Length filter (t_max=5000): 4126 -> 3868 trials
09:36:30 __main__ INFO: Signal lengths: min=201, max=5000, mean=705, median=201
09:36:30 src.data.dataset INFO: Split: 3094 train, 386 val, 388 test
09:36:30 src.data.dataset INFO: Computed channel stats from 200 trials: mean=0.2143, std=0.4529
09:36:30 __main__ INFO: Model: cnn_lstm (10746140 params)
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /root/.netrc.
wandb: Currently logged in as: dlopezkluever (dlopezkluever-aiuteur) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
wandb: setting up run bglnrqpm
wandb: Tracking run with wandb version 0.25.0
wandb: Run data is saved locally in /content/BCI-2/wandb/run-20260313_093633-bglnrqpm
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run cnn_lstm_gpu_v1
wandb: ⭐️ View project at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: 🚀 View run at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/bglnrqpm
09:36:35 src.training.trainer INFO: W&B run initialized: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/bglnrqpm
09:36:35 src.training.trainer INFO: Starting training: model=CNNLSTM, device=cuda, epochs=200, amp=True
09:36:35 src.training.trainer INFO: Parameters: 10746140
09:36:35 src.training.trainer INFO: Temporal downsample factor: 4
/content/BCI-2/src/training/trainer.py:237: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  self.scheduler.step()
09:37:13 src.training.trainer INFO: Epoch 1/200 — train_loss=50.4682, val_loss=4.3271, val_cer=1.0000, val_wer=1.0000, lr=5.16e-05, time=37.8s
09:37:13 src.training.trainer INFO:   [0] pred=''                             ref='daisy leaned forward at once horrified and fascina'
09:37:13 src.training.trainer INFO:   [1] pred=''                             ref='r'
09:37:13 src.training.trainer INFO:   [2] pred=''                             ref='k'
09:37:14 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=1.0000)
09:37:52 src.training.trainer INFO: Epoch 2/200 — train_loss=3.9611, val_loss=3.9209, val_cer=0.9879, val_wer=1.0076, lr=1.00e-04, time=37.6s
09:37:52 src.training.trainer INFO:   [0] pred='u'                            ref='daisy leaned forward at once horrified and fascina'
09:37:52 src.training.trainer INFO:   [1] pred='u'                            ref='r'
09:37:52 src.training.trainer INFO:   [2] pred='u'                            ref='k'
09:37:53 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9879)
09:38:33 src.training.trainer INFO: Epoch 3/200 — train_loss=3.5156, val_loss=3.7312, val_cer=0.9786, val_wer=0.9987, lr=1.00e-04, time=40.0s
09:38:33 src.training.trainer INFO:   [0] pred='n'                            ref='daisy leaned forward at once horrified and fascina'
09:38:33 src.training.trainer INFO:   [1] pred='n'                            ref='r'
09:38:33 src.training.trainer INFO:   [2] pred='n'                            ref='k'
09:38:33 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9786)
09:39:13 src.training.trainer INFO: Epoch 4/200 — train_loss=3.2851, val_loss=3.4317, val_cer=0.9879, val_wer=1.0076, lr=1.00e-04, time=40.3s
09:39:13 src.training.trainer INFO:   [0] pred='u'                            ref='daisy leaned forward at once horrified and fascina'
09:39:13 src.training.trainer INFO:   [1] pred='u'                            ref='r'
09:39:13 src.training.trainer INFO:   [2] pred='u'                            ref='k'
09:39:53 src.training.trainer INFO: Epoch 5/200 — train_loss=3.2587, val_loss=3.3626, val_cer=0.9879, val_wer=1.0076, lr=9.99e-05, time=39.7s
09:39:53 src.training.trainer INFO:   [0] pred='u'                            ref='daisy leaned forward at once horrified and fascina'
09:39:53 src.training.trainer INFO:   [1] pred='u'                            ref='r'
09:39:53 src.training.trainer INFO:   [2] pred='u'                            ref='k'
09:40:33 src.training.trainer INFO: Epoch 6/200 — train_loss=3.2106, val_loss=3.3819, val_cer=0.9886, val_wer=1.0063, lr=9.99e-05, time=39.7s
09:40:33 src.training.trainer INFO:   [0] pred='f'                            ref='daisy leaned forward at once horrified and fascina'
09:40:33 src.training.trainer INFO:   [1] pred='f'                            ref='r'
09:40:33 src.training.trainer INFO:   [2] pred='f'                            ref='k'
09:41:11 src.training.trainer INFO: Epoch 7/200 — train_loss=3.1436, val_loss=3.3530, val_cer=0.9803, val_wer=0.9861, lr=9.98e-05, time=37.4s
09:41:11 src.training.trainer INFO:   [0] pred='w'                            ref='daisy leaned forward at once horrified and fascina'
09:41:11 src.training.trainer INFO:   [1] pred='g'                            ref='r'
09:41:11 src.training.trainer INFO:   [2] pred='w'                            ref='k'
09:41:48 src.training.trainer INFO: Epoch 8/200 — train_loss=2.9716, val_loss=3.0425, val_cer=0.9834, val_wer=0.9899, lr=9.98e-05, time=37.1s
09:41:48 src.training.trainer INFO:   [0] pred='m'                            ref='daisy leaned forward at once horrified and fascina'
09:41:48 src.training.trainer INFO:   [1] pred='m'                            ref='r'
09:41:48 src.training.trainer INFO:   [2] pred='m'                            ref='k'
09:42:26 src.training.trainer INFO: Epoch 9/200 — train_loss=2.8169, val_loss=2.9842, val_cer=0.9852, val_wer=0.9760, lr=4.98e-05, time=37.6s
09:42:26 src.training.trainer INFO:   [0] pred='x'                            ref='daisy leaned forward at once horrified and fascina'
09:42:26 src.training.trainer INFO:   [1] pred='x'                            ref='r'
09:42:26 src.training.trainer INFO:   [2] pred='p'                            ref='k'
09:43:03 src.training.trainer INFO: Epoch 10/200 — train_loss=2.6779, val_loss=3.0027, val_cer=0.9752, val_wer=0.9622, lr=9.96e-05, time=36.6s
09:43:03 src.training.trainer INFO:   [0] pred='k'                            ref='daisy leaned forward at once horrified and fascina'
09:43:03 src.training.trainer INFO:   [1] pred='l'                            ref='r'
09:43:03 src.training.trainer INFO:   [2] pred='b'                            ref='k'
09:43:04 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9752)
09:43:40 src.training.trainer INFO: Epoch 11/200 — train_loss=2.5663, val_loss=2.8790, val_cer=0.9731, val_wer=0.9634, lr=9.95e-05, time=36.0s
09:43:40 src.training.trainer INFO:   [0] pred='r'                            ref='daisy leaned forward at once horrified and fascina'
09:43:40 src.training.trainer INFO:   [1] pred='b'                            ref='r'
09:43:40 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:43:40 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9731)
09:44:20 src.training.trainer INFO: Epoch 12/200 — train_loss=2.4639, val_loss=2.9343, val_cer=0.9748, val_wer=0.9559, lr=9.94e-05, time=39.2s
09:44:20 src.training.trainer INFO:   [0] pred='x'                            ref='daisy leaned forward at once horrified and fascina'
09:44:20 src.training.trainer INFO:   [1] pred='x'                            ref='r'
09:44:20 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:44:58 src.training.trainer INFO: Epoch 13/200 — train_loss=2.3516, val_loss=2.8907, val_cer=0.9738, val_wer=0.9609, lr=9.92e-05, time=37.8s
09:44:58 src.training.trainer INFO:   [0] pred='r'                            ref='daisy leaned forward at once horrified and fascina'
09:44:58 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:44:58 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:45:36 src.training.trainer INFO: Epoch 14/200 — train_loss=2.2789, val_loss=2.8297, val_cer=0.9690, val_wer=0.9382, lr=9.91e-05, time=38.4s
09:45:36 src.training.trainer INFO:   [0] pred='y'                            ref='daisy leaned forward at once horrified and fascina'
09:45:36 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:45:36 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:45:37 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9690)
09:46:18 src.training.trainer INFO: Epoch 15/200 — train_loss=2.2088, val_loss=2.8373, val_cer=0.9690, val_wer=0.9344, lr=9.89e-05, time=40.9s
09:46:18 src.training.trainer INFO:   [0] pred='y'                            ref='daisy leaned forward at once horrified and fascina'
09:46:18 src.training.trainer INFO:   [1] pred='p'                            ref='r'
09:46:18 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:46:55 src.training.trainer INFO: Epoch 16/200 — train_loss=2.0924, val_loss=2.8446, val_cer=0.9683, val_wer=0.9344, lr=9.88e-05, time=36.9s
09:46:55 src.training.trainer INFO:   [0] pred='y'                            ref='daisy leaned forward at once horrified and fascina'
09:46:55 src.training.trainer INFO:   [1] pred='h'                            ref='r'
09:46:55 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:46:56 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9683)
09:47:36 src.training.trainer INFO: Epoch 17/200 — train_loss=2.0241, val_loss=2.7839, val_cer=0.9645, val_wer=0.9231, lr=9.86e-05, time=40.5s
09:47:36 src.training.trainer INFO:   [0] pred='y'                            ref='daisy leaned forward at once horrified and fascina'
09:47:36 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:47:36 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:47:37 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9645)
09:48:16 src.training.trainer INFO: Epoch 18/200 — train_loss=1.9183, val_loss=2.8100, val_cer=0.9624, val_wer=0.9180, lr=9.84e-05, time=39.2s
09:48:16 src.training.trainer INFO:   [0] pred='w'                            ref='daisy leaned forward at once horrified and fascina'
09:48:16 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:48:16 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:48:16 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9624)
09:48:56 src.training.trainer INFO: Epoch 19/200 — train_loss=1.8380, val_loss=2.7190, val_cer=0.9593, val_wer=0.9079, lr=9.82e-05, time=39.2s
09:48:56 src.training.trainer INFO:   [0] pred='y'                            ref='daisy leaned forward at once horrified and fascina'
09:48:56 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:48:56 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:48:56 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9593)
09:49:35 src.training.trainer INFO: Epoch 20/200 — train_loss=1.7353, val_loss=2.8015, val_cer=0.9617, val_wer=0.9168, lr=9.80e-05, time=39.0s
09:49:35 src.training.trainer INFO:   [0] pred='w'                            ref='daisy leaned forward at once horrified and fascina'
09:49:35 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:49:35 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:50:15 src.training.trainer INFO: Epoch 21/200 — train_loss=1.6908, val_loss=2.6180, val_cer=0.9524, val_wer=0.8890, lr=9.77e-05, time=39.8s
09:50:15 src.training.trainer INFO:   [0] pred='w'                            ref='daisy leaned forward at once horrified and fascina'
09:50:15 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:50:15 src.training.trainer INFO:   [2] pred='v'                            ref='k'
09:50:16 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9524)
09:50:52 src.training.trainer INFO: Epoch 22/200 — train_loss=1.5826, val_loss=2.5835, val_cer=0.9476, val_wer=0.8625, lr=9.75e-05, time=36.3s
09:50:52 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:50:52 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:50:52 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:50:53 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9476)
09:51:33 src.training.trainer INFO: Epoch 23/200 — train_loss=1.4981, val_loss=2.5067, val_cer=0.9455, val_wer=0.8676, lr=9.72e-05, time=40.2s
09:51:33 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:51:33 src.training.trainer INFO:   [1] pred='z'                            ref='r'
09:51:33 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:51:34 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9455)
09:52:15 src.training.trainer INFO: Epoch 24/200 — train_loss=1.4342, val_loss=2.5166, val_cer=0.9465, val_wer=0.8613, lr=9.70e-05, time=40.6s
09:52:15 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:52:15 src.training.trainer INFO:   [1] pred='m'                            ref='r'
09:52:15 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:52:53 src.training.trainer INFO: Epoch 25/200 — train_loss=1.3659, val_loss=2.4062, val_cer=0.9417, val_wer=0.8487, lr=9.67e-05, time=38.4s
09:52:53 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:52:53 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:52:53 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:52:54 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9417)
09:53:34 src.training.trainer INFO: Epoch 26/200 — train_loss=1.3148, val_loss=2.4556, val_cer=0.9483, val_wer=0.8651, lr=9.64e-05, time=39.7s
09:53:34 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:53:34 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:53:34 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:54:13 src.training.trainer INFO: Epoch 27/200 — train_loss=1.2544, val_loss=2.1979, val_cer=0.9369, val_wer=0.8310, lr=9.61e-05, time=39.0s
09:54:13 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:54:13 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:54:13 src.training.trainer INFO:   [2] pred='w'                            ref='k'
09:54:14 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9369)
09:54:54 src.training.trainer INFO: Epoch 28/200 — train_loss=1.1711, val_loss=2.2460, val_cer=0.9365, val_wer=0.8272, lr=9.58e-05, time=40.6s
09:54:54 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:54:54 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:54:54 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:54:55 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9365)
09:55:33 src.training.trainer INFO: Epoch 29/200 — train_loss=1.1431, val_loss=1.9564, val_cer=0.9276, val_wer=0.7970, lr=9.55e-05, time=38.5s
09:55:33 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:55:33 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:55:33 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:55:34 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9276)
09:56:09 src.training.trainer INFO: Epoch 30/200 — train_loss=1.0213, val_loss=2.1359, val_cer=0.9324, val_wer=0.8146, lr=9.51e-05, time=35.5s
09:56:09 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:56:09 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:56:09 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:56:49 src.training.trainer INFO: Epoch 31/200 — train_loss=1.0121, val_loss=2.0140, val_cer=0.9255, val_wer=0.7957, lr=9.48e-05, time=39.6s
09:56:49 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:56:49 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:56:49 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:56:50 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9255)
09:57:27 src.training.trainer INFO: Epoch 32/200 — train_loss=0.9755, val_loss=1.8382, val_cer=0.9220, val_wer=0.7818, lr=9.44e-05, time=37.1s
09:57:27 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:57:27 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:57:27 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:57:27 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9220)
09:58:02 src.training.trainer INFO: Epoch 33/200 — train_loss=0.8871, val_loss=1.8018, val_cer=0.9203, val_wer=0.7844, lr=9.41e-05, time=35.0s
09:58:02 src.training.trainer INFO:   [0] pred='t'                            ref='daisy leaned forward at once horrified and fascina'
09:58:02 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:58:02 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:58:03 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.9203)
09:58:42 src.training.trainer INFO: Epoch 34/200 — train_loss=0.8351, val_loss=1.8352, val_cer=0.8882, val_wer=0.7907, lr=9.37e-05, time=39.2s
09:58:42 src.training.trainer INFO:   [0] pred='t '                           ref='daisy leaned forward at once horrified and fascina'
09:58:42 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:58:42 src.training.trainer INFO:   [2] pred='w'                            ref='k'
09:58:42 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.8882)
09:59:21 src.training.trainer INFO: Epoch 35/200 — train_loss=0.7686, val_loss=1.6128, val_cer=0.7247, val_wer=0.7705, lr=9.33e-05, time=39.0s
09:59:21 src.training.trainer INFO:   [0] pred='ta aes  e fefsa'              ref='daisy leaned forward at once horrified and fascina'
09:59:21 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:59:21 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:59:22 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.7247)
09:59:58 src.training.trainer INFO: Epoch 36/200 — train_loss=0.7150, val_loss=1.4708, val_cer=0.5730, val_wer=0.7654, lr=9.29e-05, time=36.2s
09:59:58 src.training.trainer INFO:   [0] pred='ta assws a s hifiea ss  eaa'  ref='daisy leaned forward at once horrified and fascina'
09:59:58 src.training.trainer INFO:   [1] pred='r'                            ref='r'
09:59:58 src.training.trainer INFO:   [2] pred='k'                            ref='k'
09:59:59 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.5730)
10:00:35 src.training.trainer INFO: Epoch 37/200 — train_loss=0.6557, val_loss=1.3267, val_cer=0.6126, val_wer=0.7289, lr=9.25e-05, time=36.6s
10:00:35 src.training.trainer INFO:   [0] pred='tars adfd   hif a ffud o'     ref='daisy leaned forward at once horrified and fascina'
10:00:35 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:00:35 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:01:14 src.training.trainer INFO: Epoch 38/200 — train_loss=0.6051, val_loss=1.1832, val_cer=0.4560, val_wer=0.7100, lr=9.20e-05, time=38.6s
10:01:14 src.training.trainer INFO:   [0] pred='tary hansdswosaf nes moinsid ane fafrmd eo' ref='daisy leaned forward at once horrified and fascina'
10:01:14 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:01:14 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:01:15 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.4560)
10:01:51 src.training.trainer INFO: Epoch 39/200 — train_loss=0.5540, val_loss=1.1072, val_cer=0.4257, val_wer=0.7238, lr=9.16e-05, time=36.7s
10:01:51 src.training.trainer INFO:   [0] pred='oarisy heanad swmmcdl af oe hnisicd  anns fas md e' ref='daisy leaned forward at once horrified and fascina'
10:01:51 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:01:51 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:01:52 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.4257)
10:02:27 src.training.trainer INFO: Epoch 40/200 — train_loss=0.5260, val_loss=1.0853, val_cer=0.3884, val_wer=0.6797, lr=9.12e-05, time=35.5s
10:02:27 src.training.trainer INFO:   [0] pred='oarsy heansd fwnrdlt at ones hointifisd ann fes in' ref='daisy leaned forward at once horrified and fascina'
10:02:27 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:02:27 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:02:28 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.3884)
10:03:03 src.training.trainer INFO: Epoch 41/200 — train_loss=0.4513, val_loss=1.0883, val_cer=0.3587, val_wer=0.6507, lr=9.07e-05, time=35.0s
10:03:03 src.training.trainer INFO:   [0] pred='tharsy heared fooracid af oee hnifiisd arne  faasa' ref='daisy leaned forward at once horrified and fascina'
10:03:03 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:03:03 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:03:04 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.3587)
10:03:42 src.training.trainer INFO: Epoch 42/200 — train_loss=0.4631, val_loss=1.0611, val_cer=0.3208, val_wer=0.6066, lr=9.02e-05, time=38.3s
10:03:42 src.training.trainer INFO:   [0] pred='oharsy heared sworccd at ones hornisiicd ar  fsine' ref='daisy leaned forward at once horrified and fascina'
10:03:42 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:03:42 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:03:43 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.3208)
10:04:22 src.training.trainer INFO: Epoch 43/200 — train_loss=0.4459, val_loss=0.9621, val_cer=0.3049, val_wer=0.6053, lr=8.98e-05, time=39.1s
10:04:22 src.training.trainer INFO:   [0] pred='oaisy hearsd forccrd  at onee hounrificd ard fasci' ref='daisy leaned forward at once horrified and fascina'
10:04:22 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:04:22 src.training.trainer INFO:   [2] pred='w'                            ref='k'
10:04:22 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.3049)
10:05:01 src.training.trainer INFO: Epoch 44/200 — train_loss=0.4076, val_loss=0.9698, val_cer=0.3329, val_wer=0.6141, lr=8.93e-05, time=39.1s
10:05:01 src.training.trainer INFO:   [0] pred='oarsy heonad  foorworndin of onee hoourifisd andr ' ref='daisy leaned forward at once horrified and fascina'
10:05:01 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:05:01 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:05:41 src.training.trainer INFO: Epoch 45/200 — train_loss=0.3602, val_loss=0.8740, val_cer=0.2935, val_wer=0.5801, lr=8.88e-05, time=39.4s
10:05:41 src.training.trainer INFO:   [0] pred='thaisy heanad  foonacindt at onee honrifiied ana f' ref='daisy leaned forward at once horrified and fascina'
10:05:41 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:05:41 src.training.trainer INFO:   [2] pred='j'                            ref='k'
10:05:42 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.2935)
10:06:22 src.training.trainer INFO: Epoch 46/200 — train_loss=0.3500, val_loss=0.9742, val_cer=0.2587, val_wer=0.5523, lr=8.83e-05, time=40.1s
10:06:22 src.training.trainer INFO:   [0] pred='caisy hearad foorwcrd af once hourifiisd ane fasci' ref='daisy leaned forward at once horrified and fascina'
10:06:22 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:06:22 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:06:22 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.2587)
10:07:01 src.training.trainer INFO: Epoch 47/200 — train_loss=0.3399, val_loss=0.8936, val_cer=0.2587, val_wer=0.5662, lr=8.78e-05, time=38.5s
10:07:01 src.training.trainer INFO:   [0] pred='oaisy heanerd forccrrd at onee hourriifiisd an fas' ref='daisy leaned forward at once horrified and fascina'
10:07:01 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:07:01 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:07:39 src.training.trainer INFO: Epoch 48/200 — train_loss=0.2992, val_loss=0.9101, val_cer=0.2487, val_wer=0.5485, lr=8.72e-05, time=38.0s
10:07:39 src.training.trainer INFO:   [0] pred='taisy heansrd foontcindit at onee hormmifiisd ane ' ref='daisy leaned forward at once horrified and fascina'
10:07:39 src.training.trainer INFO:   [1] pred='n'                            ref='r'
10:07:39 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:07:40 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.2487)
10:08:20 src.training.trainer INFO: Epoch 49/200 — train_loss=0.2937, val_loss=0.9077, val_cer=0.2463, val_wer=0.5435, lr=8.67e-05, time=40.4s
10:08:20 src.training.trainer INFO:   [0] pred='daisy haanad fwonwcndi at onae hornifiisd an fasai' ref='daisy leaned forward at once horrified and fascina'
10:08:20 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:08:20 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:08:21 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.2463)
10:08:58 src.training.trainer INFO: Epoch 50/200 — train_loss=0.2685, val_loss=0.7622, val_cer=0.2235, val_wer=0.5032, lr=8.62e-05, time=37.4s
10:08:58 src.training.trainer INFO:   [0] pred='caisy hecned  foonccrndi at once houmrifiicd  anc ' ref='daisy leaned forward at once horrified and fascina'
10:08:58 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:08:58 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:08:59 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.2235)
10:09:39 src.training.trainer INFO: Epoch 51/200 — train_loss=0.2534, val_loss=0.7705, val_cer=0.2087, val_wer=0.4779, lr=8.56e-05, time=39.8s
10:09:39 src.training.trainer INFO:   [0] pred='caisy hearad foorcrnd at once hourrifiisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:09:39 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:09:39 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:09:39 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.2087)
10:10:16 src.training.trainer INFO: Epoch 52/200 — train_loss=0.2313, val_loss=0.7736, val_cer=0.2063, val_wer=0.4691, lr=8.50e-05, time=36.7s
10:10:16 src.training.trainer INFO:   [0] pred='oaisy hearad forcrdil af once horrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:10:16 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:10:16 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:10:16 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.2063)
10:10:56 src.training.trainer INFO: Epoch 53/200 — train_loss=0.2373, val_loss=0.8121, val_cer=0.1963, val_wer=0.4704, lr=8.45e-05, time=39.8s
10:10:56 src.training.trainer INFO:   [0] pred='daisy lecned  foorcnd ot once hourrifisd  and fasc' ref='daisy leaned forward at once horrified and fascina'
10:10:56 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:10:56 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:10:57 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1963)
10:11:32 src.training.trainer INFO: Epoch 54/200 — train_loss=0.1957, val_loss=0.7334, val_cer=0.1783, val_wer=0.4237, lr=8.39e-05, time=35.7s
10:11:32 src.training.trainer INFO:   [0] pred='daisy learad  foorward at orce horrrifisd ard fasc' ref='daisy leaned forward at once horrified and fascina'
10:11:32 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:11:32 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:11:33 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1783)
10:12:11 src.training.trainer INFO: Epoch 55/200 — train_loss=0.1866, val_loss=0.7701, val_cer=0.1652, val_wer=0.4149, lr=8.33e-05, time=38.3s
10:12:11 src.training.trainer INFO:   [0] pred='daisy learad foorwand at once hourrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:12:11 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:12:11 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:12:12 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1652)
10:12:50 src.training.trainer INFO: Epoch 56/200 — train_loss=0.2119, val_loss=0.8647, val_cer=0.1594, val_wer=0.4136, lr=8.27e-05, time=37.7s
10:12:50 src.training.trainer INFO:   [0] pred='caisy learad forwandk at once hourrifiisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:12:50 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:12:50 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:12:50 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1594)
10:13:26 src.training.trainer INFO: Epoch 57/200 — train_loss=0.2062, val_loss=0.7281, val_cer=0.1549, val_wer=0.3947, lr=8.21e-05, time=36.1s
10:13:26 src.training.trainer INFO:   [0] pred='dlaisy learad foorwandik at once hourrifiisd and f' ref='daisy leaned forward at once horrified and fascina'
10:13:26 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:13:26 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:13:27 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1549)
10:14:02 src.training.trainer INFO: Epoch 58/200 — train_loss=0.1860, val_loss=0.7320, val_cer=0.1507, val_wer=0.3922, lr=8.15e-05, time=35.5s
10:14:02 src.training.trainer INFO:   [0] pred='daisy leanred forwcnd at once hourrifisd and fscin' ref='daisy leaned forward at once horrified and fascina'
10:14:02 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:14:02 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:14:03 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1507)
10:14:40 src.training.trainer INFO: Epoch 59/200 — train_loss=0.1811, val_loss=0.7284, val_cer=0.1414, val_wer=0.3695, lr=8.09e-05, time=36.7s
10:14:40 src.training.trainer INFO:   [0] pred='daisy leanad fonwandk at once hourrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:14:40 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:14:40 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:14:40 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1414)
10:15:19 src.training.trainer INFO: Epoch 60/200 — train_loss=0.1615, val_loss=0.6480, val_cer=0.1338, val_wer=0.3594, lr=8.03e-05, time=38.6s
10:15:19 src.training.trainer INFO:   [0] pred='daisy learad forwarrdik at once hourrifiisd  ard f' ref='daisy leaned forward at once horrified and fascina'
10:15:19 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:15:19 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:15:19 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1338)
10:15:56 src.training.trainer INFO: Epoch 61/200 — train_loss=0.1667, val_loss=0.7242, val_cer=0.1325, val_wer=0.3506, lr=7.96e-05, time=37.2s
10:15:56 src.training.trainer INFO:   [0] pred='daisy hearad forwandik at once hourrifisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:15:56 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:15:56 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:15:57 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1325)
10:16:37 src.training.trainer INFO: Epoch 62/200 — train_loss=0.1662, val_loss=0.6789, val_cer=0.1214, val_wer=0.3291, lr=7.90e-05, time=39.9s
10:16:37 src.training.trainer INFO:   [0] pred='daisy leared forwarrdk at once hourrifisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:16:37 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:16:37 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:16:37 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1214)
10:17:14 src.training.trainer INFO: Epoch 63/200 — train_loss=0.1361, val_loss=0.6789, val_cer=0.1131, val_wer=0.3291, lr=7.83e-05, time=36.1s
10:17:14 src.training.trainer INFO:   [0] pred='daisy heanad forwandk at once horrrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:17:14 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:17:14 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:17:14 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1131)
10:17:49 src.training.trainer INFO: Epoch 64/200 — train_loss=0.1294, val_loss=0.7088, val_cer=0.1111, val_wer=0.3178, lr=7.77e-05, time=35.2s
10:17:49 src.training.trainer INFO:   [0] pred='daisy leanad forwandk at once hourrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:17:49 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:17:49 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:17:50 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1111)
10:18:28 src.training.trainer INFO: Epoch 65/200 — train_loss=0.1438, val_loss=0.6297, val_cer=0.1111, val_wer=0.3203, lr=7.70e-05, time=37.9s
10:18:28 src.training.trainer INFO:   [0] pred='daisy leanad forwandk ot once hourrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:18:28 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:18:28 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:19:04 src.training.trainer INFO: Epoch 66/200 — train_loss=0.1301, val_loss=0.6948, val_cer=0.1152, val_wer=0.3304, lr=7.63e-05, time=35.6s
10:19:04 src.training.trainer INFO:   [0] pred='daisy leanad forwardk at once hourrifiisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:19:04 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:19:04 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:19:44 src.training.trainer INFO: Epoch 67/200 — train_loss=0.1349, val_loss=0.6957, val_cer=0.1069, val_wer=0.2850, lr=7.57e-05, time=40.6s
10:19:44 src.training.trainer INFO:   [0] pred='daisy leaned forwvandk at once hourrifisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:19:44 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:19:44 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:19:45 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.1069)
10:20:20 src.training.trainer INFO: Epoch 68/200 — train_loss=0.1151, val_loss=0.6235, val_cer=0.1076, val_wer=0.3064, lr=7.50e-05, time=35.2s
10:20:20 src.training.trainer INFO:   [0] pred='daisy leanad forwondk at once hourrifiisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:20:20 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:20:20 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:20:55 src.training.trainer INFO: Epoch 69/200 — train_loss=0.1125, val_loss=0.6149, val_cer=0.0976, val_wer=0.2863, lr=7.43e-05, time=34.7s
10:20:55 src.training.trainer INFO:   [0] pred='daisy leanad forwardk at once horrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:20:55 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:20:55 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:20:56 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0976)
10:21:35 src.training.trainer INFO: Epoch 70/200 — train_loss=0.1171, val_loss=0.5626, val_cer=0.0966, val_wer=0.2863, lr=7.36e-05, time=38.8s
10:21:35 src.training.trainer INFO:   [0] pred='daisy leaned forwandk ot once horrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:21:35 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:21:35 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:21:35 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0966)
10:22:12 src.training.trainer INFO: Epoch 71/200 — train_loss=0.0993, val_loss=0.5732, val_cer=0.0869, val_wer=0.2585, lr=7.29e-05, time=36.4s
10:22:12 src.training.trainer INFO:   [0] pred='daisy leaned  forcard at once horrifisd  and fasci' ref='daisy leaned forward at once horrified and fascina'
10:22:12 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:22:12 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:22:12 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0869)
10:22:51 src.training.trainer INFO: Epoch 72/200 — train_loss=0.1146, val_loss=0.6482, val_cer=0.0949, val_wer=0.2863, lr=7.22e-05, time=38.8s
10:22:51 src.training.trainer INFO:   [0] pred='daisy heaned forward at once horrified ar fascinat' ref='daisy leaned forward at once horrified and fascina'
10:22:51 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:22:51 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:23:27 src.training.trainer INFO: Epoch 73/200 — train_loss=0.0909, val_loss=0.5627, val_cer=0.0845, val_wer=0.2673, lr=7.15e-05, time=36.1s
10:23:27 src.training.trainer INFO:   [0] pred='daisy leaned forward at once hourrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:23:27 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:23:27 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:23:28 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0845)
10:24:05 src.training.trainer INFO: Epoch 74/200 — train_loss=0.1049, val_loss=0.5759, val_cer=0.0852, val_wer=0.2585, lr=7.07e-05, time=37.1s
10:24:05 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrifiisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:24:05 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:24:05 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:24:45 src.training.trainer INFO: Epoch 75/200 — train_loss=0.1019, val_loss=0.5846, val_cer=0.0890, val_wer=0.2585, lr=7.00e-05, time=39.4s
10:24:45 src.training.trainer INFO:   [0] pred='daisy lecned forcarrd at once horrrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:24:45 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:24:45 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:25:21 src.training.trainer INFO: Epoch 76/200 — train_loss=0.0947, val_loss=0.5414, val_cer=0.0862, val_wer=0.2598, lr=6.93e-05, time=35.8s
10:25:21 src.training.trainer INFO:   [0] pred='daisy leaned forwardx at once horrified ann fascin' ref='daisy leaned forward at once horrified and fascina'
10:25:21 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:25:21 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:25:57 src.training.trainer INFO: Epoch 77/200 — train_loss=0.0909, val_loss=0.4916, val_cer=0.0866, val_wer=0.2573, lr=6.86e-05, time=36.5s
10:25:57 src.training.trainer INFO:   [0] pred='daisy heanad forwardx at once hourrifisd arn fasci' ref='daisy leaned forward at once horrified and fascina'
10:25:57 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:25:57 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:26:33 src.training.trainer INFO: Epoch 78/200 — train_loss=0.0797, val_loss=0.5375, val_cer=0.0793, val_wer=0.2421, lr=6.78e-05, time=35.3s
10:26:33 src.training.trainer INFO:   [0] pred='daisy leanad forwarrdk at once hourrifisd ard fasc' ref='daisy leaned forward at once horrified and fascina'
10:26:33 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:26:33 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:26:34 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0793)
10:27:09 src.training.trainer INFO: Epoch 79/200 — train_loss=0.0877, val_loss=0.6065, val_cer=0.0821, val_wer=0.2484, lr=6.71e-05, time=35.9s
10:27:09 src.training.trainer INFO:   [0] pred='daisy heanad forward at once hoxrrifisd ann fascin' ref='daisy leaned forward at once horrified and fascina'
10:27:09 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:27:09 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:27:48 src.training.trainer INFO: Epoch 80/200 — train_loss=0.0785, val_loss=0.5879, val_cer=0.0797, val_wer=0.2383, lr=6.63e-05, time=38.5s
10:27:48 src.training.trainer INFO:   [0] pred='daisy leaned forward at once hourrifisd anr fascin' ref='daisy leaned forward at once horrified and fascina'
10:27:48 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:27:48 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:28:29 src.training.trainer INFO: Epoch 81/200 — train_loss=0.0787, val_loss=0.5078, val_cer=0.0814, val_wer=0.2434, lr=6.56e-05, time=40.5s
10:28:29 src.training.trainer INFO:   [0] pred='daisy leanad forward at once hourrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:28:29 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:28:29 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:29:05 src.training.trainer INFO: Epoch 82/200 — train_loss=0.0848, val_loss=0.4868, val_cer=0.0714, val_wer=0.2219, lr=6.48e-05, time=36.1s
10:29:05 src.training.trainer INFO:   [0] pred='daisy leanad forwarrd at once hoxrrifiisd ard fasc' ref='daisy leaned forward at once horrified and fascina'
10:29:05 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:29:05 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:29:06 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0714)
10:29:45 src.training.trainer INFO: Epoch 83/200 — train_loss=0.0795, val_loss=0.5585, val_cer=0.0800, val_wer=0.2484, lr=6.41e-05, time=39.5s
10:29:45 src.training.trainer INFO:   [0] pred='daisy leanad forward at once hoarrifiisd ard fasci' ref='daisy leaned forward at once horrified and fascina'
10:29:45 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:29:45 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:30:26 src.training.trainer INFO: Epoch 84/200 — train_loss=0.0810, val_loss=0.4936, val_cer=0.0717, val_wer=0.2282, lr=6.33e-05, time=40.6s
10:30:26 src.training.trainer INFO:   [0] pred='daisy leanad forward at once horrrifisd ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:30:26 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:30:26 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:31:05 src.training.trainer INFO: Epoch 85/200 — train_loss=0.0710, val_loss=0.4818, val_cer=0.0762, val_wer=0.2232, lr=6.25e-05, time=38.5s
10:31:05 src.training.trainer INFO:   [0] pred='daisy leaned forwardk at once horrrifiisd and fasc' ref='daisy leaned forward at once horrified and fascina'
10:31:05 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:31:05 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:31:46 src.training.trainer INFO: Epoch 86/200 — train_loss=0.0658, val_loss=0.5715, val_cer=0.0773, val_wer=0.2320, lr=6.18e-05, time=40.4s
10:31:46 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrifisd ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:31:46 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:31:46 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:32:22 src.training.trainer INFO: Epoch 87/200 — train_loss=0.0647, val_loss=0.5978, val_cer=0.0697, val_wer=0.2207, lr=6.10e-05, time=36.2s
10:32:22 src.training.trainer INFO:   [0] pred='daisy heaned forward at once horrifiied ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:32:22 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:32:22 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:32:23 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0697)
10:33:00 src.training.trainer INFO: Epoch 88/200 — train_loss=0.0647, val_loss=0.5405, val_cer=0.0655, val_wer=0.2043, lr=6.02e-05, time=36.9s
10:33:00 src.training.trainer INFO:   [0] pred='daisy heaned forwardx at once horrifiisd avd fasci' ref='daisy leaned forward at once horrified and fascina'
10:33:00 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:33:00 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:33:00 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0655)
10:33:39 src.training.trainer INFO: Epoch 89/200 — train_loss=0.0656, val_loss=0.5873, val_cer=0.0780, val_wer=0.2446, lr=5.94e-05, time=38.9s
10:33:39 src.training.trainer INFO:   [0] pred='daisy leaned forward at once hourrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:33:39 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:33:39 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:34:16 src.training.trainer INFO: Epoch 90/200 — train_loss=0.0861, val_loss=0.4997, val_cer=0.0673, val_wer=0.2144, lr=5.87e-05, time=36.1s
10:34:16 src.training.trainer INFO:   [0] pred='daisy heaned forward at once hoxrrified and fascin' ref='daisy leaned forward at once horrified and fascina'
10:34:16 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:34:16 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:34:53 src.training.trainer INFO: Epoch 91/200 — train_loss=0.0573, val_loss=0.5830, val_cer=0.0666, val_wer=0.2093, lr=5.79e-05, time=37.5s
10:34:54 src.training.trainer INFO:   [0] pred='daisy heaned forwardx at once horrifid and fascina' ref='daisy leaned forward at once horrified and fascina'
10:34:54 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:34:54 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:35:32 src.training.trainer INFO: Epoch 92/200 — train_loss=0.0594, val_loss=0.5593, val_cer=0.0680, val_wer=0.2106, lr=5.71e-05, time=38.5s
10:35:32 src.training.trainer INFO:   [0] pred='daisy heaned forwardx at once hourrified ard fasci' ref='daisy leaned forward at once horrified and fascina'
10:35:32 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:35:32 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:36:13 src.training.trainer INFO: Epoch 93/200 — train_loss=0.0638, val_loss=0.5186, val_cer=0.0683, val_wer=0.2169, lr=5.63e-05, time=40.1s
10:36:13 src.training.trainer INFO:   [0] pred='daisy leaned forwardx at once horrifiicd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:36:13 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:36:13 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:36:52 src.training.trainer INFO: Epoch 94/200 — train_loss=0.0595, val_loss=0.4706, val_cer=0.0638, val_wer=0.2018, lr=5.55e-05, time=39.6s
10:36:52 src.training.trainer INFO:   [0] pred='daisy heaned forwardx at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
10:36:52 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:36:52 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:36:53 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0638)
10:37:30 src.training.trainer INFO: Epoch 95/200 — train_loss=0.0511, val_loss=0.6676, val_cer=0.0638, val_wer=0.2043, lr=5.47e-05, time=36.6s
10:37:30 src.training.trainer INFO:   [0] pred='daisy heanad forwardk at once horrrified and fasci' ref='daisy leaned forward at once horrified and fascina'
10:37:30 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:37:30 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:38:06 src.training.trainer INFO: Epoch 96/200 — train_loss=0.0485, val_loss=0.3965, val_cer=0.0614, val_wer=0.1917, lr=5.39e-05, time=36.3s
10:38:06 src.training.trainer INFO:   [0] pred='daisy heaned forwardx at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
10:38:06 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:38:06 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:38:07 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0614)
10:38:46 src.training.trainer INFO: Epoch 97/200 — train_loss=0.0671, val_loss=0.5377, val_cer=0.0673, val_wer=0.2081, lr=5.31e-05, time=38.8s
10:38:46 src.training.trainer INFO:   [0] pred='daisy heanad forward at once horrified ann fascina' ref='daisy leaned forward at once horrified and fascina'
10:38:46 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:38:46 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:39:22 src.training.trainer INFO: Epoch 98/200 — train_loss=0.0482, val_loss=0.5757, val_cer=0.0604, val_wer=0.2018, lr=5.24e-05, time=35.6s
10:39:22 src.training.trainer INFO:   [0] pred='daisy leaned forwardx at once hourrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:39:22 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:39:22 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:39:22 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0604)
10:40:01 src.training.trainer INFO: Epoch 99/200 — train_loss=0.0577, val_loss=0.4721, val_cer=0.0624, val_wer=0.1980, lr=5.16e-05, time=38.3s
10:40:01 src.training.trainer INFO:   [0] pred='daisy heanad forward at once hoxrrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:40:01 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:40:01 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:40:41 src.training.trainer INFO: Epoch 100/200 — train_loss=0.0403, val_loss=0.4535, val_cer=0.0628, val_wer=0.2005, lr=5.08e-05, time=40.3s
10:40:41 src.training.trainer INFO:   [0] pred='daisy heaned forwardx at once horrrifisd and fasci' ref='daisy leaned forward at once horrified and fascina'
10:40:41 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:40:41 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:41:20 src.training.trainer INFO: Epoch 101/200 — train_loss=0.0400, val_loss=0.5634, val_cer=0.0621, val_wer=0.2018, lr=5.00e-05, time=38.9s
10:41:20 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrifised and fasci' ref='daisy leaned forward at once horrified and fascina'
10:41:20 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:41:20 src.training.trainer INFO:   [2] pred='n'                            ref='k'
10:41:58 src.training.trainer INFO: Epoch 102/200 — train_loss=0.0394, val_loss=0.5713, val_cer=0.0614, val_wer=0.1992, lr=4.92e-05, time=37.4s
10:41:58 src.training.trainer INFO:   [0] pred='daisy leaned forwardx at once horrrified and fasci' ref='daisy leaned forward at once horrified and fascina'
10:41:58 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:41:58 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:42:36 src.training.trainer INFO: Epoch 103/200 — train_loss=0.0407, val_loss=0.4775, val_cer=0.0659, val_wer=0.2068, lr=4.84e-05, time=37.3s
10:42:36 src.training.trainer INFO:   [0] pred='daisy leaned forward at once hourrified and fascin' ref='daisy leaned forward at once horrified and fascina'
10:42:36 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:42:36 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:43:12 src.training.trainer INFO: Epoch 104/200 — train_loss=0.0535, val_loss=0.4237, val_cer=0.0552, val_wer=0.1728, lr=4.76e-05, time=36.0s
10:43:12 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:43:12 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:43:12 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:43:12 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0552)
10:43:51 src.training.trainer INFO: Epoch 105/200 — train_loss=0.0539, val_loss=0.5712, val_cer=0.0548, val_wer=0.1753, lr=4.68e-05, time=38.5s
10:43:51 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified ard fascina' ref='daisy leaned forward at once horrified and fascina'
10:43:51 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:43:51 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:43:52 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0548)
10:44:30 src.training.trainer INFO: Epoch 106/200 — train_loss=0.0518, val_loss=0.4735, val_cer=0.0562, val_wer=0.1753, lr=4.60e-05, time=38.5s
10:44:30 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:44:30 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:44:30 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:45:09 src.training.trainer INFO: Epoch 107/200 — train_loss=0.0465, val_loss=0.4537, val_cer=0.0576, val_wer=0.1866, lr=4.52e-05, time=38.9s
10:45:09 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:45:09 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:45:09 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:45:49 src.training.trainer INFO: Epoch 108/200 — train_loss=0.0433, val_loss=0.4918, val_cer=0.0573, val_wer=0.1866, lr=4.44e-05, time=39.8s
10:45:49 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:45:49 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:45:49 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:46:29 src.training.trainer INFO: Epoch 109/200 — train_loss=0.0375, val_loss=0.5101, val_cer=0.0552, val_wer=0.1778, lr=4.36e-05, time=39.8s
10:46:29 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:46:29 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:46:29 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:47:06 src.training.trainer INFO: Epoch 110/200 — train_loss=0.0336, val_loss=0.4212, val_cer=0.0531, val_wer=0.1715, lr=4.29e-05, time=35.9s
10:47:06 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:47:06 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:47:06 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:47:06 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0531)
10:47:42 src.training.trainer INFO: Epoch 111/200 — train_loss=0.0338, val_loss=0.5174, val_cer=0.0531, val_wer=0.1690, lr=4.21e-05, time=36.3s
10:47:42 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:47:42 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:47:42 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:48:19 src.training.trainer INFO: Epoch 112/200 — train_loss=0.0495, val_loss=0.5480, val_cer=0.0583, val_wer=0.1828, lr=4.13e-05, time=36.7s
10:48:19 src.training.trainer INFO:   [0] pred='daisy leaned forward at once hrrified and fascinat' ref='daisy leaned forward at once horrified and fascina'
10:48:19 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:48:19 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:48:59 src.training.trainer INFO: Epoch 113/200 — train_loss=0.0465, val_loss=0.5635, val_cer=0.0548, val_wer=0.1765, lr=4.05e-05, time=39.1s
10:48:59 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified arr fascin' ref='daisy leaned forward at once horrified and fascina'
10:48:59 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:48:59 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:49:35 src.training.trainer INFO: Epoch 114/200 — train_loss=0.0314, val_loss=0.4507, val_cer=0.0524, val_wer=0.1702, lr=3.97e-05, time=35.7s
10:49:35 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
10:49:35 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:49:35 src.training.trainer INFO:   [2] pred='y'                            ref='k'
10:49:35 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0524)
10:50:13 src.training.trainer INFO: Epoch 115/200 — train_loss=0.0383, val_loss=0.5477, val_cer=0.0538, val_wer=0.1740, lr=3.90e-05, time=37.3s
10:50:13 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:50:13 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:50:13 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:50:52 src.training.trainer INFO: Epoch 116/200 — train_loss=0.0282, val_loss=0.5427, val_cer=0.0528, val_wer=0.1740, lr=3.82e-05, time=38.9s
10:50:52 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrifisd and fascin' ref='daisy leaned forward at once horrified and fascina'
10:50:52 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:50:52 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:51:31 src.training.trainer INFO: Epoch 117/200 — train_loss=0.0343, val_loss=0.5081, val_cer=0.0524, val_wer=0.1753, lr=3.74e-05, time=39.2s
10:51:31 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:51:31 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:51:31 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:52:09 src.training.trainer INFO: Epoch 118/200 — train_loss=0.0363, val_loss=0.5576, val_cer=0.0531, val_wer=0.1702, lr=3.67e-05, time=37.1s
10:52:09 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:52:09 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:52:09 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:52:47 src.training.trainer INFO: Epoch 119/200 — train_loss=0.0275, val_loss=0.5315, val_cer=0.0531, val_wer=0.1665, lr=3.59e-05, time=38.4s
10:52:47 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:52:47 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:52:47 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:53:26 src.training.trainer INFO: Epoch 120/200 — train_loss=0.0245, val_loss=0.5120, val_cer=0.0576, val_wer=0.1866, lr=1.76e-05, time=38.1s
10:53:26 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:53:26 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:53:26 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:54:04 src.training.trainer INFO: Epoch 121/200 — train_loss=0.0262, val_loss=0.4626, val_cer=0.0524, val_wer=0.1665, lr=3.44e-05, time=38.3s
10:54:04 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:54:04 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:54:04 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:54:46 src.training.trainer INFO: Epoch 122/200 — train_loss=0.0370, val_loss=0.4794, val_cer=0.0524, val_wer=0.1652, lr=3.36e-05, time=40.9s
10:54:46 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified and fascin' ref='daisy leaned forward at once horrified and fascina'
10:54:46 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:54:46 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:55:23 src.training.trainer INFO: Epoch 123/200 — train_loss=0.0341, val_loss=0.5587, val_cer=0.0562, val_wer=0.1753, lr=3.29e-05, time=36.7s
10:55:23 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified ard fascin' ref='daisy leaned forward at once horrified and fascina'
10:55:23 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:55:23 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:56:02 src.training.trainer INFO: Epoch 124/200 — train_loss=0.0353, val_loss=0.5715, val_cer=0.0538, val_wer=0.1753, lr=3.21e-05, time=39.1s
10:56:02 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:56:02 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:56:02 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:56:42 src.training.trainer INFO: Epoch 125/200 — train_loss=0.0292, val_loss=0.4633, val_cer=0.0521, val_wer=0.1589, lr=3.14e-05, time=39.6s
10:56:42 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified ard fascina' ref='daisy leaned forward at once horrified and fascina'
10:56:42 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:56:42 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:56:43 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0521)
10:57:20 src.training.trainer INFO: Epoch 126/200 — train_loss=0.0301, val_loss=0.4202, val_cer=0.0517, val_wer=0.1652, lr=3.07e-05, time=37.1s
10:57:20 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:57:20 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:57:20 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:57:20 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0517)
10:57:59 src.training.trainer INFO: Epoch 127/200 — train_loss=0.0306, val_loss=0.4646, val_cer=0.0490, val_wer=0.1652, lr=2.99e-05, time=38.8s
10:57:59 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:57:59 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:57:59 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:58:00 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0490)
10:58:41 src.training.trainer INFO: Epoch 128/200 — train_loss=0.0413, val_loss=0.5477, val_cer=0.0507, val_wer=0.1602, lr=2.92e-05, time=41.0s
10:58:41 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:58:41 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:58:41 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:59:20 src.training.trainer INFO: Epoch 129/200 — train_loss=0.0291, val_loss=0.4940, val_cer=0.0528, val_wer=0.1652, lr=2.85e-05, time=39.2s
10:59:20 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:59:20 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:59:20 src.training.trainer INFO:   [2] pred='k'                            ref='k'
10:59:59 src.training.trainer INFO: Epoch 130/200 — train_loss=0.0359, val_loss=0.5049, val_cer=0.0542, val_wer=0.1702, lr=2.78e-05, time=39.0s
10:59:59 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
10:59:59 src.training.trainer INFO:   [1] pred='r'                            ref='r'
10:59:59 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:00:36 src.training.trainer INFO: Epoch 131/200 — train_loss=0.0218, val_loss=0.4641, val_cer=0.0524, val_wer=0.1690, lr=2.71e-05, time=36.0s
11:00:36 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified ard fascina' ref='daisy leaned forward at once horrified and fascina'
11:00:36 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:00:36 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:01:13 src.training.trainer INFO: Epoch 132/200 — train_loss=0.0230, val_loss=0.4808, val_cer=0.0493, val_wer=0.1538, lr=2.64e-05, time=37.1s
11:01:13 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:01:13 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:01:13 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:01:49 src.training.trainer INFO: Epoch 133/200 — train_loss=0.0232, val_loss=0.4699, val_cer=0.0476, val_wer=0.1526, lr=2.57e-05, time=35.3s
11:01:49 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified ard fascin' ref='daisy leaned forward at once horrified and fascina'
11:01:49 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:01:49 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:01:49 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0476)
11:02:28 src.training.trainer INFO: Epoch 134/200 — train_loss=0.0279, val_loss=0.4936, val_cer=0.0504, val_wer=0.1602, lr=2.50e-05, time=38.7s
11:02:28 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:02:28 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:02:28 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:03:03 src.training.trainer INFO: Epoch 135/200 — train_loss=0.0290, val_loss=0.4503, val_cer=0.0517, val_wer=0.1677, lr=2.43e-05, time=35.2s
11:03:03 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:03:03 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:03:03 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:03:40 src.training.trainer INFO: Epoch 136/200 — train_loss=0.0168, val_loss=0.4664, val_cer=0.0497, val_wer=0.1564, lr=2.36e-05, time=36.9s
11:03:40 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:03:40 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:03:40 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:04:17 src.training.trainer INFO: Epoch 137/200 — train_loss=0.0191, val_loss=0.4627, val_cer=0.0497, val_wer=0.1589, lr=2.30e-05, time=36.0s
11:04:17 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:04:17 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:04:17 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:04:55 src.training.trainer INFO: Epoch 138/200 — train_loss=0.0181, val_loss=0.4278, val_cer=0.0531, val_wer=0.1639, lr=2.23e-05, time=37.8s
11:04:55 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:04:55 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:04:55 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:05:34 src.training.trainer INFO: Epoch 139/200 — train_loss=0.0228, val_loss=0.4959, val_cer=0.0497, val_wer=0.1576, lr=1.08e-05, time=39.4s
11:05:34 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:05:34 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:05:34 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:06:12 src.training.trainer INFO: Epoch 140/200 — train_loss=0.0183, val_loss=0.4887, val_cer=0.0514, val_wer=0.1639, lr=2.10e-05, time=37.5s
11:06:12 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:06:12 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:06:12 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:06:52 src.training.trainer INFO: Epoch 141/200 — train_loss=0.0167, val_loss=0.4971, val_cer=0.0483, val_wer=0.1526, lr=2.03e-05, time=39.6s
11:06:52 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:06:52 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:06:52 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:07:28 src.training.trainer INFO: Epoch 142/200 — train_loss=0.0218, val_loss=0.4961, val_cer=0.0511, val_wer=0.1627, lr=1.97e-05, time=35.2s
11:07:28 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:07:28 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:07:28 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:08:04 src.training.trainer INFO: Epoch 143/200 — train_loss=0.0202, val_loss=0.4140, val_cer=0.0511, val_wer=0.1589, lr=1.91e-05, time=36.4s
11:08:04 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrrified ard fascin' ref='daisy leaned forward at once horrified and fascina'
11:08:04 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:08:04 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:08:43 src.training.trainer INFO: Epoch 144/200 — train_loss=0.0132, val_loss=0.4765, val_cer=0.0483, val_wer=0.1551, lr=1.85e-05, time=38.1s
11:08:43 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified ard fascina' ref='daisy leaned forward at once horrified and fascina'
11:08:43 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:08:43 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:09:24 src.training.trainer INFO: Epoch 145/200 — train_loss=0.0147, val_loss=0.4731, val_cer=0.0448, val_wer=0.1463, lr=1.79e-05, time=40.7s
11:09:24 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:09:24 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:09:24 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:09:24 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0448)
11:10:01 src.training.trainer INFO: Epoch 146/200 — train_loss=0.0272, val_loss=0.4701, val_cer=0.0473, val_wer=0.1513, lr=1.72e-05, time=36.6s
11:10:01 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:10:01 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:10:01 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:10:39 src.training.trainer INFO: Epoch 147/200 — train_loss=0.0192, val_loss=0.4860, val_cer=0.0507, val_wer=0.1589, lr=1.67e-05, time=38.1s
11:10:39 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:10:39 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:10:39 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:11:18 src.training.trainer INFO: Epoch 148/200 — train_loss=0.0247, val_loss=0.4709, val_cer=0.0466, val_wer=0.1450, lr=1.61e-05, time=38.3s
11:11:18 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:11:18 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:11:18 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:11:57 src.training.trainer INFO: Epoch 149/200 — train_loss=0.0127, val_loss=0.4848, val_cer=0.0469, val_wer=0.1488, lr=1.55e-05, time=38.7s
11:11:57 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:11:57 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:11:57 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:12:38 src.training.trainer INFO: Epoch 150/200 — train_loss=0.0224, val_loss=0.5080, val_cer=0.0483, val_wer=0.1564, lr=1.49e-05, time=41.3s
11:12:38 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:12:38 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:12:38 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:13:15 src.training.trainer INFO: Epoch 151/200 — train_loss=0.0273, val_loss=0.4700, val_cer=0.0476, val_wer=0.1538, lr=7.18e-06, time=35.9s
11:13:15 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:13:15 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:13:15 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:13:50 src.training.trainer INFO: Epoch 152/200 — train_loss=0.0124, val_loss=0.4783, val_cer=0.0476, val_wer=0.1513, lr=1.38e-05, time=35.4s
11:13:50 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:13:50 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:13:50 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:14:28 src.training.trainer INFO: Epoch 153/200 — train_loss=0.0275, val_loss=0.5227, val_cer=0.0424, val_wer=0.1400, lr=1.33e-05, time=37.7s
11:14:28 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:14:28 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:14:28 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:14:29 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/CNNLSTM_best.pt (val_cer=0.0424)
11:15:08 src.training.trainer INFO: Epoch 154/200 — train_loss=0.0134, val_loss=0.4816, val_cer=0.0469, val_wer=0.1513, lr=1.27e-05, time=38.9s
11:15:08 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified ard fascina' ref='daisy leaned forward at once horrified and fascina'
11:15:08 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:15:08 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:15:46 src.training.trainer INFO: Epoch 155/200 — train_loss=0.0191, val_loss=0.4449, val_cer=0.0442, val_wer=0.1400, lr=1.22e-05, time=37.8s
11:15:46 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified ard fascina' ref='daisy leaned forward at once horrified and fascina'
11:15:46 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:15:46 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:16:24 src.training.trainer INFO: Epoch 156/200 — train_loss=0.0178, val_loss=0.4259, val_cer=0.0469, val_wer=0.1475, lr=1.17e-05, time=38.2s
11:16:24 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:16:24 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:16:24 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:17:00 src.training.trainer INFO: Epoch 157/200 — train_loss=0.0160, val_loss=0.4590, val_cer=0.0455, val_wer=0.1488, lr=1.12e-05, time=35.8s
11:17:00 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified ard fascina' ref='daisy leaned forward at once horrified and fascina'
11:17:00 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:17:00 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:17:37 src.training.trainer INFO: Epoch 158/200 — train_loss=0.0162, val_loss=0.5078, val_cer=0.0462, val_wer=0.1488, lr=1.07e-05, time=36.5s
11:17:37 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:17:37 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:17:37 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:18:17 src.training.trainer INFO: Epoch 159/200 — train_loss=0.0122, val_loss=0.4867, val_cer=0.0486, val_wer=0.1526, lr=5.10e-06, time=39.6s
11:18:17 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:18:17 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:18:17 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:18:53 src.training.trainer INFO: Epoch 160/200 — train_loss=0.0163, val_loss=0.4623, val_cer=0.0466, val_wer=0.1526, lr=9.73e-06, time=35.7s
11:18:53 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:18:53 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:18:53 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:19:33 src.training.trainer INFO: Epoch 161/200 — train_loss=0.0165, val_loss=0.4570, val_cer=0.0473, val_wer=0.1488, lr=9.27e-06, time=39.7s
11:19:33 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:19:33 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:19:33 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:20:11 src.training.trainer INFO: Epoch 162/200 — train_loss=0.0231, val_loss=0.4452, val_cer=0.0431, val_wer=0.1387, lr=8.81e-06, time=37.9s
11:20:11 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:20:11 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:20:11 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:20:48 src.training.trainer INFO: Epoch 163/200 — train_loss=0.0168, val_loss=0.4612, val_cer=0.0442, val_wer=0.1400, lr=8.37e-06, time=36.3s
11:20:48 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:20:48 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:20:48 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:21:28 src.training.trainer INFO: Epoch 164/200 — train_loss=0.0248, val_loss=0.4814, val_cer=0.0473, val_wer=0.1513, lr=7.93e-06, time=39.8s
11:21:28 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:21:28 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:21:28 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:22:09 src.training.trainer INFO: Epoch 165/200 — train_loss=0.0234, val_loss=0.4828, val_cer=0.0442, val_wer=0.1425, lr=3.75e-06, time=40.2s
11:22:09 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:22:09 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:22:09 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:22:48 src.training.trainer INFO: Epoch 166/200 — train_loss=0.0151, val_loss=0.4740, val_cer=0.0469, val_wer=0.1475, lr=7.10e-06, time=39.2s
11:22:48 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:22:48 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:22:48 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:23:26 src.training.trainer INFO: Epoch 167/200 — train_loss=0.0146, val_loss=0.4726, val_cer=0.0476, val_wer=0.1551, lr=6.69e-06, time=38.2s
11:23:26 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:23:26 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:23:26 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:24:07 src.training.trainer INFO: Epoch 168/200 — train_loss=0.0173, val_loss=0.4692, val_cer=0.0448, val_wer=0.1450, lr=6.30e-06, time=40.7s
11:24:07 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:24:07 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:24:07 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:24:46 src.training.trainer INFO: Epoch 169/200 — train_loss=0.0152, val_loss=0.4561, val_cer=0.0438, val_wer=0.1438, lr=5.92e-06, time=38.6s
11:24:46 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:24:46 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:24:46 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:25:24 src.training.trainer INFO: Epoch 170/200 — train_loss=0.0202, val_loss=0.4442, val_cer=0.0459, val_wer=0.1488, lr=5.55e-06, time=37.2s
11:25:24 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:25:24 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:25:24 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:26:03 src.training.trainer INFO: Epoch 171/200 — train_loss=0.0089, val_loss=0.4633, val_cer=0.0459, val_wer=0.1475, lr=2.60e-06, time=38.4s
11:26:03 src.training.trainer INFO:   [0] pred='daisy leaned forward at once horrified and fascina' ref='daisy leaned forward at once horrified and fascina'
11:26:03 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:26:03 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:26:42 src.training.trainer INFO: Epoch 172/200 — train_loss=0.0129, val_loss=0.4850, val_cer=0.0469, val_wer=0.1488, lr=4.85e-06, time=39.5s
11:26:42 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:26:42 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:26:42 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:27:22 src.training.trainer INFO: Epoch 173/200 — train_loss=0.0137, val_loss=0.4685, val_cer=0.0448, val_wer=0.1450, lr=4.52e-06, time=38.9s
11:27:22 src.training.trainer INFO:   [0] pred='daisy leaned forwarrd at once horrified and fascin' ref='daisy leaned forward at once horrified and fascina'
11:27:22 src.training.trainer INFO:   [1] pred='r'                            ref='r'
11:27:22 src.training.trainer INFO:   [2] pred='k'                            ref='k'
11:27:22 src.training.trainer INFO: Early stopping: no improvement for 20 epochs (best_cer=0.0424)
11:27:22 src.training.trainer INFO: Training complete. Best val CER: 0.0424
wandb: uploading artifact run-bglnrqpm-decoded_samples; updating run metadata
wandb: uploading artifact run-bglnrqpm-decoded_samples
wandb: uploading output.log; uploading wandb-summary.json; uploading config.yaml
wandb: uploading history steps 171-172, summary, console lines 764-769
wandb: 
wandb: Run history:
wandb:          best/epoch ▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇███
wandb:        best/val_cer ███████████▇▇▇▆▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:        epoch_time_s ▄▄▃▄▃▅▇▅▅▅▁▇▅▇▅▂▇▃▇▂▆▂▃▆▅█▂▃▁▁▄▄▅█▁▄▆▄▂▅
wandb:  final/best_val_cer ▁
wandb:  final/total_epochs ▁
wandb:         train/epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:   train/global_step ▁▁▁▂▂▂▂▂▂▃▃▃▄▄▄▄▄▄▄▄▄▄▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb: train/learning_rate █████████▇▇▇▇▇▇▇▇▇▇▆▆▆▆▆▅▅▄▄▄▄▃▃▃▂▂▂▁▂▁▁
wandb:          train/loss ██▇▆▅▄▄▃▃▃▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:             val/cer ████████▇▄▃▃▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  +2 ...
wandb: 
wandb: Run summary:
wandb:          best/epoch 153
wandb:        best/val_cer 0.04243
wandb:        epoch_time_s 38.88723
wandb:  final/best_val_cer 0.04243
wandb:  final/total_epochs 173
wandb:         train/epoch 173
wandb:   train/global_step 44634
wandb: train/learning_rate 0.0
wandb:          train/loss 0.01369
wandb:             val/cer 0.04484
wandb:                  +2 ...
wandb: 
wandb: 🚀 View run cnn_lstm_gpu_v1 at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/bglnrqpm
wandb: ⭐️ View project at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: Synced 5 W&B file(s), 173 media file(s), 346 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260313_093633-bglnrqpm/logs
11:27:27 __main__ INFO: Best validation CER: 0.0424

cnn_lstm finished in 111.0 min (exit code 0)

======================================================================
TRAINING: transformer
======================================================================
Command: python scripts/train.py --model transformer --epochs 200 --batch-size 8 --lr 0.0001 --t-max 4096 --normalize --filter-by-length --wandb --wandb-project brain-text-decoder --wandb-run-name transformer_gpu_v1 --wandb-tags transformer gpu full-size --n-layers 6 --dropout 0.1

11:27:31 __main__ INFO: Loading dataset...
11:27:31 src.data.loader INFO: Loading cached trial index from data/willett_handwriting/trial_index.csv
11:27:31 __main__ INFO: Loaded 4126 trials
11:27:31 __main__ INFO: Length filter (t_max=4096): 4126 -> 3691 trials
11:27:31 __main__ INFO: Signal lengths: min=201, max=4094, mean=522, median=201
11:27:31 src.data.dataset INFO: Split: 2952 train, 369 val, 370 test
11:27:31 src.data.dataset INFO: Computed channel stats from 200 trials: mean=0.2085, std=0.4468
11:27:31 __main__ INFO: Model: transformer (20731932 params)
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /root/.netrc.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
wandb: setting up run 5lzpbp1x
wandb: Tracking run with wandb version 0.25.0
wandb: Run data is saved locally in /content/BCI-2/wandb/run-20260313_112739-5lzpbp1x
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run transformer_gpu_v1
wandb: ⭐️ View project at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: 🚀 View run at https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/5lzpbp1x
11:27:41 src.training.trainer INFO: W&B run initialized: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/5lzpbp1x
11:27:41 src.training.trainer INFO: Starting training: model=TransformerDecoder, device=cuda, epochs=200, amp=True
11:27:41 src.training.trainer INFO: Parameters: 20731932
11:27:41 src.training.trainer INFO: Temporal downsample factor: 4
/content/BCI-2/src/training/trainer.py:237: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  self.scheduler.step()
11:28:16 src.training.trainer INFO: Epoch 1/200 — train_loss=11.2504, val_loss=4.8848, val_cer=1.0000, val_wer=1.0000, lr=7.38e-05, time=35.6s
11:28:16 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:28:16 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:28:16 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:28:19 src.training.trainer INFO: Saved checkpoint to outputs/checkpoints/TransformerDecoder_best.pt (val_cer=1.0000)
11:28:53 src.training.trainer INFO: Epoch 2/200 — train_loss=4.1150, val_loss=5.6580, val_cer=1.0000, val_wer=1.0000, lr=1.00e-04, time=33.8s
11:28:53 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:28:53 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:28:53 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:29:27 src.training.trainer INFO: Epoch 3/200 — train_loss=4.0122, val_loss=5.6087, val_cer=1.0000, val_wer=1.0000, lr=1.00e-04, time=33.7s
11:29:27 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:29:27 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:29:27 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:30:01 src.training.trainer INFO: Epoch 4/200 — train_loss=3.6102, val_loss=4.0579, val_cer=1.0000, val_wer=1.0000, lr=1.00e-04, time=33.8s
11:30:01 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:30:01 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:30:01 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:30:35 src.training.trainer INFO: Epoch 5/200 — train_loss=2.9734, val_loss=3.7383, val_cer=1.0000, val_wer=1.0000, lr=9.99e-05, time=33.7s
11:30:35 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:30:35 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:30:35 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:31:09 src.training.trainer INFO: Epoch 6/200 — train_loss=2.5060, val_loss=3.5156, val_cer=1.0000, val_wer=1.0000, lr=9.99e-05, time=33.7s
11:31:09 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:31:09 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:31:09 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:31:43 src.training.trainer INFO: Epoch 7/200 — train_loss=2.0699, val_loss=3.2221, val_cer=1.0000, val_wer=1.0000, lr=4.99e-05, time=33.8s
11:31:43 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:31:43 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:31:43 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:32:17 src.training.trainer INFO: Epoch 8/200 — train_loss=1.8692, val_loss=3.0932, val_cer=1.0000, val_wer=1.0000, lr=9.97e-05, time=33.7s
11:32:17 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:32:17 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:32:17 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:32:51 src.training.trainer INFO: Epoch 9/200 — train_loss=1.7209, val_loss=2.8744, val_cer=1.0000, val_wer=1.0000, lr=9.96e-05, time=33.6s
11:32:51 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:32:51 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:32:51 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:33:25 src.training.trainer INFO: Epoch 10/200 — train_loss=1.6130, val_loss=2.6763, val_cer=1.0000, val_wer=1.0000, lr=9.95e-05, time=33.8s
11:33:25 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:33:25 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:33:25 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:33:59 src.training.trainer INFO: Epoch 11/200 — train_loss=1.5470, val_loss=2.3710, val_cer=1.0000, val_wer=1.0000, lr=9.94e-05, time=33.4s
11:33:59 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:33:59 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:33:59 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:34:33 src.training.trainer INFO: Epoch 12/200 — train_loss=1.4958, val_loss=2.6224, val_cer=1.0000, val_wer=1.0000, lr=9.93e-05, time=33.6s
11:34:33 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:34:33 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:34:33 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:35:07 src.training.trainer INFO: Epoch 13/200 — train_loss=1.4593, val_loss=2.6501, val_cer=1.0000, val_wer=1.0000, lr=4.96e-05, time=33.9s
11:35:07 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:35:07 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:35:07 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:35:41 src.training.trainer INFO: Epoch 14/200 — train_loss=1.4470, val_loss=2.4288, val_cer=1.0000, val_wer=1.0000, lr=9.90e-05, time=34.0s
11:35:41 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:35:41 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:35:41 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:36:15 src.training.trainer INFO: Epoch 15/200 — train_loss=1.4044, val_loss=2.1839, val_cer=1.0000, val_wer=1.0000, lr=9.88e-05, time=33.7s
11:36:15 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:36:15 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:36:15 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:36:49 src.training.trainer INFO: Epoch 16/200 — train_loss=1.3774, val_loss=2.2195, val_cer=1.0000, val_wer=1.0000, lr=9.87e-05, time=33.8s
11:36:49 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:36:49 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:36:49 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:37:24 src.training.trainer INFO: Epoch 17/200 — train_loss=1.3939, val_loss=2.4486, val_cer=1.0000, val_wer=1.0000, lr=9.85e-05, time=34.1s
11:37:24 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:37:24 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:37:24 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:37:58 src.training.trainer INFO: Epoch 18/200 — train_loss=1.3763, val_loss=2.4696, val_cer=1.0000, val_wer=1.0000, lr=9.83e-05, time=33.8s
11:37:58 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:37:58 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:37:58 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:38:33 src.training.trainer INFO: Epoch 19/200 — train_loss=1.3671, val_loss=2.2521, val_cer=1.0000, val_wer=1.0000, lr=4.90e-05, time=34.1s
11:38:33 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:38:33 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:38:33 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:39:07 src.training.trainer INFO: Epoch 20/200 — train_loss=1.3596, val_loss=2.4210, val_cer=1.0000, val_wer=1.0000, lr=9.78e-05, time=33.5s
11:39:07 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:39:07 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:39:07 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:39:41 src.training.trainer INFO: Epoch 21/200 — train_loss=1.3357, val_loss=2.2693, val_cer=1.0000, val_wer=1.0000, lr=9.76e-05, time=33.3s
11:39:41 src.training.trainer INFO:   [0] pred=''                             ref='h'
11:39:41 src.training.trainer INFO:   [1] pred=''                             ref='x'
11:39:41 src.training.trainer INFO:   [2] pred=''                             ref='r'
11:39:41 src.training.trainer INFO: Early stopping: no improvement for 20 epochs (best_cer=1.0000)
11:39:41 src.training.trainer INFO: Training complete. Best val CER: 1.0000
wandb: uploading history steps 19-19, summary, console lines 87-90; uploading artifact run-5lzpbp1x-decoded_samples; updating run metadata
wandb: uploading artifact run-5lzpbp1x-decoded_samples
wandb: uploading config.yaml; uploading media/table/decoded_samples_21_6e8b8d191acab08565ed.table.json
wandb: 
wandb: Run history:
wandb:          best/epoch ▁
wandb:        best/val_cer ▁
wandb:        epoch_time_s █▂▂▃▂▂▃▂▂▂▁▂▃▃▂▂▃▂▃▂▁
wandb:  final/best_val_cer ▁
wandb:  final/total_epochs ▁
wandb:         train/epoch ▁▁▂▂▂▃▃▃▄▄▅▅▅▆▆▆▇▇▇██
wandb:   train/global_step ▁▁▂▂▂▃▃▃▄▄▅▅▅▆▆▆▇▇▇██
wandb: train/learning_rate ▄█████▁█████▁█████▁██
wandb:          train/loss █▃▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:             val/cer ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  +2 ...
wandb: 
wandb: Run summary:
wandb:          best/epoch 1
wandb:        best/val_cer 1
wandb:        epoch_time_s 33.31487
wandb:  final/best_val_cer 1
wandb:  final/total_epochs 21
wandb:         train/epoch 21
wandb:   train/global_step 7749
wandb: train/learning_rate 0.0001
wandb:          train/loss 1.3357
wandb:             val/cer 1
wandb:                  +2 ...
wandb: 
wandb: 🚀 View run transformer_gpu_v1 at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder/runs/5lzpbp1x
wandb: ⭐️ View project at: https://wandb.ai/dlopezkluever-aiuteur/brain-text-decoder
wandb: Synced 5 W&B file(s), 21 media file(s), 42 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260313_112739-5lzpbp1x/logs
11:39:45 __main__ INFO: Best validation CER: 1.0000

transformer finished in 12.3 min (exit code 0)

======================================================================
TRAINING SUMMARY
======================================================================
  cnn_transformer      —   37.7 min — OK
  gru_decoder          —   14.2 min — OK
  cnn_lstm             —  111.0 min — OK
  transformer          —   12.3 min — OK
