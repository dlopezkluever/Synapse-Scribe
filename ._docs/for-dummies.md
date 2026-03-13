1. The Big Picture: What is a BCI?

A Brain-Computer Interface (BCI) reads electrical activity from the brain and translates it into something useful — in our case, text. When someone thinks about   writing the letter "a", neurons in their motor cortex fire in a specific pattern. If you can record those patterns and learn what they mean, you can decode    
thought into text.

Our project does this offline (from saved recordings, not live), but the pipeline is identical to what clinical BCIs use.

---
2. The Neural Signals

Neurons and Electrical Activity

Your brain has ~86 billion neurons. When neurons communicate, they produce tiny electrical voltages. Different recording methods capture this at different      
scales:

┌─────────────────────┬────────────────────────────────────────┬──────────────────────────────────┬───────────────────┐
│       Method        │            What It Records             │             Quality              │   Invasiveness    │
├─────────────────────┼────────────────────────────────────────┼──────────────────────────────────┼───────────────────┤
│ EEG                 │ Scalp surface electrical activity      │ Low (signal is blurred by skull) │ None              │
├─────────────────────┼────────────────────────────────────────┼──────────────────────────────────┼───────────────────┤
│ ECoG                │ Cortical surface (electrodes on brain) │ High                             │ Surgical          │
├─────────────────────┼────────────────────────────────────────┼──────────────────────────────────┼───────────────────┤
│ Utah Array (spikes) │ Individual neuron firing               │ Very high                        │ Implanted needles │
└─────────────────────┴────────────────────────────────────────┴──────────────────────────────────┴───────────────────┘

Our primary dataset (Willett handwriting) uses a Utah array — a small chip implanted in motor cortex with 192 tiny electrodes. Each electrode records when      
nearby neurons "fire" (produce a voltage spike). The participant imagines writing letters by hand, and the array captures the neural activity for each letter.  

What the Raw Data Looks Like

A single trial (one letter) is a matrix:
[time_steps × channels]
e.g., [2000 × 192]
    2000 time samples (8 seconds at 250 Hz)
    192 electrodes recording simultaneously
Think of it like 192 microphones all recording at once — except instead of sound, they're recording electrical brain activity.

---
3. Preprocessing: Cleaning the Signals

Raw neural data is noisy. Before any ML model can use it, we need to clean it:

Bandpass Filtering

Neural signals contain useful frequencies (brain activity) mixed with useless ones (electrical noise, muscle artifacts). A bandpass filter keeps only
frequencies in a range (e.g., 1–200 Hz) and removes everything else — like an equalizer on a stereo that cuts out static.

Notch Filtering

Power lines emit 60 Hz electrical hum that contaminates recordings. A notch filter surgically removes exactly 60 Hz (and its harmonic at 120 Hz) while keeping  
everything else.

Z-Score Normalization

Each electrode has different baseline voltage levels. Z-scoring standardizes each channel to have mean=0 and standard deviation=1:
normalized = (value - mean) / standard_deviation
This puts all channels on the same scale so the model treats them equally.

Bad Channel Rejection

Some electrodes malfunction (dead channels with no signal, or noisy channels with garbage data). We detect and remove these automatically.

---
4. Feature Extraction: Making Signals Model-Friendly

Raw cleaned signals are still not ideal for ML models. Feature extraction transforms them into representations that are easier to learn from.

Pathway C: Firing Rate Binning (for Willett data)

Instead of raw voltage spikes, we count how many times each neuron fires in small time windows (10 ms bins). This gives a firing rate — a smoother, more stable 
signal. Then we apply a square-root transform (√rate) because firing rates follow Poisson statistics, and the square root stabilizes variance (makes the numbers   more evenly distributed, which helps training).

raw spikes: [0,0,1,0,0,1,1,0,0,0]  →  bin count: 3 spikes per 10ms  →  √3 ≈ 1.73

This is applied to ALL models when using Willett data — it's a data-level step.

Pathway A: Temporal Convolution

A bank of 1D convolutions with different kernel sizes (3, 7, 15) slides across the time axis. Small kernels capture fast, local patterns; large kernels capture 
slower, broader patterns. The outputs are concatenated — like looking at the signal through multiple different magnifying glasses simultaneously.

Pathway B: Linear Projection

For Transformer models, we simply multiply each timestep's channel vector by a learned weight matrix to project it into the model's working dimensionality. Plus   positional encodings — sine/cosine patterns added to tell the model where in time each sample is (since Transformers have no inherent sense of order).

---
5. Sequence Models: Learning the Brain-to-Text Mapping

This is the core ML. The model takes a sequence of neural features and outputs a sequence of character probabilities.

What is a Neural Network?

A function that takes numbers in and produces numbers out, with millions of adjustable internal parameters (weights). During training, we show it examples      
(neural recording → known text label) and adjust the weights to reduce errors. After enough examples, it learns to generalize to recordings it hasn't seen      
before.

GRU (Gated Recurrent Unit) — Model A

Processes the sequence one timestep at a time, maintaining a hidden state — a summary of everything it has seen so far. At each step, it decides:
- What to forget from its memory
- What new information to add
- What to output

timestep 1: sees neural frame → updates hidden state
timestep 2: sees next frame + previous hidden state → updates
...
timestep 2000: has accumulated context from the entire trial → outputs prediction

GRUs are simpler and faster than LSTMs (below), and are what the original Willett paper used.

LSTM (Long Short-Term Memory) — Model B

Very similar to GRU but with a more complex gating mechanism (3 gates instead of 2) and a separate "cell state" for long-term memory. Bidirectional means we run   two LSTMs — one reading forward in time, one reading backward — and combine their outputs. This lets the model use future context to interpret past signals.   

Transformer — Model C

Instead of processing sequentially, the Transformer uses self-attention: every timestep looks at every other timestep simultaneously and decides which are most 
relevant. This is massively parallel (fast on GPUs) and can capture very long-range dependencies.

Self-attention in plain English: for each timestep, ask "which other timesteps in this trial should I pay attention to when making my prediction?" The model    
learns these attention patterns during training.

Hybrid CNN-Transformer — Model D

Uses a CNN front-end to extract local features AND reduce the sequence length by 4× (making the Transformer much cheaper to run), then applies a Transformer to 
the compressed sequence. Best of both worlds.

---
6. CTC: Solving the Alignment Problem

Here's a key challenge: for a trial where someone imagines writing "hello," we have 2000 timesteps of neural data but only 5 characters. Which timesteps        
correspond to which characters? We don't know, and manually labeling this would be impractical.

Connectionist Temporal Classification (CTC) solves this. It:

1. Adds a special blank token (meaning "no character here")
2. Allows the model to output any combination of characters and blanks
3. During training, sums the probability over ALL valid alignments that produce the target text

Example — all of these outputs collapse to "hello":
h h h _ e e _ l l l _ l _ o o  →  hello
_ h _ e _ l _ l _ o _ _ _ _ _  →  hello
h _ _ _ e l _ l o _ _ _ _ _ _  →  hello

CTC says: "I don't care WHICH alignment you use, as long as the collapsed result matches the target." This is computed efficiently using dynamic programming (an   algorithm that avoids redundant computation by breaking the problem into overlapping subproblems).

---
7. Decoding: From Probabilities to Text

The model outputs a [2000 × 28] matrix — at each of the 2000 timesteps, a probability for each of 28 classes (a-z + space + blank).

Greedy Decoding

At each timestep, pick the highest-probability character. Collapse repeats. Remove blanks.
h h h _ e e l l l _ o  →  collapse repeats  →  h _ e l _ o  →  remove blanks  →  "helo"
Fast but can make errors because it never considers the full sequence — each timestep is decided independently.

Beam Search

Instead of picking one best character per timestep, keep track of the top-k hypotheses (candidate sequences) and expand all of them at each step. Prune to keep 
only the best k at each point. This explores more possibilities and finds better global solutions.

Think of it like navigating a maze: greedy decoding always turns in the direction that looks best right now, while beam search explores multiple paths
simultaneously and picks the one that turns out best overall.

---
8. Language Model Correction

Even with beam search, the decoder might output "helo wrld" — phonetically close but not real words. A language model (LM) knows what real English looks like   
and can fix this.

KenLM (N-gram model)

Learns the probability of character sequences from a large text corpus. It knows that "hell" is very likely to be followed by "o", and "wrld" is extremely      
unlikely. During beam search, the LM score is combined with the CTC score:

final_score = λ × CTC_confidence + (1-λ) × language_model_confidence

This is called shallow fusion — the same technique used in speech recognition systems like Siri or Alexa.

---
9. Evaluation Metrics

CER (Character Error Rate)

How many characters need to be inserted, deleted, or substituted to turn the prediction into the ground truth, divided by the reference length:
predicted: "helo wrld"
reference: "hello world"
edits needed: insert 'l', insert 'o' → 2 edits / 11 chars = 18% CER

WER (Word Error Rate)

Same idea but at the word level.

---
How It All Fits Together

Brain imagines writing "hello"
        ↓
Utah array records 192 channels × 2000 timesteps
        ↓
Preprocessing: filter noise, normalize, remove bad channels
        ↓
Feature extraction: bin spikes → firing rates → √transform
        ↓
Sequence model (GRU/LSTM/Transformer): learns neural patterns → character probabilities
        ↓
CTC decoding: collapse probability matrix → "helo"
        ↓
Language model: "helo" → "hello"

Every phase in the task list builds one piece of this pipeline, starting from the bottom (data loading) and working up to the full system with a web demo. 

