The decoder used in **High‑performance brain‑to‑text communication via handwriting** is not a generic LSTM or transformer. It uses a **very specific two-stage architecture** designed for neural spike sequences from implanted electrode arrays. ([Preprints][1])

If you replicate even a simplified version of this architecture, your project will look **much closer to real BCI research code**.

Below is the architecture, explained as it is typically implemented.

---

# The Willett Handwriting Decoder Architecture

## Overview

The system is a **sequence-to-sequence neural decoder**.

Pipeline:

```
neural spikes
   ↓
feature extraction
   ↓
GRU / RNN decoder
   ↓
character probabilities
   ↓
CTC decoding
   ↓
language model correction
```

This design works because handwriting is **a time-varying motor pattern**, and neural signals encode the trajectory of movements over time. ([Nature][2])

---

# Step 1: Neural Feature Representation

The participant had **192 electrodes** implanted in motor cortex.

Each time step stores spike counts.

Typical representation:

```
time bin: 10 ms
channels: 192
```

Input tensor:

```
T × 192
```

Example:

```
100 time steps × 192 electrodes
```

Each value represents how many neural spikes occurred in that bin.

Researchers typically normalize these values:

```
z-score normalization per channel
```

---

# Step 2: Temporal Feature Encoder

Instead of feeding raw spikes to the decoder, the model first learns temporal features.

Typical implementation:

```
Fully Connected Layer
↓
Nonlinearity (ReLU)
↓
Dropout
```

Example:

```
Input: 192
Hidden: 256
```

Output:

```
T × 256
```

This stage transforms raw electrode activity into **neural features**.

---

# Step 3: Recurrent Neural Decoder

The core of the Willett decoder is a **stacked GRU network**.

Architecture:

```
GRU layer 1
GRU layer 2
GRU layer 3
```

Example configuration:

```
hidden size: 512
layers: 3
bidirectional: no
```

Why GRUs?

Because they model **temporal dependencies in neural signals**.

RNN architectures like **GRU and LSTM** maintain internal memory through gating mechanisms that control what information is remembered or forgotten. ([Wikipedia][3])

The network learns:

```
neural pattern → letter trajectory
```

---

# Step 4: Character Probability Layer

The GRU outputs a vector for every time step.

This goes into a classifier:

```
Linear layer
↓
Softmax
```

Output:

```
T × vocab_size
```

Example vocabulary:

```
26 letters
space
punctuation
blank token
```

Example:

```
T × 32
```

Each timestep predicts a **probability distribution over characters**.

---

# Step 5: CTC Loss (Critical Component)

The system uses **Connectionist Temporal Classification (CTC)**.

Reason:

The dataset contains:

```
neural sequence → full sentence
```

But we do **not know the alignment** between characters and time steps.

CTC solves this by computing probabilities across all possible alignments.

Example:

```
time sequence:
h h h e e l l l o

CTC collapse:
hello
```

This allows the network to learn timing automatically.

---

# Step 6: Language Model Correction

The final stage improves text quality.

Pipeline:

```
neural decoder output
↓
beam search
↓
language model scoring
↓
best sentence
```

Example:

Neural decoder outputs:

```
helo wrld
```

Language model corrects to:

```
hello world
```

This dramatically improves performance.

---

# Real Architecture Diagram

```
Neural spikes (192 channels)
        │
        ▼
Feature projection layer
        │
        ▼
3-layer GRU network
        │
        ▼
Linear classifier
        │
        ▼
Softmax probabilities
        │
        ▼
CTC decoding
        │
        ▼
Language model correction
        │
        ▼
Final text
```

---

# Typical Input Example

One sentence might produce:

```
192 electrodes
1000 time steps
```

Input tensor:

```
1000 × 192
```

Output:

```
"the quick brown fox"
```

---

# Example PyTorch Skeleton

Simplified version:

```python
class HandwritingDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(192, 256)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=512,
            num_layers=3,
            batch_first=True
        )

        self.classifier = nn.Linear(512, vocab_size)

    def forward(self, x):

        x = F.relu(self.input_layer(x))

        x, _ = self.gru(x)

        x = self.classifier(x)

        return x
```

Training:

```
loss = CTC_loss(predictions, targets)
```

---

# Why This Architecture Works

Handwriting movements produce **rich temporal neural patterns**.

Letters differ in trajectory:

```
a → circular motion
l → vertical stroke
s → curve
```

Those trajectories produce **distinct neural time patterns**, which makes them easier to decode than simple cursor movement tasks. ([Neuron][4])

That is why the handwriting BCI reached ~90 characters per minute.

---

# If You Want to Make This Even More Research-Grade

You can extend the architecture with:

### 1. Neural smoothing filter

```
Gaussian smoothing across time
```

### 2. Adaptive normalization

```
per-session baseline correction
```

### 3. Transformer temporal decoder

Some recent work replaces GRUs with attention models to capture long-range neural dependencies. ([ScienceDirect][5])

---

# The Version I Would Build (for Your Project)

If I were building your **portfolio-level but lab-grade version**, I would do:

```
neural spikes
↓
temporal convolution layer
↓
2-layer GRU
↓
linear classifier
↓
CTC loss
↓
language model
```

That is very close to real research systems.

---

If you'd like, I can also show you **one improvement to the Willett architecture that modern BCI labs are starting to add**, which would make your project look **extremely cutting-edge rather than just reproducing the 2021 paper.**

[1]: https://www.preprints.org/manuscript/202312.0674/v1?utm_source=chatgpt.com "Comparing Transformer and RNN Models in BCIs for Handwritten Text Decoding via Neural Signals[v1] | Preprints.org"
[2]: https://www.nature.com/articles/s41562-025-02158-w?utm_source=chatgpt.com "Neural encoding of handwriting in the human motor cortex | Nature Human Behaviour"
[3]: https://en.wikipedia.org/wiki/Long_short-term_memory?utm_source=chatgpt.com "Long short-term memory"
[4]: https://www.neuron.yale.edu/ftp/ted/nsgoutreach/rehs2024/willett2021.pdf?utm_source=chatgpt.com "Article
High-performance brain-to-text 
communicat"
[5]: https://www.sciencedirect.com/science/article/pii/S0925231224000146?utm_source=chatgpt.com "Temporal-channel cascaded transformer for imagined handwriting character recognition - ScienceDirect"
