#!/usr/bin/env python
"""Real-time neural decoding demo — terminal-based streaming simulation.

Loads a neural trial (from file or synthetic) and feeds it chunk-by-chunk
into the streaming decoder, displaying characters as they are decoded.

Usage:
    # Synthetic demo (no model checkpoint needed — uses random GRU weights):
    python scripts/realtime_demo.py

    # With a real checkpoint and trial file:
    python scripts/realtime_demo.py --checkpoint outputs/checkpoints/best.pt \\
                                     --trial data/sample_trial.npy

    # Adjust streaming speed and chunk size:
    python scripts/realtime_demo.py --chunk-size 25 --delay 0.05

Architecture for Lab Streaming Layer (LSL) integration:

    ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │  LSL Inlet   │────►│ StreamingBuffer  │────►│ StreamingDecoder│
    │ (pylsl)      │     │ (rolling window) │     │ (model + CTC)   │
    └──────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Terminal / UI   │
                                                  │  (char display)  │
                                                  └─────────────────┘

To integrate with real LSL hardware, replace the chunk-feeder loop in
``main()`` with:

    import pylsl
    inlet = pylsl.StreamInlet(pylsl.resolve_stream('type', 'EEG')[0])
    while True:
        chunk, timestamps = inlet.pull_chunk(max_samples=chunk_size)
        if chunk:
            result = decoder.feed(np.array(chunk, dtype=np.float32))
            ...
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time neural decoding demo (streaming simulation)",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt). If omitted, uses untrained GRU.",
    )
    p.add_argument(
        "--trial", type=str, default=None,
        help="Path to a .npy neural trial [T, C]. If omitted, generates synthetic data.",
    )
    p.add_argument(
        "--model-type", type=str, default="gru_decoder",
        choices=["gru_decoder", "cnn_lstm", "transformer", "cnn_transformer"],
        help="Model architecture (default: gru_decoder).",
    )
    p.add_argument(
        "--n-channels", type=int, default=192,
        help="Number of neural channels (default: 192).",
    )
    p.add_argument(
        "--t-max", type=int, default=2000,
        help="Sliding window size in timesteps (default: 2000).",
    )
    p.add_argument(
        "--chunk-size", type=int, default=50,
        help="Samples per chunk (default: 50). Smaller = more updates.",
    )
    p.add_argument(
        "--delay", type=float, default=0.04,
        help="Delay between chunks in seconds (default: 0.04). "
             "Simulates real-time data arrival rate.",
    )
    p.add_argument(
        "--stable", action="store_true",
        help="Use stable mode (only display characters consistent across "
             "multiple inferences).",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (default: cpu).",
    )
    p.add_argument(
        "--duration", type=float, default=5.0,
        help="Duration of synthetic trial in seconds at 250 Hz (default: 5.0).",
    )
    return p


def load_model(
    model_type: str, n_channels: int, checkpoint: str | None, device: str,
) -> torch.nn.Module:
    """Load a decoder model."""
    n_classes = 28

    if model_type == "gru_decoder":
        from src.models.gru_decoder import GRUDecoder
        model = GRUDecoder(
            n_channels=n_channels, n_classes=n_classes,
            proj_dim=256, hidden_size=512, n_layers=3, dropout=0.0,
        )
    elif model_type == "cnn_lstm":
        from src.models.cnn_lstm import CNNLSTM
        model = CNNLSTM(n_channels=n_channels, n_classes=n_classes)
    elif model_type == "transformer":
        from src.models.transformer import TransformerDecoder
        model = TransformerDecoder(n_channels=n_channels, n_classes=n_classes)
    elif model_type == "cnn_transformer":
        from src.models.cnn_transformer import CNNTransformer
        model = CNNTransformer(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}")

    model.to(device)
    model.eval()
    return model


def generate_synthetic_trial(
    n_channels: int, duration_sec: float, fs: float = 250.0,
) -> np.ndarray:
    """Generate a synthetic neural trial with structured patterns.

    Creates a mix of oscillatory signals and noise to simulate the kind
    of structured input a trained model would see.
    """
    rng = np.random.RandomState(42)
    T = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, T).reshape(-1, 1)

    # Base: mixture of low-frequency oscillations per channel
    freqs = rng.uniform(2, 20, size=(1, n_channels))
    phases = rng.uniform(0, 2 * np.pi, size=(1, n_channels))
    signal = np.sin(2 * np.pi * freqs * t + phases).astype(np.float32)

    # Add some gaussian noise
    signal += rng.randn(T, n_channels).astype(np.float32) * 0.3

    return signal


def terminal_callback(update) -> None:
    """Display streaming decoder updates in the terminal."""
    text = update.text or "(waiting...)"
    status = "stable" if update.is_stable else "updating"
    sys.stdout.write(
        f"\r\033[K"  # clear line
        f"[{update.latency_ms:6.1f}ms | buf={update.buffer_length:4d}] "
        f"Decoded: \033[1m{text}\033[0m "
        f"({status})"
    )
    sys.stdout.flush()


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 60)
    print("  Real-Time Neural Decoding Demo")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model: {args.model_type}...", end=" ", flush=True)
    model = load_model(
        args.model_type, args.n_channels, args.checkpoint, args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"done ({n_params:,} parameters)")

    # Load or generate trial data
    if args.trial is not None:
        print(f"Loading trial: {args.trial}")
        trial_data = np.load(args.trial).astype(np.float32)
    else:
        print(f"Generating synthetic trial ({args.duration}s at 250 Hz)...")
        trial_data = generate_synthetic_trial(
            args.n_channels, args.duration,
        )

    T, C = trial_data.shape
    print(f"Trial shape: [{T}, {C}]")
    print(f"Chunk size: {args.chunk_size} samples")
    print(f"Window size: {args.t_max} timesteps")
    print(f"Simulated delay: {args.delay}s per chunk")
    mode = "stable" if args.stable else "eager"
    print(f"Decode mode: {mode}")
    print()
    print("-" * 60)
    print("Streaming...")
    print()

    # Import streaming module
    from src.inference.streaming import StreamingDecoder

    decoder = StreamingDecoder(
        model=model,
        n_channels=C,
        t_max=args.t_max,
        chunk_size=args.chunk_size,
        device=args.device,
        stable_mode=args.stable,
    )

    updates = []
    t_start = time.perf_counter()

    for start in range(0, T, args.chunk_size):
        end = min(start + args.chunk_size, T)
        chunk = trial_data[start:end]

        result = decoder.feed(chunk)
        if result is not None:
            updates.append(result)
            terminal_callback(result)

        # Simulate real-time arrival
        time.sleep(args.delay)

    # Final decode
    final = decoder.force_decode()
    updates.append(final)
    terminal_callback(final)

    wall_time = (time.perf_counter() - t_start) * 1000.0

    # Summary
    print()
    print()
    print("-" * 60)
    print("Summary")
    print("-" * 60)
    print(f"Final decoded text: \"{decoder.current_text}\"")
    print(f"Total updates:      {len(updates)}")
    print(f"Wall-clock time:    {wall_time:.0f} ms")
    print()

    stats = decoder.latency.summary()
    print("Inference Latency:")
    print(f"  Mean:  {stats['mean_ms']:7.2f} ms")
    print(f"  Min:   {stats['min_ms']:7.2f} ms")
    print(f"  Max:   {stats['max_ms']:7.2f} ms")
    print(f"  P95:   {stats['p95_ms']:7.2f} ms")
    print(f"  Count: {stats['count']}")

    target_met = stats["p95_ms"] < 300.0
    marker = "PASS" if target_met else "WARN"
    print(f"\n  < 300ms target: [{marker}] (P95 = {stats['p95_ms']:.1f} ms)")
    print()


if __name__ == "__main__":
    main()
