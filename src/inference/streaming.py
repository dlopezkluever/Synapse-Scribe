"""Streaming inference pipeline for real-time neural decoding.

Provides a sliding-window buffer and incremental CTC decoder that accepts
incoming neural data in chunks, runs model inference on each window, and
emits decoded characters with low latency.

Architecture for future LSL (Lab Streaming Layer) integration:

    LSL inlet  ──►  StreamingBuffer  ──►  StreamingDecoder  ──►  display / UI
    (real ECoG)      (rolling window)      (model + CTC)         (terminal / app)

Replace the simulated chunk feeder in ``scripts/realtime_demo.py`` with an
LSL inlet (``pylsl.StreamInlet``) to receive live neural data.  The rest of
the pipeline (buffer → model → decode → display) remains unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.decoding.greedy import greedy_decode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming buffer
# ---------------------------------------------------------------------------

class StreamingBuffer:
    """Rolling FIFO buffer for incoming neural time-series data.

    New chunks are appended to the right; when the buffer exceeds
    ``max_length``, the oldest samples are dropped from the left.

    Args:
        n_channels: Number of neural channels (C).
        max_length: Maximum buffer length in timesteps (T).
    """

    def __init__(self, n_channels: int, max_length: int = 2000) -> None:
        self.n_channels = n_channels
        self.max_length = max_length
        self._buffer: np.ndarray = np.zeros((0, n_channels), dtype=np.float32)
        self._total_samples_received: int = 0

    @property
    def length(self) -> int:
        """Current number of timesteps in the buffer."""
        return self._buffer.shape[0]

    @property
    def total_samples_received(self) -> int:
        """Total samples fed since creation / last reset."""
        return self._total_samples_received

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached its maximum length."""
        return self.length >= self.max_length

    def feed(self, chunk: np.ndarray) -> None:
        """Append a chunk of data to the buffer.

        Args:
            chunk: [T_chunk, C] array of new neural samples.

        Raises:
            ValueError: If ``chunk`` has the wrong number of channels.
        """
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {chunk.shape[1]}"
            )
        chunk = chunk.astype(np.float32)
        self._buffer = np.concatenate([self._buffer, chunk], axis=0)
        self._total_samples_received += chunk.shape[0]

        # Trim to max_length (keep most recent samples)
        if self._buffer.shape[0] > self.max_length:
            self._buffer = self._buffer[-self.max_length :]

    def get_window(self) -> np.ndarray:
        """Return the current buffer contents as a [T, C] array.

        If the buffer is shorter than ``max_length``, the returned array
        is zero-padded on the left so that the output is always
        ``[max_length, C]``.
        """
        if self._buffer.shape[0] >= self.max_length:
            return self._buffer[-self.max_length :].copy()
        # Zero-pad on the left
        pad_len = self.max_length - self._buffer.shape[0]
        pad = np.zeros((pad_len, self.n_channels), dtype=np.float32)
        return np.concatenate([pad, self._buffer], axis=0)

    def get_raw(self) -> np.ndarray:
        """Return the unpadded buffer contents as [T_actual, C]."""
        return self._buffer.copy()

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer = np.zeros((0, self.n_channels), dtype=np.float32)
        self._total_samples_received = 0


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Tracks per-update inference latency."""

    latencies_ms: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.latencies_ms.append(ms)

    @property
    def count(self) -> int:
        return len(self.latencies_ms)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return float(np.max(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def min_ms(self) -> float:
        return float(np.min(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 95))

    def summary(self) -> dict:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
        }


# ---------------------------------------------------------------------------
# Streaming decoder
# ---------------------------------------------------------------------------

@dataclass
class DecoderUpdate:
    """Result of a single streaming decode step."""

    text: str
    new_chars: str
    is_stable: bool
    latency_ms: float
    buffer_length: int
    total_samples: int


class StreamingDecoder:
    """Incremental neural decoder using a sliding-window approach.

    Feeds incoming neural data into a :class:`StreamingBuffer`, runs model
    inference on the current window, and performs CTC greedy decoding.  Two
    modes are supported:

    * **eager** (default): emits the full decoded text each update.  The
      output may change as more data arrives ("flickering").
    * **stable**: only emits characters that remain consistent across
      ``stability_window`` consecutive inferences.

    Args:
        model: Trained decoder model (``nn.Module``).
        n_channels: Number of input channels.
        t_max: Sliding window length (timesteps).
        chunk_size: Minimum samples between inference updates.
        device: Torch device for inference.
        stable_mode: If True, use stable (non-flickering) output.
        stability_window: Number of consecutive matches required to
            confirm a character in stable mode.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_channels: int = 192,
        t_max: int = 2000,
        chunk_size: int = 50,
        device: str | torch.device = "cpu",
        stable_mode: bool = False,
        stability_window: int = 3,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.buffer = StreamingBuffer(n_channels=n_channels, max_length=t_max)
        self.chunk_size = chunk_size
        self.t_max = t_max

        # Decoding state
        self._prev_text: str = ""
        self._samples_since_last_inference: int = 0

        # Stable mode
        self.stable_mode = stable_mode
        self.stability_window = stability_window
        self._recent_texts: list[str] = []
        self._stable_text: str = ""

        # Latency tracking
        self.latency = LatencyStats()

    @property
    def current_text(self) -> str:
        """The latest decoded text."""
        if self.stable_mode:
            return self._stable_text
        return self._prev_text

    def feed(self, chunk: np.ndarray) -> Optional[DecoderUpdate]:
        """Feed a chunk of neural data and optionally run inference.

        Inference is triggered once at least ``chunk_size`` new samples have
        been accumulated since the last inference.

        Args:
            chunk: [T_chunk, C] neural data.

        Returns:
            A :class:`DecoderUpdate` if inference was triggered, else None.
        """
        self.buffer.feed(chunk)
        self._samples_since_last_inference += chunk.shape[0] if chunk.ndim == 2 else 1

        if self._samples_since_last_inference < self.chunk_size:
            return None

        self._samples_since_last_inference = 0
        return self._run_inference()

    def force_decode(self) -> DecoderUpdate:
        """Force an inference pass on whatever is currently in the buffer."""
        self._samples_since_last_inference = 0
        return self._run_inference()

    def reset(self) -> None:
        """Reset all state for a new trial."""
        self.buffer.reset()
        self._prev_text = ""
        self._samples_since_last_inference = 0
        self._recent_texts.clear()
        self._stable_text = ""
        self.latency = LatencyStats()

    def _run_inference(self) -> DecoderUpdate:
        """Run model inference on the current buffer window."""
        t0 = time.perf_counter()

        window = self.buffer.get_window()  # [t_max, C]
        tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)  # [1, T, C]

        with torch.no_grad():
            logits = self.model(tensor)  # [1, T', n_classes]

        decoded = greedy_decode(logits)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.latency.record(elapsed_ms)

        # Determine new characters
        new_chars = ""
        if len(decoded) > len(self._prev_text):
            # Check if decoded starts with previous text (common case)
            if decoded.startswith(self._prev_text):
                new_chars = decoded[len(self._prev_text) :]
            else:
                new_chars = decoded

        old_text = self._prev_text
        self._prev_text = decoded

        # Stable mode: track recent texts and find longest stable prefix
        is_stable = True
        if self.stable_mode:
            self._recent_texts.append(decoded)
            if len(self._recent_texts) > self.stability_window:
                self._recent_texts = self._recent_texts[-self.stability_window :]
            self._stable_text = self._compute_stable_prefix()
            is_stable = self._stable_text == decoded
            new_chars = self._stable_text[len(old_text):] if len(self._stable_text) > len(old_text) else ""

        return DecoderUpdate(
            text=self._stable_text if self.stable_mode else decoded,
            new_chars=new_chars,
            is_stable=is_stable,
            latency_ms=elapsed_ms,
            buffer_length=self.buffer.length,
            total_samples=self.buffer.total_samples_received,
        )

    def _compute_stable_prefix(self) -> str:
        """Find the longest common prefix across recent decoded texts."""
        if not self._recent_texts:
            return ""
        if len(self._recent_texts) < self.stability_window:
            # Not enough history yet — return empty
            return self._stable_text

        texts = self._recent_texts[-self.stability_window :]
        prefix = []
        for chars in zip(*texts):
            if len(set(chars)) == 1:
                prefix.append(chars[0])
            else:
                break
        return "".join(prefix)


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def simulate_streaming(
    model: torch.nn.Module,
    trial_data: np.ndarray,
    chunk_size: int = 50,
    n_channels: int = 192,
    t_max: int = 2000,
    device: str | torch.device = "cpu",
    stable_mode: bool = False,
    callback=None,
) -> dict:
    """Simulate streaming inference on a complete trial.

    Feeds ``trial_data`` chunk-by-chunk into a :class:`StreamingDecoder`
    and collects all updates.

    Args:
        model: Trained decoder model.
        trial_data: [T, C] array of neural data for one trial.
        chunk_size: Samples per chunk.
        n_channels: Number of channels.
        t_max: Sliding window length.
        device: Torch device.
        stable_mode: Use stable decoding mode.
        callback: Optional ``callable(DecoderUpdate)`` invoked after each
            inference step (e.g., for terminal display).

    Returns:
        Dict with ``final_text``, ``updates`` list, and ``latency`` stats.
    """
    decoder = StreamingDecoder(
        model=model,
        n_channels=n_channels,
        t_max=t_max,
        chunk_size=chunk_size,
        device=device,
        stable_mode=stable_mode,
    )

    updates: list[DecoderUpdate] = []
    total_t = trial_data.shape[0]

    for start in range(0, total_t, chunk_size):
        end = min(start + chunk_size, total_t)
        chunk = trial_data[start:end]
        result = decoder.feed(chunk)
        if result is not None:
            updates.append(result)
            if callback is not None:
                callback(result)

    # Final decode on any remaining buffered data
    final = decoder.force_decode()
    if not updates or final.text != updates[-1].text:
        updates.append(final)
        if callback is not None:
            callback(final)

    return {
        "final_text": decoder.current_text,
        "updates": updates,
        "latency": decoder.latency.summary(),
        "n_updates": len(updates),
    }
