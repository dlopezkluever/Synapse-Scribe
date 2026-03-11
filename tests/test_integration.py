"""End-to-end integration tests for the BCI decoding pipeline.

Tests the full flow: synthetic data -> preprocessing -> feature extraction ->
model forward pass -> CTC loss -> greedy decode -> metrics (CER).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.preprocessing.filter import bandpass_filter, notch_filter
from src.preprocessing.normalize import (
    compute_normalization_stats,
    zscore_normalize,
)
from src.features.firing_rate import compute_firing_rate_features
from src.models.gru_decoder import GRUDecoder
from src.models.cnn_lstm import CNNLSTM
from src.training.ctc_loss import CTCLossWrapper
from src.decoding.greedy import greedy_decode, greedy_decode_batch
from src.evaluation.metrics import compute_cer
from src.data.dataset import text_to_indices, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create synthetic neural data: [4 trials, 200 timesteps, 192 channels].

    Simulates spike-count-like data with a sampling rate of 250 Hz.
    """
    rng = np.random.RandomState(42)
    data = rng.randn(4, 200, 192).astype(np.float32)
    return data


@pytest.fixture
def reference_texts():
    """Short reference texts for CER computation."""
    return ["hello", "world", "test", "brain"]


@pytest.fixture
def fs():
    """Sampling rate in Hz."""
    return 250.0


# ---------------------------------------------------------------------------
# Helper: run full pipeline for a single model class
# ---------------------------------------------------------------------------

def _run_pipeline(model_cls, synthetic_data, reference_texts, fs, **model_kwargs):
    """Run the complete pipeline for a given model and return (decoded, cer)."""
    n_trials, T, C = synthetic_data.shape

    # --- 1. Preprocessing: filter + normalize ---
    preprocessed = []
    for i in range(n_trials):
        trial = synthetic_data[i]  # [T, C]

        # Bandpass filter (low=1, high=100 to stay within Nyquist for fs=250)
        filtered = bandpass_filter(trial, fs=fs, low=1.0, high=100.0)

        # Notch filter at 60 Hz
        filtered = notch_filter(filtered, fs=fs, freqs=[60.0])

        preprocessed.append(filtered)

    # Compute normalization stats from all trials, then normalize
    norm_stats = compute_normalization_stats(preprocessed)
    normalized = [zscore_normalize(t, norm_stats) for t in preprocessed]

    # --- 2. Feature extraction: firing rate binning ---
    features = [compute_firing_rate_features(t, bin_width_ms=10.0, fs=fs) for t in normalized]

    # Pad/truncate to the same length for batching
    max_len = max(f.shape[0] for f in features)
    padded = []
    input_lengths = []
    for f in features:
        actual_len = f.shape[0]
        input_lengths.append(actual_len)
        if f.shape[0] < max_len:
            pad = np.zeros((max_len - f.shape[0], f.shape[1]), dtype=np.float32)
            f = np.concatenate([f, pad], axis=0)
        padded.append(f)

    batch = torch.from_numpy(np.stack(padded))  # [B, T', C]
    n_channels = batch.shape[2]

    # --- 3. Model forward pass ---
    model = model_cls(n_channels=n_channels, n_classes=VOCAB_SIZE, **model_kwargs)
    model.eval()
    with torch.no_grad():
        logits = model(batch)  # [B, T', 28]

    assert logits.ndim == 3
    assert logits.shape[0] == n_trials
    assert logits.shape[2] == VOCAB_SIZE

    # --- 4. CTC loss computation ---
    targets_list = [text_to_indices(t) for t in reference_texts]
    targets_cat = torch.tensor(
        [idx for seq in targets_list for idx in seq], dtype=torch.long
    )
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths_tensor = torch.tensor(
        [len(seq) for seq in targets_list], dtype=torch.long
    )

    ctc = CTCLossWrapper()
    # Re-run with grad enabled for loss computation
    model.train()
    logits_with_grad = model(batch)
    loss = ctc(logits_with_grad, targets_cat, input_lengths_tensor, target_lengths_tensor)

    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)

    # --- 5. Greedy decode ---
    model.eval()
    with torch.no_grad():
        logits_for_decode = model(batch)

    # Test single-sample decode
    decoded_single = greedy_decode(logits_for_decode[0])
    assert isinstance(decoded_single, str)

    # Test batch decode
    decoded_batch = greedy_decode_batch(logits_for_decode)
    assert isinstance(decoded_batch, list)
    assert len(decoded_batch) == n_trials
    for s in decoded_batch:
        assert isinstance(s, str)

    # --- 6. Metrics ---
    cer = compute_cer(decoded_batch, reference_texts)
    assert isinstance(cer, float)
    assert cer >= 0.0  # CER is non-negative

    return decoded_batch, cer, loss.item()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestEndToEndGRU:
    """End-to-end integration test with the GRU Decoder (Model A)."""

    def test_full_pipeline(self, synthetic_data, reference_texts, fs):
        decoded, cer, loss = _run_pipeline(
            GRUDecoder, synthetic_data, reference_texts, fs,
        )
        assert len(decoded) == 4
        assert cer >= 0.0
        assert loss > 0.0

    def test_decoded_outputs_are_strings(self, synthetic_data, reference_texts, fs):
        decoded, _, _ = _run_pipeline(
            GRUDecoder, synthetic_data, reference_texts, fs,
        )
        for s in decoded:
            assert isinstance(s, str)
            # Decoded characters should be lowercase letters or spaces
            for ch in s:
                assert ch in "abcdefghijklmnopqrstuvwxyz "

    def test_cer_does_not_error(self, synthetic_data, reference_texts, fs):
        _, cer, _ = _run_pipeline(
            GRUDecoder, synthetic_data, reference_texts, fs,
        )
        assert isinstance(cer, float)
        # With random weights, CER should be high but computable
        assert not np.isnan(cer)
        assert not np.isinf(cer)


class TestEndToEndCNNLSTM:
    """End-to-end integration test with the CNN+LSTM (Model B)."""

    def test_full_pipeline(self, synthetic_data, reference_texts, fs):
        decoded, cer, loss = _run_pipeline(
            CNNLSTM, synthetic_data, reference_texts, fs,
            conv_channels=64, lstm_hidden=64,
        )
        assert len(decoded) == 4
        assert cer >= 0.0
        assert loss > 0.0

    def test_decoded_outputs_are_strings(self, synthetic_data, reference_texts, fs):
        decoded, _, _ = _run_pipeline(
            CNNLSTM, synthetic_data, reference_texts, fs,
            conv_channels=64, lstm_hidden=64,
        )
        for s in decoded:
            assert isinstance(s, str)
            for ch in s:
                assert ch in "abcdefghijklmnopqrstuvwxyz "

    def test_cer_does_not_error(self, synthetic_data, reference_texts, fs):
        _, cer, _ = _run_pipeline(
            CNNLSTM, synthetic_data, reference_texts, fs,
            conv_channels=64, lstm_hidden=64,
        )
        assert isinstance(cer, float)
        assert not np.isnan(cer)
        assert not np.isinf(cer)


class TestPipelineConsistency:
    """Cross-model consistency checks."""

    def test_both_models_produce_valid_output(self, synthetic_data, reference_texts, fs):
        """Both models should produce valid decoded strings and finite CER."""
        decoded_gru, cer_gru, _ = _run_pipeline(
            GRUDecoder, synthetic_data, reference_texts, fs,
        )
        decoded_cnn, cer_cnn, _ = _run_pipeline(
            CNNLSTM, synthetic_data, reference_texts, fs,
            conv_channels=64, lstm_hidden=64,
        )

        # Both should produce lists of strings
        assert len(decoded_gru) == len(decoded_cnn) == 4
        assert isinstance(cer_gru, float) and isinstance(cer_cnn, float)

    def test_preprocessing_preserves_shape(self, synthetic_data, fs):
        """Preprocessing should preserve [T, C] shape per trial."""
        for i in range(synthetic_data.shape[0]):
            trial = synthetic_data[i]
            filtered = bandpass_filter(trial, fs=fs, low=1.0, high=100.0)
            filtered = notch_filter(filtered, fs=fs, freqs=[60.0])
            assert filtered.shape == trial.shape

    def test_feature_extraction_reduces_time(self, synthetic_data, fs):
        """Firing rate binning should reduce the time dimension."""
        trial = synthetic_data[0]
        features = compute_firing_rate_features(trial, bin_width_ms=10.0, fs=fs)
        assert features.shape[1] == trial.shape[1]  # channels preserved
        # With bin_width_ms=10 and fs=250, bin_size=2, so T'=T//2
        assert features.shape[0] <= trial.shape[0]
