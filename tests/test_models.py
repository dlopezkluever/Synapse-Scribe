"""Tests for all model architectures — GRU, CNN+LSTM, Transformer, CNN-Transformer."""

import pytest
import torch

from src.models.base import BaseDecoder
from src.models.gru_decoder import GRUDecoder
from src.models.cnn_lstm import CNNLSTM
from src.models.transformer import TransformerDecoder
from src.models.cnn_transformer import CNNTransformer


@pytest.fixture
def dummy_input():
    """Standard dummy input: [batch=4, time=2000, channels=192]."""
    return torch.randn(4, 2000, 192)


@pytest.fixture
def small_input():
    """Smaller input for faster tests: [batch=2, time=100, channels=32]."""
    return torch.randn(2, 100, 32)


# --- GRU Decoder (Model A) ---

class TestGRUDecoder:
    def test_output_shape(self, dummy_input):
        model = GRUDecoder(n_channels=192, n_classes=28)
        logits = model(dummy_input)
        assert logits.shape == (4, 2000, 28)

    def test_output_shape_small(self, small_input):
        model = GRUDecoder(n_channels=32, n_classes=28)
        logits = model(small_input)
        assert logits.shape == (2, 100, 28)

    def test_gradient_flow(self, small_input):
        model = GRUDecoder(n_channels=32, n_classes=28)
        logits = model(small_input)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_is_base_decoder(self):
        model = GRUDecoder()
        assert isinstance(model, BaseDecoder)

    def test_count_parameters(self):
        model = GRUDecoder(n_channels=192, n_classes=28)
        n_params = model.count_parameters()
        assert n_params > 0
        assert n_params > 1_000_000


# --- CNN+LSTM (Model B) ---

class TestCNNLSTM:
    def test_output_shape(self, dummy_input):
        model = CNNLSTM(n_channels=192, n_classes=28)
        logits = model(dummy_input)
        assert logits.shape == (4, 2000, 28)

    def test_output_shape_small(self, small_input):
        model = CNNLSTM(n_channels=32, n_classes=28, conv_channels=64, lstm_hidden=64)
        logits = model(small_input)
        assert logits.shape == (2, 100, 28)

    def test_gradient_flow(self, small_input):
        model = CNNLSTM(n_channels=32, n_classes=28, conv_channels=64, lstm_hidden=64)
        logits = model(small_input)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_is_base_decoder(self):
        model = CNNLSTM()
        assert isinstance(model, BaseDecoder)

    def test_count_parameters(self):
        model = CNNLSTM(n_channels=192, n_classes=28)
        n_params = model.count_parameters()
        assert n_params > 0
        assert n_params > 1_000_000


# --- Transformer Encoder (Model C) ---

class TestTransformerDecoder:
    def test_output_shape(self):
        model = TransformerDecoder(n_channels=192, n_classes=28)
        x = torch.randn(2, 200, 192)
        logits = model(x)
        assert logits.shape == (2, 200, 28)

    def test_output_shape_small(self, small_input):
        model = TransformerDecoder(
            n_channels=32, n_classes=28,
            d_model=64, n_heads=4, n_layers=2, ffn_dim=128,
        )
        logits = model(small_input)
        assert logits.shape == (2, 100, 28)

    def test_gradient_flow(self, small_input):
        model = TransformerDecoder(
            n_channels=32, n_classes=28,
            d_model=64, n_heads=4, n_layers=2, ffn_dim=128,
        )
        logits = model(small_input)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_is_base_decoder(self):
        model = TransformerDecoder(
            n_channels=32, d_model=64, n_heads=4, n_layers=2, ffn_dim=128,
        )
        assert isinstance(model, BaseDecoder)

    def test_count_parameters(self):
        model = TransformerDecoder(n_channels=192, n_classes=28)
        n_params = model.count_parameters()
        assert n_params > 0
        assert n_params > 1_000_000

    def test_with_padding_mask(self, small_input):
        model = TransformerDecoder(
            n_channels=32, n_classes=28,
            d_model=64, n_heads=4, n_layers=2, ffn_dim=128,
        )
        # Mask: True means padded (ignored)
        mask = torch.zeros(2, 100, dtype=torch.bool)
        mask[0, 80:] = True  # first sample padded from position 80
        mask[1, 90:] = True
        logits = model(small_input, src_key_padding_mask=mask)
        assert logits.shape == (2, 100, 28)

    def test_preserves_time_dimension(self):
        """Transformer should not change the time dimension."""
        model = TransformerDecoder(
            n_channels=32, d_model=64, n_heads=4, n_layers=2, ffn_dim=128,
        )
        for T in [50, 100, 200]:
            x = torch.randn(1, T, 32)
            out = model(x)
            assert out.shape[1] == T


# --- Hybrid CNN-Transformer (Model D) ---

class TestCNNTransformer:
    def test_output_shape_standard(self):
        model = CNNTransformer(n_channels=192, n_classes=28)
        x = torch.randn(2, 2000, 192)
        logits = model(x)
        # 8x temporal reduction: 2000 // 8 = 250
        assert logits.shape == (2, 250, 28)

    def test_output_shape_small(self, small_input):
        model = CNNTransformer(
            n_channels=32, n_classes=28,
            cnn_channels=64, d_model=64, n_heads=4,
            n_transformer_layers=2, ffn_dim=128,
        )
        logits = model(small_input)
        # 100 // 8 = 12
        assert logits.shape == (2, 12, 28)

    def test_temporal_reduction_factor(self):
        """Verify 8x temporal reduction from 3 MaxPool(2) layers."""
        model = CNNTransformer(
            n_channels=32, n_classes=28,
            cnn_channels=64, d_model=64, n_heads=4,
            n_transformer_layers=2, ffn_dim=128,
        )
        for T in [80, 160, 240, 2000]:
            x = torch.randn(1, T, 32)
            out = model(x)
            assert out.shape[1] == T // 8

    def test_gradient_flow(self, small_input):
        model = CNNTransformer(
            n_channels=32, n_classes=28,
            cnn_channels=64, d_model=64, n_heads=4,
            n_transformer_layers=2, ffn_dim=128,
        )
        logits = model(small_input)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_is_base_decoder(self):
        model = CNNTransformer(
            n_channels=32, cnn_channels=64, d_model=64, n_heads=4,
            n_transformer_layers=2, ffn_dim=128,
        )
        assert isinstance(model, BaseDecoder)

    def test_count_parameters(self):
        model = CNNTransformer(n_channels=192, n_classes=28)
        n_params = model.count_parameters()
        assert n_params > 0
        assert n_params > 1_000_000

    def test_2000_to_250(self):
        """Explicit test matching the spec: T=2000 → T=250."""
        model = CNNTransformer(n_channels=192, n_classes=28)
        x = torch.randn(4, 2000, 192)
        logits = model(x)
        assert logits.shape == (4, 250, 28)
