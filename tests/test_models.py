"""Tests for model architectures — GRU Decoder and CNN+LSTM."""

import pytest
import torch

from src.models.base import BaseDecoder
from src.models.gru_decoder import GRUDecoder
from src.models.cnn_lstm import CNNLSTM


@pytest.fixture
def dummy_input():
    """Standard dummy input: [batch=4, time=2000, channels=192]."""
    return torch.randn(4, 2000, 192)


@pytest.fixture
def small_input():
    """Smaller input for faster tests: [batch=2, time=100, channels=32]."""
    return torch.randn(2, 100, 32)


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
        # Check that gradients are present
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
        # GRU with 3 layers × 512 hidden should have several million params
        assert n_params > 1_000_000


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
