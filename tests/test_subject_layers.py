"""Tests for multi-subject training (subject layers + multi-subject utilities).

Verifies SubjectNormalization, SubjectAwareModel, subject-aware datasets and
dataloaders, and per-subject evaluation logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.models.subject_layers import SubjectAwareModel, SubjectNormalization
from src.models.gru_decoder import GRUDecoder
from src.training.multi_subject import (
    SubjectAwareDataset,
    build_subject_map,
    create_multi_subject_dataloaders,
    evaluate_per_subject,
    subject_aware_collate_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    return GRUDecoder(n_channels=4, n_classes=28, proj_dim=8, hidden_size=16, n_layers=1, dropout=0.0, use_downsample=False)


@pytest.fixture
def subject_model(tiny_model):
    return SubjectAwareModel(tiny_model, n_subjects=3, n_channels=4)


@pytest.fixture
def multi_subject_trial_index(tmp_path):
    """Create a minimal multi-subject trial index with real .npy/.txt files."""
    rows = []
    for subj in ["s01", "s02", "s03"]:
        for trial in range(4):
            sig_dir = tmp_path / subj / "neural"
            sig_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir = tmp_path / subj / "labels"
            lbl_dir.mkdir(parents=True, exist_ok=True)

            sig_path = sig_dir / f"trial_{trial:04d}_signals.npy"
            lbl_path = lbl_dir / f"trial_{trial:04d}_transcript.txt"

            np.save(sig_path, np.random.randn(20, 4).astype(np.float32))
            lbl_path.write_text("ab", encoding="utf-8")

            rows.append({
                "subject": subj,
                "session": "1",
                "trial_id": trial,
                "signal_path": str(sig_path),
                "label_path": str(lbl_path),
                "n_timesteps": 20,
                "n_channels": 4,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SubjectNormalization
# ---------------------------------------------------------------------------

class TestSubjectNormalization:
    def test_output_shape(self):
        layer = SubjectNormalization(n_subjects=3, n_channels=8)
        x = torch.randn(4, 10, 8)
        ids = torch.tensor([0, 1, 2, 0])
        out = layer(x, ids)
        assert out.shape == (4, 10, 8)

    def test_identity_at_init(self):
        """At initialization (gamma=1, beta=0), output should equal input."""
        layer = SubjectNormalization(n_subjects=2, n_channels=4)
        x = torch.randn(2, 5, 4)
        ids = torch.tensor([0, 1])
        out = layer(x, ids)
        torch.testing.assert_close(out, x)

    def test_different_subjects_differ(self):
        """After training, different subjects should produce different outputs."""
        layer = SubjectNormalization(n_subjects=2, n_channels=4)
        # Manually set different parameters
        with torch.no_grad():
            layer.gamma[0] = torch.tensor([2.0, 2.0, 2.0, 2.0])
            layer.gamma[1] = torch.tensor([0.5, 0.5, 0.5, 0.5])

        x = torch.ones(2, 3, 4)
        ids = torch.tensor([0, 1])
        out = layer(x, ids)
        assert not torch.allclose(out[0], out[1])

    def test_gradient_flows(self):
        layer = SubjectNormalization(n_subjects=2, n_channels=4)
        x = torch.randn(2, 5, 4, requires_grad=True)
        ids = torch.tensor([0, 1])
        out = layer(x, ids)
        loss = out.sum()
        loss.backward()
        assert layer.gamma.grad is not None
        assert layer.beta.grad is not None


# ---------------------------------------------------------------------------
# SubjectAwareModel
# ---------------------------------------------------------------------------

class TestSubjectAwareModel:
    def test_forward_with_subject_ids(self, subject_model):
        x = torch.randn(3, 10, 4)
        ids = torch.tensor([0, 1, 2])
        out = subject_model(x, ids)
        assert out.shape == (3, 10, 28)

    def test_forward_without_subject_ids(self, subject_model):
        """When subject_ids is None, should bypass normalization."""
        x = torch.randn(2, 10, 4)
        out = subject_model(x, subject_ids=None)
        assert out.shape == (2, 10, 28)

    def test_count_parameters(self, subject_model):
        n = subject_model.count_parameters()
        assert n > 0

    def test_gradient_flow(self, subject_model):
        x = torch.randn(2, 10, 4)
        ids = torch.tensor([0, 1])
        out = subject_model(x, ids)
        loss = out.sum()
        loss.backward()
        # Check gradients flow through subject norm and base model
        assert subject_model.subject_norm.gamma.grad is not None
        for p in subject_model.base_model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break


# ---------------------------------------------------------------------------
# Multi-subject data utilities
# ---------------------------------------------------------------------------

class TestBuildSubjectMap:
    def test_maps_subjects(self, multi_subject_trial_index):
        smap = build_subject_map(multi_subject_trial_index)
        assert len(smap) == 3
        assert set(smap.values()) == {0, 1, 2}

    def test_sorted_order(self, multi_subject_trial_index):
        smap = build_subject_map(multi_subject_trial_index)
        assert smap["s01"] < smap["s02"] < smap["s03"]


class TestSubjectAwareDataset:
    def test_returns_subject_id(self, multi_subject_trial_index):
        smap = build_subject_map(multi_subject_trial_index)
        ds = SubjectAwareDataset(multi_subject_trial_index, smap, t_max=20)
        item = ds[0]
        assert "subject_id" in item
        assert isinstance(item["subject_id"], int)

    def test_all_items_have_subject_id(self, multi_subject_trial_index):
        smap = build_subject_map(multi_subject_trial_index)
        ds = SubjectAwareDataset(multi_subject_trial_index, smap, t_max=20)
        for i in range(len(ds)):
            assert "subject_id" in ds[i]


class TestSubjectAwareCollate:
    def test_collate_includes_subject_ids(self, multi_subject_trial_index):
        smap = build_subject_map(multi_subject_trial_index)
        ds = SubjectAwareDataset(multi_subject_trial_index, smap, t_max=20)
        batch = [ds[0], ds[1]]
        collated = subject_aware_collate_fn(batch)
        assert "subject_ids" in collated
        assert collated["subject_ids"].shape == (2,)


class TestCreateMultiSubjectDataloaders:
    def test_returns_loaders_and_map(self, multi_subject_trial_index):
        train_loader, val_loader, test_loader, smap = create_multi_subject_dataloaders(
            multi_subject_trial_index, t_max=20, batch_size=4,
        )
        assert len(smap) == 3
        # At least train loader should have data
        batch = next(iter(train_loader))
        assert "subject_ids" in batch


# ---------------------------------------------------------------------------
# Per-subject evaluation
# ---------------------------------------------------------------------------

class TestEvaluatePerSubject:
    def test_returns_per_subject_cer(self, multi_subject_trial_index, tiny_model):
        smap = build_subject_map(multi_subject_trial_index)
        ds = SubjectAwareDataset(multi_subject_trial_index, smap, t_max=20)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=subject_aware_collate_fn,
        )
        result = evaluate_per_subject(tiny_model, loader, device="cpu")
        assert "overall_cer" in result
        assert "per_subject" in result
        assert isinstance(result["overall_cer"], float)

    def test_per_subject_has_cer_and_count(self, multi_subject_trial_index, tiny_model):
        smap = build_subject_map(multi_subject_trial_index)
        ds = SubjectAwareDataset(multi_subject_trial_index, smap, t_max=20)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=subject_aware_collate_fn,
        )
        result = evaluate_per_subject(tiny_model, loader, device="cpu")
        for sid, info in result["per_subject"].items():
            assert "cer" in info
            assert "n_trials" in info
            assert info["n_trials"] > 0
