"""Tests for UCSF ECoG and OpenNeuro dataset integration in src/data/loader.py."""

import numpy as np
import pytest
import pandas as pd
from pathlib import Path

from src.data.loader import (
    download_ucsf_ecog,
    preprocess_ecog,
    download_openneuro,
    _parse_bids_events,
    _parse_bids_path,
)


class TestDownloadUCSFECoG:
    def test_creates_directory(self, tmp_path):
        dataset_dir = download_ucsf_ecog(data_root=tmp_path)
        assert dataset_dir.exists()
        assert (dataset_dir / "raw").exists()

    def test_returns_path(self, tmp_path):
        result = download_ucsf_ecog(data_root=tmp_path)
        assert isinstance(result, Path)


class TestPreprocessECoG:
    def test_output_shape(self):
        signals = np.random.randn(1000, 64).astype(np.float32)
        result = preprocess_ecog(signals, fs=1000.0)
        assert result.shape == signals.shape

    def test_nonnegative_amplitude(self):
        signals = np.random.randn(500, 32).astype(np.float32)
        result = preprocess_ecog(signals, fs=1000.0)
        assert (result >= 0).all()

    def test_dtype_preserved(self):
        signals = np.random.randn(500, 16).astype(np.float32)
        result = preprocess_ecog(signals, fs=1000.0)
        assert result.dtype == np.float32

    def test_low_fs_graceful(self):
        # fs too low for 70-150 Hz bandpass
        signals = np.random.randn(100, 8).astype(np.float32)
        result = preprocess_ecog(signals, fs=100.0)
        assert result.shape == signals.shape


class TestDownloadOpenNeuro:
    def test_creates_directory(self, tmp_path):
        dataset_dir = download_openneuro("ds003688", data_root=tmp_path)
        assert dataset_dir.exists()

    def test_custom_dataset_id(self, tmp_path):
        dataset_dir = download_openneuro("ds001234", data_root=tmp_path)
        assert "ds001234" in str(dataset_dir)


class TestParseBIDSPath:
    def test_with_subject_session(self):
        path = Path("/data/sub-01/ses-02/ieeg/recording.edf")
        subj, sess = _parse_bids_path(path)
        assert subj == 1
        assert sess == 2

    def test_without_session(self):
        path = Path("/data/sub-05/ieeg/recording.edf")
        subj, sess = _parse_bids_path(path)
        assert subj == 5
        assert sess == 1  # default

    def test_no_bids_structure(self):
        path = Path("/data/recording.edf")
        subj, sess = _parse_bids_path(path)
        assert subj == 1  # default
        assert sess == 1  # default


class TestParseBIDSEvents:
    def test_with_trial_type(self, tmp_path):
        events_file = tmp_path / "events.tsv"
        events_file.write_text("onset\tduration\ttrial_type\n1.0\t0.5\thello\n2.0\t0.5\tworld\n")
        labels = _parse_bids_events(events_file)
        assert labels == ["hello", "world"]

    def test_with_value_column(self, tmp_path):
        events_file = tmp_path / "events.tsv"
        events_file.write_text("onset\tduration\tvalue\n1.0\t0.5\ttest\n")
        labels = _parse_bids_events(events_file)
        assert labels == ["test"]

    def test_missing_file(self, tmp_path):
        labels = _parse_bids_events(tmp_path / "nonexistent.tsv")
        assert labels == []

    def test_empty_file(self, tmp_path):
        events_file = tmp_path / "events.tsv"
        events_file.write_text("")
        labels = _parse_bids_events(events_file)
        assert labels == []
