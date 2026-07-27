"""Regression tests for sparse target-event files in :mod:`daly2020`."""

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from moabb.datasets.daly2020 import Daly2020


def _raw_with_events(descriptions):
    """Return a small raw object whose annotations are Daly event codes."""
    info = mne.create_info(["C3", "C4"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 500)), info, verbose=False)
    if descriptions:
        raw.set_annotations(
            mne.Annotations(
                onset=np.arange(len(descriptions), dtype=float),
                duration=np.zeros(len(descriptions)),
                description=descriptions,
            )
        )
    return raw


def _paths(*run_numbers):
    return [SimpleNamespace(task=f"run{run_number}") for run_number in run_numbers]


def test_daly2020_skips_only_targetless_noncalibration_runs(monkeypatch):
    """A released empty events.tsv is skipped and valid runs stay re-indexed."""
    dataset = Daly2020()
    paths = _paths(1, 2, 3, 4)
    raws = {
        "run2": _raw_with_events([]),
        "run3": _raw_with_events(["1", "2"]),
        "run4": _raw_with_events(["2"]),
    }
    monkeypatch.setattr(dataset, "bids_paths", lambda subject: paths)
    monkeypatch.setattr(dataset, "_read_raw_bids", lambda path: raws[path.task])

    runs = dataset._get_single_subject_data(5)["0"]

    assert list(runs) == ["0", "1"]
    assert list(runs["0"].annotations.description) == ["right_hand", "relax"]
    assert list(runs["1"].annotations.description) == ["relax"]


def test_daly2020_raises_when_no_noncalibration_run_has_targets(monkeypatch):
    """A subject with only targetless runs gets an actionable loader error."""
    dataset = Daly2020()
    paths = _paths(1, 2, 3)
    raws = {"run2": _raw_with_events([]), "run3": _raw_with_events([])}
    monkeypatch.setattr(dataset, "bids_paths", lambda subject: paths)
    monkeypatch.setattr(dataset, "_read_raw_bids", lambda path: raws[path.task])

    with pytest.raises(ValueError, match="subject 5 has no usable motor-imagery runs"):
        dataset._get_single_subject_data(5)
