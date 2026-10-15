"""Offline regressions: Mainsah2025 runs all return exactly 256 Hz.

Each BigP3BCI EDF stores the run as one data record whose 8-character
duration field is truncated, so the header rate differs slightly from file to
file (35208 samples over "137.5312" s = 256.00009 Hz). No downloads: EDF reads
are synthetic.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import Mainsah2025_A
from moabb.datasets.base import BaseDataset
from moabb.datasets.mainsah2025 import Mainsah2025, _restore_nominal_rate
from moabb.paradigms import P300


# Header rates of real files: n_samples / truncated duration.
_RATES = {"r1.edf": 35208 / 137.5312, "r2.edf": 39944 / 156.0312, "r3.edf": 256.0}
_N = {"r1.edf": 35208, "r2.edf": 39944, "r3.edf": 2560 * 4}
_ONSETS = np.arange(512, 8192, 128)  # sample indices of flashes


def _edf_like(path, *args, **kwargs):
    name = path.rsplit("/", 1)[-1]
    sfreq, n = _RATES[name], _N[name]
    eeg = [f"EEG_{c}" for c in ("FP1", "Fz", "Cz", "Pz")]
    names = eeg + ["StimulusBegin", "StimulusType", "StimulusCode"]
    data = np.random.default_rng(n).standard_normal((len(names), n)) * 1e-5
    data[4:] = 0
    for i, s in enumerate(_ONSETS):
        data[4, s : s + 20] = 1  # StimulusBegin held for a few samples
        data[5, s : s + 20] = float(i % 6 == 0)  # one flash in six is a target
    return mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose=False)


def test_truncated_header_rate_is_relabelled_without_resampling():
    raw = _edf_like("x/r1.edf")
    assert raw.info["sfreq"] != 256.0
    data = raw.get_data()
    fixed = _restore_nominal_rate(raw, "x/r1.edf")
    assert fixed.info["sfreq"] == 256.0
    assert fixed.n_times == raw.n_times
    np.testing.assert_array_equal(fixed.get_data(), data)


def test_load_edf_returns_256hz_with_events_on_same_samples(monkeypatch):
    monkeypatch.setattr(mne.io, "read_raw_edf", _edf_like)
    for name in _RATES:
        raw = Mainsah2025._load_edf(f"x/{name}")
        assert raw.info["sfreq"] == 256.0
        assert raw.ch_names == ["Fp1", "Fz", "Cz", "Pz", "STI"]
        events = mne.find_events(raw, shortest_event=1, verbose=False)
        np.testing.assert_array_equal(events[:, 0], _ONSETS)
        assert (events[:, 2] == 2).sum() == len(_ONSETS[::6])


def test_other_rates_raise():
    raw = mne.io.RawArray(
        np.zeros((1, 2560)), mne.create_info(["EEG_Cz"], 250.0, "eeg"), verbose=False
    )
    with pytest.raises(ValueError, match="expected 256 Hz"):
        _restore_nominal_rate(raw, "x/odd.edf")


def test_epochs_of_runs_with_different_header_rates_concatenate(monkeypatch):
    """``return_epochs`` failed with "epochs[3].info['sfreq'] must match"."""
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(BaseDataset, "_prefetch_nemar_sourcedata", lambda *a, **k: None)
    monkeypatch.setattr(mne.io, "read_raw_edf", _edf_like)
    ds = Mainsah2025_A()
    monkeypatch.setattr(ds, "data_path", lambda *a, **k: [])
    files = {
        ds.subject_list[0]: {1: ["A/r1.edf", "A/r3.edf"]},
        ds.subject_list[1]: {1: ["A/r2.edf"]},
    }
    monkeypatch.setattr(ds, "_get_subject_manifest", lambda s: files[s])
    subjects = ds.subject_list[:2]
    epochs, y, meta = P300().get_data(ds, subjects=subjects, return_epochs=True)
    assert epochs.info["sfreq"] == 256.0
    assert len(epochs) == 3 * len(_ONSETS)
    X, _, _ = P300().get_data(ds, subjects=subjects)
    assert X.shape == (3 * len(_ONSETS), 4, 257)
