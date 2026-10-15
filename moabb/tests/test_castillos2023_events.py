"""Offline regressions: the Castillos2023 c-VEP loaders read their trials.

``labels = epochs.events[..., -1]`` was a view, so making the labels 0-based
also rewrote ``epochs.events`` and every subject failed when the epochs were
read ("Missing event codes: [0]"). No downloads: the EEGLAB reads are
synthetic, with annotations shaped like the released ``.set`` files.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import (
    CastillosBurstVEP40,
    CastillosBurstVEP100,
    CastillosCVEP40,
    CastillosCVEP100,
)
from moabb.datasets.base import BaseDataset
from moabb.paradigms import CVEP


_SFREQ = 500.0
_N_BITS = 132  # 2.2 s at 60 Hz
_N_TRIALS = 8
_TRIAL_ONSETS = 2.0 + 3.0 * np.arange(_N_TRIALS)
_CODES = np.random.default_rng(0).integers(0, 2, size=(4, _N_BITS))
# Channel labels of subject 1's released files (BrainProducts LiveAmp 32).
_NAMES = (
    "Fp1 Fz F3 F7 F9 FC5 FC1 C3 T7 CP5 CP1 Pz P3 P7 P9 O1 Oz O2 P10 P8 P4 CP2 "
    "CP6 T8 C4 Cz FC2 FC6 F10 F8 F4 Fp2"
).split()


def _description(target):
    bits = " ".join(f"{b}." for b in _CODES[target])
    return f"[{bits}]_{target + 1}"


# events_from_annotations numbers the trial descriptions in sorted order, and
# the loader's 0-based label of a trial is that rank.
_RANK = {
    t: sorted(_description(k) for k in range(4)).index(_description(t)) for t in range(4)
}


def _set_like(path, *args, **kwargs):
    n = int((_TRIAL_ONSETS[-1] + 4.0) * _SFREQ)
    data = np.random.default_rng(1).standard_normal((len(_NAMES), n)) * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(_NAMES, _SFREQ, "eeg"), verbose=False)
    onsets, desc = [], []
    for i, t in enumerate(_TRIAL_ONSETS):
        onsets += [t - 1.0, t, t + 2.4]
        desc += ["collects", _description(i % 4), "iti"]
    raw.set_annotations(mne.Annotations(onsets, 0.0, desc))
    return raw


def _offline(monkeypatch, cls):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(BaseDataset, "_prefetch_nemar_sourcedata", lambda *a, **k: None)
    monkeypatch.setattr(mne.io, "read_raw_eeglab", _set_like)
    ds = cls()
    monkeypatch.setattr(ds, "data_path", lambda subject, *a, **k: [f"P{subject}.set"])
    return ds


@pytest.mark.parametrize(
    "cls", [CastillosBurstVEP100, CastillosBurstVEP40, CastillosCVEP100, CastillosCVEP40]
)
def test_subject_loads_with_zero_based_trial_labels(monkeypatch, cls):
    ds = _offline(monkeypatch, cls)
    raw = ds._get_single_subject_data(1)["0"]["0"]
    meta = [a for a in raw.annotations if a["description"] == "_trial_meta"]
    np.testing.assert_allclose([a["onset"] for a in meta], _TRIAL_ONSETS)
    labels = [_RANK[i % 4] for i in range(_N_TRIALS)]
    assert [a["extras"]["trial_id"] for a in meta] == labels
    # stim_trial holds 200 + the 0-based label at each trial onset.
    trials = mne.find_events(raw, stim_channel="stim_trial", verbose=False)
    np.testing.assert_array_equal(trials[:, 0], (_TRIAL_ONSETS * _SFREQ).astype(int))
    np.testing.assert_array_equal(trials[:, 2], [200 + lab for lab in labels])
    # The codebook is indexed by the same 0-based labels.
    assert sorted(ds.codes) == [0, 1, 2, 3]
    for target in range(4):
        np.testing.assert_array_equal(ds.codes[_RANK[target]], _CODES[target])


def test_paradigm_returns_one_epoch_per_flash(monkeypatch):
    ds = _offline(monkeypatch, CastillosBurstVEP100)
    raw = ds._get_single_subject_data(1)["0"]["0"]
    n_flashes = len(mne.find_events(raw, stim_channel="stim_epoch", verbose=False))
    X, y, meta = CVEP().get_data(ds, subjects=[1, 2])
    assert X.shape == (2 * n_flashes, len(_NAMES), int(0.25 * _SFREQ) + 1)
    assert set(y) == {"0", "1"}
