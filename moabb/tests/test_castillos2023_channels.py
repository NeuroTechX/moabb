"""Offline regressions: every Castillos2023 subject has the same channels.

The released ``.set`` files use two 32-channel layouts with different labels
and channel order (LiveAmp-32 with F9/P9/P10/F10 for subjects 1, 5, 6, 7, 11,
12; actiCAP-32 with FT9/TP9/TP10/FT10 for subjects 2, 3, 4, 8, 9, 10). Arrays
stacked a different electrode at the same index and ``return_epochs`` failed
with "epochs[1].info['ch_names'] must match". No downloads: the EEGLAB reads
are synthetic, with the channel order of the released files.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import CastillosCVEP40, castillos2023
from moabb.datasets.base import BaseDataset
from moabb.datasets.castillos2023 import _SHARED_EEG
from moabb.paradigms import CVEP


_SFREQ = 500.0
_LIVEAMP = (
    "Fp1 Fz F3 F7 F9 FC5 FC1 C3 T7 CP5 CP1 Pz P3 P7 P9 O1 Oz O2 P10 P8 P4 CP2 "
    "CP6 T8 C4 Cz FC2 FC6 F10 F8 F4 Fp2"
).split()
_ACTICAP = (
    "Fp1 Fz F3 F7 FT9 FC5 FC1 C3 T7 TP9 CP5 CP1 Pz P3 P7 O1 Oz O2 P4 P8 TP10 "
    "CP6 CP2 Cz C4 T8 FT10 FC6 FC2 F4 F8 Fp2"
).split()
_LAYOUT = dict.fromkeys((1, 5, 6, 7, 11, 12), _LIVEAMP)
_LAYOUT.update(dict.fromkeys((2, 3, 4, 8, 9, 10), _ACTICAP))
_TRIAL_ONSETS = 2.0 + 3.0 * np.arange(8)
_CODES = np.random.default_rng(0).integers(0, 2, size=(4, 132))


def _signal(name, n):
    """Channel content that identifies the electrode, whatever its index."""
    freq = 3.0 + 0.7 * sorted(set(_LIVEAMP + _ACTICAP)).index(name)
    return 1e-5 * np.sin(2 * np.pi * freq * np.arange(n) / _SFREQ)


def _set_like(path, *args, **kwargs):
    names = _LAYOUT[int(path.split(".")[0][1:])]
    n = int((_TRIAL_ONSETS[-1] + 4.0) * _SFREQ)
    data = np.array([_signal(c, n) for c in names])
    raw = mne.io.RawArray(data, mne.create_info(names, _SFREQ, "eeg"), verbose=False)
    desc = [
        "[" + " ".join(f"{b}." for b in _CODES[i % 4]) + f"]_{i % 4 + 1}"
        for i in range(8)
    ]
    raw.set_annotations(mne.Annotations(_TRIAL_ONSETS, 0.0, desc))
    return raw


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(BaseDataset, "_prefetch_nemar_sourcedata", lambda *a, **k: None)
    monkeypatch.setattr(mne.io, "read_raw_eeglab", _set_like)
    ds = CastillosCVEP40()
    monkeypatch.setattr(ds, "data_path", lambda subject, *a, **k: [f"P{subject}.set"])
    return ds


@pytest.mark.parametrize("subject", range(1, 13))
def test_every_subject_has_the_shared_channels_in_one_order(dataset, subject):
    raw = dataset._get_single_subject_data(subject)["0"]["0"]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    assert eeg == _SHARED_EEG
    n = raw.n_times
    expected = np.array([_signal(c, n) for c in _SHARED_EEG])
    np.testing.assert_array_equal(raw.get_data(picks="eeg"), expected)
    assert set(raw.ch_names) - set(eeg) == {"stim_trial", "stim_epoch"}


def test_metadata_matches_the_returned_channels():
    acq = CastillosCVEP40.METADATA.acquisition
    assert acq.channel_types == {"eeg": len(_SHARED_EEG)}
    assert sorted(acq.sensors) == sorted(_SHARED_EEG)


def test_subjects_of_both_layouts_concatenate(dataset):
    epochs, _, meta = CVEP().get_data(dataset, subjects=[1, 2], return_epochs=True)
    assert epochs.ch_names == _SHARED_EEG
    X, _, meta = CVEP().get_data(dataset, subjects=[1, 2])
    # Both subjects hold the same signal per electrode and the same flashes,
    # so the same index must give the same data.
    x1, x2 = (X[(meta.subject == s).to_numpy()] for s in (1, 2))
    assert x1.shape == x2.shape and x1.shape[1] == len(_SHARED_EEG)
    np.testing.assert_allclose(x1, x2)


@pytest.mark.parametrize(
    "names",
    [
        _LIVEAMP[:-1],  # a shared electrode missing
        _LIVEAMP[:28] + ["FT9", "TP9", "P10", "F10"],  # mixed layouts
    ],
)
def test_unknown_layout_raises(names):
    raw = mne.io.RawArray(
        np.zeros((len(names), 10)), mne.create_info(names, _SFREQ, "eeg"), verbose=False
    )
    with pytest.raises(ValueError, match="shared electrodes"):
        castillos2023._pick_shared_channels(raw, "odd.set")
