"""Offline regressions: PhysionetMI returns 160 Hz for every subject.

Subjects 88, 92 and 100 were recorded at 128 Hz, all other subjects at
160 Hz. No downloads: EDF reads are synthetic.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import PhysionetMI, physionet_mi
from moabb.datasets.base import BaseDataset
from moabb.paradigms import LeftRightImagery, MotorImagery


_SENSORS = PhysionetMI.METADATA.acquisition.sensors
_ONSETS = [0.0, 4.2, 8.4, 12.6, 16.8, 21.0]
_DESC = ["T0", "T1", "T0", "T2", "T0", "T1"]
_AT_128 = (88, 92, 100)


def _edf_like(path, *args, **kwargs):
    sfreq = 128.0 if any(f"S{s:03d}" in path for s in _AT_128) else 160.0
    n = int(30 * sfreq)
    names = [c.upper() + "." for c in _SENSORS]  # EDF labels, e.g. "FC5."
    data = np.random.default_rng(n).standard_normal((len(names), n)) * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations(_ONSETS, 4.1, _DESC))
    return raw


def _offline(monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(BaseDataset, "_prefetch_nemar_sourcedata", lambda *a, **k: None)
    monkeypatch.setattr(physionet_mi, "read_raw_edf", _edf_like)
    ds = PhysionetMI()
    monkeypatch.setattr(
        ds,
        "_load_data",
        lambda subject, runs, **k: [
            f"S{subject:03d}/S{subject:03d}R{r:02d}.edf" for r in runs
        ],
    )
    return ds


@pytest.mark.parametrize("subject", _AT_128)
def test_128hz_subjects_are_resampled_to_160hz(monkeypatch, subject):
    ds = _offline(monkeypatch)
    runs = ds._get_single_subject_data(subject)["0"]
    assert len(runs) == 6
    for raw in runs.values():
        assert raw.info["sfreq"] == 160.0
        assert raw.n_times == 30 * 160
        assert raw.ch_names[:64] == _SENSORS
        np.testing.assert_allclose(raw.annotations.onset, _ONSETS)


def test_160hz_subject_is_untouched(monkeypatch):
    ds = _offline(monkeypatch)
    raw = ds._get_single_subject_data(1)["0"]["0"]
    expected = _edf_like("S001/S001R04.edf").get_data()
    assert raw.info["sfreq"] == 160.0
    np.testing.assert_array_equal(raw.get_data(picks="eeg"), expected)


def test_paradigm_stacks_128hz_subjects_with_others(monkeypatch):
    """The call that crashed: 481 vs 385 samples per 3 s epoch."""
    ds = _offline(monkeypatch)
    subjects = [1, *_AT_128]
    X, y, meta = LeftRightImagery().get_data(ds, subjects=subjects)
    assert X.shape == (len(subjects) * 3 * 3, 64, 481)
    assert sorted(set(meta.subject)) == subjects
    epochs, _, _ = MotorImagery().get_data(
        ds, subjects=[88, 1, 92, 100], return_epochs=True
    )
    assert epochs.info["sfreq"] == 160.0
    assert sorted(set(epochs.metadata.subject)) == subjects
