"""Offline regressions: PerezBlanco2026 returns one sampling rate.

One released run (sub-09 run-01) is stored at ~1200 Hz while every other run
is at 512 Hz. No downloads: EDF reads are synthetic.
"""

from unittest.mock import Mock

import mne
import numpy as np
import pytest

from moabb.datasets import PerezBlanco2026
from moabb.datasets.base import BaseDataset
from moabb.datasets.perezblanco2026 import _EEG_NAMES, _EMG_NAMES
from moabb.paradigms import MotorImagery


# The header rate of the off-rate run (8 samples per 0.00667 s EDF record).
_ODD_SFREQ = 8 / 0.00667
_ONSETS_S = [3.0, 13.0, 23.0, 33.0]
_CODES = [1, 2, 3, 4]


def _edf_like(sfreq, seconds=45.0):
    """A BCI2000->EDF run: 16 data channels then the two state channels used."""
    names = _EEG_NAMES + _EMG_NAMES + ["CurrentTarget", "TrialInitMovStam"]
    n = int(round(seconds * sfreq))
    rng = np.random.default_rng(int(sfreq))
    data = rng.standard_normal((len(names), n)) * 1e-5
    data[-2:] = 0
    for t, code in zip(_ONSETS_S, _CODES):
        s = int(round(t * sfreq))
        data[-2, s : s + int(sfreq)] = code
        data[-1, s : s + int(sfreq)] = 1
    return mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose=False)


def test_off_rate_run_is_resampled_with_events_in_place():
    raw = PerezBlanco2026()._build_run(_edf_like(_ODD_SFREQ))
    assert raw.info["sfreq"] == 512.0
    assert raw.ch_names == _EEG_NAMES + ["STIM"]
    events = mne.find_events(raw, shortest_event=1, verbose=False)
    np.testing.assert_array_equal(events[:, 2], _CODES)
    # Event times are kept to within one sample of the native onset.
    np.testing.assert_allclose(events[:, 0] / 512.0, _ONSETS_S, atol=1 / 512)


def test_512hz_run_is_untouched():
    native = _edf_like(512.0)
    expected = native.get_data(picks=_EEG_NAMES)
    raw = PerezBlanco2026()._build_run(native)
    assert raw.info["sfreq"] == 512.0
    np.testing.assert_array_equal(raw.get_data(picks=_EEG_NAMES), expected)
    events = mne.find_events(raw, shortest_event=1, verbose=False)
    np.testing.assert_array_equal(events[:, 0], np.round(np.array(_ONSETS_S) * 512))


def test_paradigm_stacks_subject_with_off_rate_run(monkeypatch):
    """Subject 9 (one ~1200 Hz run) with subject 8: the call that crashed."""
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(BaseDataset, "_prefetch_nemar_sourcedata", lambda *a, **k: None)
    files = {8: ["s8_run-01.edf", "s8_run-02.edf"], 9: ["s9_run-01.edf", "s9_run-02.edf"]}
    monkeypatch.setattr(
        mne.io,
        "read_raw_edf",
        lambda path, **k: _edf_like(_ODD_SFREQ if path == "s9_run-01.edf" else 512.0),
    )
    ds = PerezBlanco2026()
    monkeypatch.setattr(ds, "data_path", Mock(side_effect=lambda s, *a, **k: files[s]))
    X, y, meta = MotorImagery(n_classes=4).get_data(ds, subjects=[8, 9])
    assert X.shape == (16, 8, 1281)
    assert sorted(set(meta.subject)) == [8, 9]
    epochs, _, _ = MotorImagery(n_classes=4).get_data(
        ds, subjects=[9], return_epochs=True
    )
    assert epochs.info["sfreq"] == 512.0
    assert len(epochs) == 8


@pytest.mark.parametrize("sfreq", [256.0, 1024.0])
def test_any_other_rate_is_brought_to_512(sfreq):
    assert PerezBlanco2026()._build_run(_edf_like(sfreq)).info["sfreq"] == 512.0
