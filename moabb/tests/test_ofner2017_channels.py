"""Offline regressions: every Ofner2017 run has the same channel layout.

Subject 1's released execution runs store the EOG channels as eog-r, eog-m,
eog-l and its imagination runs hold "armeodummy-0".."armeodummy-12" instead
of the Armeo channels. With ``return_all_modalities`` the arrays swapped the
two EOG channels and ``return_epochs`` failed with "epochs[10].info['ch_names']
must match". No downloads: the GDF reads are synthetic.
"""

import mne
import numpy as np
import pytest

from moabb.datasets import Ofner2017, upper_limb
from moabb.datasets.base import BaseDataset
from moabb.datasets.upper_limb import _ARMEO
from moabb.paradigms import MotorImagery


_SFREQ = 512.0
_EEG = [f"E{i}" for i in range(61)]
_FINGERS = [f"finger{i}" for i in range(19)]
_EOG = ["eog-l", "eog-m", "eog-r"]
_DUMMY = [f"armeodummy-{i}" for i in range(13)]
_ONSETS = 2.0 + 7.0 * np.arange(7)
_CODES = [str(c) for c in range(1536, 1543)]


def _names(path):
    if path.startswith("execution_s1"):
        return _EEG + _EOG[::-1] + _FINGERS + _ARMEO
    if path.startswith("imagination_s1"):
        return _EEG + _EOG + _FINGERS + _DUMMY
    return _EEG + _EOG + _FINGERS + _ARMEO


def _signal(name, n):
    """Content that identifies the channel; Armeo channels are NaN as released."""
    if name in _ARMEO:
        return np.full(n, np.nan)
    if name in _DUMMY:
        return np.random.default_rng(len(name)).uniform(-50, 50, n)
    freq = 2.0 + 0.1 * sorted(_EEG + _EOG + _FINGERS).index(name)
    return 10 * np.sin(2 * np.pi * freq * np.arange(n) / _SFREQ)


def _gdf_like(path, eog=(), misc=(), **kwargs):
    names = _names(path)
    n = int((_ONSETS[-1] + 8) * _SFREQ)
    types = [
        "eog" if c in eog else "misc" if i in misc else "eeg" for i, c in enumerate(names)
    ]
    raw = mne.io.RawArray(
        np.array([_signal(c, n) for c in names]),
        mne.create_info(names, _SFREQ, types),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations(_ONSETS, 0.0, _CODES))
    return raw


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(BaseDataset, "_prefetch_nemar_sourcedata", lambda *a, **k: None)
    monkeypatch.setattr(upper_limb, "read_raw_gdf", _gdf_like)
    monkeypatch.setattr(upper_limb, "make_standard_montage", lambda name: None)
    monkeypatch.setattr(mne.io.BaseRaw, "set_montage", lambda self, m, **k: self)
    ds = Ofner2017(return_all_modalities=True)
    monkeypatch.setattr(
        ds,
        "data_path",
        lambda subject, session=None, **k: [f"{session}_s{subject}_r{r}" for r in (1, 2)],
    )
    return ds


def _layout(raw):
    return list(zip(raw.ch_names, raw.get_channel_types()))


def test_subject_1_runs_get_the_layout_of_the_other_subjects(dataset):
    reference = dataset._get_single_subject_data(2)["0execution"]["0"]
    for session, runs in dataset._get_single_subject_data(1).items():
        for raw in runs.values():
            assert _layout(raw) == _layout(reference), session
            # The EOG channels moved with their labels, not just their names.
            n = raw.n_times
            for ch in _EOG:
                np.testing.assert_allclose(
                    raw.get_data(picks=[ch])[0], _signal(ch, n) * 1e-6
                )
            # Armeo channels are 0: placeholders zeroed, released NaN set to 0.
            assert not raw.get_data(picks=_ARMEO).any()


def test_other_subjects_are_unchanged(dataset):
    raw = dataset._get_single_subject_data(2)["1imagination"]["0"]
    assert raw.ch_names[:-1] == _names("imagination_s2_r1")
    n = raw.n_times
    np.testing.assert_allclose(
        raw.get_data(picks=_EEG + _EOG),
        np.array([_signal(c, n) for c in _EEG + _EOG]) * 1e-6,
    )


@pytest.mark.parametrize("modalities", [True, {"eeg": True, "eog": True}])
def test_all_modalities_concatenate_across_runs_and_subjects(dataset, modalities):
    dataset.return_all_modalities = modalities
    epochs, _, meta = MotorImagery().get_data(
        dataset, subjects=[1, 2], return_epochs=True
    )
    assert len(epochs) == 2 * 2 * 2 * len(_ONSETS)
    assert [c for c in epochs.ch_names if c.startswith("eog")] == _EOG


def test_metadata_lists_the_returned_misc_channels():
    sensors = Ofner2017.METADATA.acquisition.sensors
    assert set(_ARMEO) <= set(sensors)
    assert not [s for s in sensors if s.startswith("armeodummy")]
    assert len(sensors) == 96
