"""Offline synthetic regression tests for the MATLAB MI loader batch."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import mne
import numpy as np
import pytest
from scipy.io import savemat

from moabb.datasets import Jia2019, Ortiz2023, Yilmaz2024, ZjuMI2025
from moabb.datasets.preprocessing import SetRawAnnotations


@pytest.mark.parametrize(
    "cls,count", [(Jia2019, 2), (Ortiz2023, 1), (Yilmaz2024, 4), (ZjuMI2025, 4)]
)
def test_invalid_subject_and_download_flags(cls, count, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Invalid subject"):
        cls().data_path(999)
    mod = sys.modules[cls.__module__]
    calls = []

    def download(url, sign, path=None, force_update=False, verbose=None):
        calls.append((path, force_update, verbose))
        return str(tmp_path / "archive")

    monkeypatch.setattr(mod.dl, "data_dl", download)
    monkeypatch.setattr(mod.dl, "fs_get_file_list", lambda _: [])
    monkeypatch.setattr(
        mod.dl,
        "fs_get_file_id",
        lambda _: {"exp1-S1-left.mat": "1", "exp1-S1-right.mat": "2"},
    )
    if cls is Ortiz2023:
        monkeypatch.setattr(mod.z, "ZipFile", MagicMock())
    cls().data_path(
        1, path=str(tmp_path), force_update=True, update_path=False, verbose=False
    )
    assert calls == [(str(tmp_path), True, False)] * count


def _check_epochs(dataset, raw, expected, amplitude):
    boundaries = sum(raw.annotations.description == "EDGE boundary")
    raw = SetRawAnnotations(dataset.event_id, dataset.interval).transform(raw)
    assert sum(raw.annotations.description == "EDGE boundary") == boundaries
    events, _ = mne.events_from_annotations(raw, dataset.event_id, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        dataset.event_id,
        tmin=0,
        tmax=dataset.interval[1],
        baseline=None,
        preload=True,
        picks="eeg",
        on_missing="ignore",
        verbose=False,
    )
    assert epochs.events[:, 2].tolist() == expected
    np.testing.assert_allclose(epochs.get_data(), amplitude)


def test_jia_first_last_and_units():
    trial = np.full((63, 3500), 7.0)
    raw = Jia2019._build_raw([trial], [trial])
    _check_epochs(Jia2019(), raw, [1, 2], 7e-6)
    with pytest.raises(ValueError, match="analysis interval"):
        Jia2019._build_raw([trial[:, :200]], [trial])


def test_zju_first_last_and_units(tmp_path):
    path = tmp_path / "run.mat"
    savemat(path, {"EEG_data": np.full((62, 1280, 2), 8.0), "labels": [1, 4]})
    _check_epochs(ZjuMI2025(), ZjuMI2025._load_raw(path), [1, 4], 8e-6)
    savemat(path, {"EEG_data": np.ones((62, 1280, 2)), "labels": [1]})
    with pytest.raises(ValueError, match="label"):
        ZjuMI2025._load_raw(path)


def test_yilmaz_first_last_and_units(tmp_path):
    data, labels = tmp_path / "data.set", tmp_path / "labels.mat"
    with h5py.File(data, "w") as f:
        f["data"] = np.full((2, 448, 13), 9.0)
    savemat(labels, {"labels": [1, 2]})
    _check_epochs(Yilmaz2024(), Yilmaz2024._reconstruct_raw(data, labels), [1, 2], 9e-6)
    savemat(labels, {"labels": [1]})
    with pytest.raises(ValueError, match="label"):
        Yilmaz2024._reconstruct_raw(data, labels)


def test_ortiz_codes_units(monkeypatch):
    task = np.repeat([402, 404, 406, 402], 2000)
    mat = SimpleNamespace(data_EEG=np.full((31, len(task)), 6.0), task_EEG=task)
    monkeypatch.setattr(
        "moabb.datasets.ortiz2023.loadmat", lambda *a, **k: {"session": mat}
    )
    raw = Ortiz2023()._make_raw("synthetic.mat")
    labels = "relax motor_imagery regressive_count relax".split()
    assert list(raw.annotations.description) == labels
    np.testing.assert_allclose(raw.get_data(), 6e-6)
    assert raw.get_channel_types().count("eog") == 4


def test_ortiz_missing_duplicate_sessions(monkeypatch, tmp_path):
    ds = Ortiz2023()
    monkeypatch.setattr(ds, "data_path", lambda _: tmp_path)
    with pytest.raises(ValueError, match="No EXPERIENCE"):
        ds._get_single_subject_data(1)
    folder = tmp_path / "EXPERIENCE"
    folder.mkdir()
    (folder / "M05_20210928_openloop_03.mat").touch()
    with pytest.raises(ValueError, match="3..18"):
        ds._get_single_subject_data(1)
    duplicate = folder / "duplicate"
    duplicate.mkdir()
    (duplicate / "M05_20210928_openloop_03.mat").touch()
    with pytest.raises(ValueError, match="Duplicate"):
        ds._get_single_subject_data(1)


@pytest.mark.parametrize("cls,count", [(Yilmaz2024, 4), (ZjuMI2025, 4)])
def test_session_mapping(cls, count, monkeypatch):
    ds = cls()
    monkeypatch.setattr(ds, "data_path", lambda _: list(range(count)))
    if cls is Yilmaz2024:
        monkeypatch.setattr(ds, "_reconstruct_raw", lambda a, b: (a, b))
        assert ds._get_single_subject_data(1) == {"0": {"0": (0, 1)}, "1": {"0": (2, 3)}}
    else:
        monkeypatch.setattr(ds, "_load_raw", lambda p: p)
        assert ds._get_single_subject_data(1) == {
            "0": {"0calibration": 0, "1feedback": 1},
            "1": {"0calibration": 2, "1feedback": 3},
        }
