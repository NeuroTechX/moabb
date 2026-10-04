"""Offline synthetic regression tests for the MATLAB MI loader batch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import h5py
import mne
import numpy as np
import pytest
from scipy.io import savemat

from moabb.datasets import MIBMPI2024, Jia2019, Ortiz2023, Wang2025
from moabb.datasets import download as dl
from moabb.datasets.preprocessing import SetRawAnnotations


@pytest.mark.parametrize(
    "cls,count", [(Jia2019, 2), (Ortiz2023, 1), (MIBMPI2024, 4), (Wang2025, 4)]
)
def test_invalid_subject_and_download_flags(cls, count, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Invalid subject"):
        cls().data_path(999)
    data_dl = Mock(return_value=str(tmp_path / "archive"))
    monkeypatch.setattr(dl, "data_dl", data_dl)
    monkeypatch.setattr(dl, "fs_get_file_list", Mock(return_value=[]))
    monkeypatch.setattr(
        dl,
        "fs_get_file_id",
        Mock(return_value={"exp1-S1-left.mat": "1", "exp1-S1-right.mat": "2"}),
    )
    if cls is Ortiz2023:
        (tmp_path / "archive").touch()
        monkeypatch.setattr("zipfile.ZipFile", MagicMock())
    cls().data_path(
        1, path=str(tmp_path), force_update=True, update_path=False, verbose=False
    )
    calls = [c.args[2:] for c in data_dl.call_args_list]
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


def test_wang2025_first_last_and_units(tmp_path):
    path = tmp_path / "run.mat"
    savemat(path, {"EEG_data": np.full((62, 1280, 2), 8.0), "labels": [1, 4]})
    _check_epochs(Wang2025(), Wang2025._load_raw(path), [1, 4], 8e-6)
    savemat(path, {"EEG_data": np.ones((62, 1280, 2)), "labels": [1]})
    with pytest.raises(ValueError, match="label"):
        Wang2025._load_raw(path)


def test_mibmpi_first_last_and_units(tmp_path):
    data, labels = tmp_path / "data.set", tmp_path / "labels.mat"
    with h5py.File(data, "w") as f:
        f["data"] = np.full((2, 448, 13), 9.0)
    savemat(labels, {"labels": [1, 2]})
    _check_epochs(MIBMPI2024(), MIBMPI2024._reconstruct_raw(data, labels), [1, 2], 9e-6)
    savemat(labels, {"labels": [1]})
    with pytest.raises(ValueError, match="label"):
        MIBMPI2024._reconstruct_raw(data, labels)


def test_ortiz_codes_units(monkeypatch):
    task = np.repeat([402, 404, 406, 402], 2000)
    mat = SimpleNamespace(data_EEG=np.full((31, len(task)), 6.0), task_EEG=task)
    monkeypatch.setattr(
        "moabb.datasets.ortiz2023.loadmat", Mock(return_value={"session": mat})
    )
    raw = Ortiz2023()._make_raw("synthetic.mat")
    labels = "relax motor_imagery regressive_count relax".split()
    assert list(raw.annotations.description) == labels
    np.testing.assert_allclose(raw.get_data(), 6e-6)
    assert raw.get_channel_types().count("eog") == 4


def test_ortiz_missing_duplicate_sessions(monkeypatch, tmp_path):
    ds = Ortiz2023()
    monkeypatch.setattr(ds, "data_path", Mock(return_value=tmp_path))
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


@pytest.mark.parametrize("cls,count", [(MIBMPI2024, 4), (Wang2025, 4)])
def test_session_mapping(cls, count, monkeypatch):
    ds = cls()
    monkeypatch.setattr(ds, "data_path", Mock(return_value=list(range(count))))
    if cls is MIBMPI2024:
        reader = Mock(side_effect=["s0", "s1"])
        monkeypatch.setattr(ds, "_reconstruct_raw", reader)
        assert ds._get_single_subject_data(1) == {"0": {"0": "s0"}, "1": {"0": "s1"}}
        assert reader.call_args_list == [call(0, 1), call(2, 3)]
    else:
        reader = Mock(side_effect=["r0", "r1", "r2", "r3"])
        monkeypatch.setattr(ds, "_load_raw", reader)
        assert ds._get_single_subject_data(1) == {
            "0": {"0calibration": "r0", "1feedback": "r1"},
            "1": {"0calibration": "r2", "1feedback": "r3"},
        }
        assert reader.call_args_list == [call(p) for p in range(4)]
