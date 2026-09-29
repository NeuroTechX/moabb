"""Offline reader contracts; no released recordings are downloaded."""

import sys
import types
import zipfile
from unittest.mock import Mock

import numpy as np
import pytest

from moabb.datasets import Shin2022, neuroTUMBCI, neurotumbci, shin2022


def test_shin_feedback_edges_and_units(monkeypatch):
    reader = types.SimpleNamespace(
        samplingrate=10,
        signals=np.full((16, 12), 25.0),
        states={
            "Feedback": np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]),
            "TargetCode": np.array([1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 1, 1]),
        },
    )
    module = types.ModuleType("BCI2kReader.BCI2kReader")
    module.BCI2kReader = Mock(return_value=reader)
    monkeypatch.setitem(sys.modules, "BCI2kReader", types.ModuleType("BCI2kReader"))
    monkeypatch.setitem(sys.modules, "BCI2kReader.BCI2kReader", module)
    raw = Shin2022()._read_bci2000_dat("synthetic.dat")
    module.BCI2kReader.assert_called_once_with("synthetic.dat")
    np.testing.assert_allclose(raw.get_data(), 25e-6)
    np.testing.assert_allclose(raw.annotations.onset, [0, 0.4, 0.9])
    np.testing.assert_allclose(raw.annotations.duration, [0.2, 0.2, 0.3])
    assert raw.annotations.description.tolist() == [
        "right_hand",
        "left_hand",
        "right_hand",
    ]


def test_shin_download_flags_and_live_only(monkeypatch, tmp_path):
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("S01_test/S01_test_LR_S1001/BW/BW30.dat", b"test")
        zf.writestr("SIMULATED_S01_test/fake.csv", b"test")
    data_dl = Mock(return_value=str(archive))
    monkeypatch.setattr(shin2022.dl, "data_dl", data_dl)
    result = Shin2022().data_path(1, path=str(tmp_path), force_update=True, verbose=False)
    assert data_dl.call_args_list[0].args[2:] == (str(tmp_path), True, False)
    assert "S01_test_LR_S1001" in result
    assert not (tmp_path / "SIMULATED_S01_test").exists()


@pytest.mark.parametrize("subject,count", [(1, 6), (2, 4)])
def test_neurotum_download_flags(monkeypatch, tmp_path, subject, count):
    data_dl = Mock(return_value="fetched")
    monkeypatch.setattr(neurotumbci.dl, "data_dl", data_dl)
    paths = neuroTUMBCI().data_path(
        subject, path=str(tmp_path), force_update=True, verbose=False
    )
    assert len(paths) == count == data_dl.call_count
    assert all(
        call.kwargs == {"path": str(tmp_path), "force_update": True, "verbose": False}
        for call in data_dl.call_args_list
    )


def test_neurotum_mapping_timing_and_units(monkeypatch, tmp_path):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("mapping:\n  L: LEFT HAND MI\n  R: REST\n")
    dataset = neuroTUMBCI()
    monkeypatch.setattr(
        dataset, "data_path", Mock(return_value=["session.xdf", str(mapping)])
    )
    eeg = {
        "time_series": np.full((1000, 24), 20.0),
        "time_stamps": 100 + np.arange(1000) / 250,
    }
    markers = {"time_stamps": [100, 101, 102], "time_series": [["L"], ["ignore"], ["R"]]}
    monkeypatch.setattr(dataset, "_load_xdf", Mock(return_value=(eeg, markers)))
    raw = dataset._get_single_subject_data(1)["0"]["0"]
    dataset._load_xdf.assert_called_once_with("session.xdf")
    np.testing.assert_allclose(raw.get_data(), 20e-6)
    np.testing.assert_allclose(raw.annotations.onset, [0, 2])
    assert raw.annotations.description.tolist() == ["left_hand", "rest"]


def test_neurotum_missing_stream_fails(monkeypatch):
    monkeypatch.setattr(neurotumbci, "read_xdf", Mock(return_value=([], {})))
    with pytest.raises(RuntimeError, match="EEG or Marker"):
        neuroTUMBCI._load_xdf("synthetic.xdf")


@pytest.mark.parametrize("dataset", [Shin2022, neuroTUMBCI])
def test_invalid_subject(dataset):
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset().data_path(999)
