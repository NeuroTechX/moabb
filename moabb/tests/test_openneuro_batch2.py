"""Synthetic, no-download regressions for the second OpenNeuro batch."""

from types import SimpleNamespace
from unittest.mock import Mock

import mne
import numpy as np
import pandas as pd
import pytest

from moabb.datasets import Daly2020, Damm2026, Peterson2022
from moabb.datasets.base import BaseBIDSDataset


@pytest.mark.parametrize("cls,subject", [(Peterson2022, 2), (Daly2020, 1)])
def test_bids_flag_contract(cls, subject, monkeypatch, tmp_path):
    dataset = cls()
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(dataset, "_download_subject", download)
    monkeypatch.setattr(dataset, "_find_matching_paths", lambda **kwargs: [])
    assert dataset.data_path(subject, tmp_path, True, False, "ERROR") == []
    download.assert_called_once_with(subject, tmp_path, True, False, "ERROR")


@pytest.mark.parametrize(
    "module,cls,subject,n_files",
    [("peterson2022", Peterson2022, 2, 12), ("daly2020", Daly2020, 1, 45)],
)
def test_bids_transport_only_mock(module, cls, subject, n_files, monkeypatch, tmp_path):
    """Exercise the real download manifest and root creation, mocking transport only."""
    transport = Mock()
    monkeypatch.setattr(f"moabb.datasets.{module}.data_dl", transport)
    root = cls()._download_subject(subject, tmp_path, True, False, "ERROR")
    from pathlib import Path

    assert (Path(root) / "dataset_description.json").exists()
    assert transport.call_count == n_files
    for call in transport.call_args_list:
        assert call.kwargs["path"] == tmp_path
        assert call.kwargs["force_update"] is True
        assert call.kwargs["verbose"] == "ERROR"
        assert call.kwargs["fname"].startswith(f"sub-{subject:02d}/eeg/")


def test_damm_download_flags(monkeypatch, tmp_path):
    transport = Mock(return_value="dummy.edf")
    monkeypatch.setattr("moabb.datasets.damm2026.dl.data_dl", transport)
    assert len(Damm2026().data_path(1, tmp_path, True, False, "ERROR")) == 4
    assert transport.call_count == 8
    for call in transport.call_args_list:
        assert call.kwargs == {"path": tmp_path, "force_update": True, "verbose": "ERROR"}


def _raw(descriptions):
    raw = mne.io.RawArray(
        np.vstack([np.full(1001, 2e-6), np.full(1001, 7)]),
        mne.create_info(["C3", "native"], 100, ["eeg", "stim"]),
        verbose=False,
    )
    raw.info["bads"] = ["C3"]
    raw.set_annotations(mne.Annotations([0, 6], [0, 0], descriptions))
    return raw


def test_peterson_units_native_stim_and_boundary_trials(monkeypatch):
    raw = _raw(["OVTK_GDF_Right", "OVTK_GDF_Tongue"])
    monkeypatch.setattr(
        BaseBIDSDataset,
        "_get_single_subject_data",
        lambda self, subject: {"0": {"1": raw}},
    )
    ds = Peterson2022()
    assert ds._get_read_extra_params(2) == {"units": "uV"}
    result = ds._get_single_subject_data(2)["0"]["1"]
    assert result.ch_names == ["C3", "STIM"]
    assert result.info["bads"] == ["C3"]
    np.testing.assert_allclose(result.get_data(picks=["C3"]), 2e-6)
    events = mne.find_events(result, initial_event=True, shortest_event=1)
    np.testing.assert_array_equal(events[:, [0, 2]], [[0, 1], [600, 2]])
    epochs = mne.Epochs(
        result,
        events,
        ds.event_id,
        0,
        4,
        baseline=None,
        picks=["C3"],
        preload=True,
        verbose=False,
    )
    assert len(epochs) == 2
    assert ds._get_path_search_params(2)["runs"] == ["1", "2", "3", "4"]


def test_daly_native_stim_units_and_missing_runs(monkeypatch):
    ds = Daly2020()
    monkeypatch.setattr(ds, "bids_paths", lambda subject: [SimpleNamespace(task="run9")])
    monkeypatch.setattr(ds, "_read_raw_bids", lambda path: _raw(["1", "2"]))
    raw = ds._get_single_subject_data(1)["0"]["0"]
    assert raw.ch_names == ["C3", "STIM"]
    assert raw.info["bads"] == ["C3"]
    np.testing.assert_allclose(raw.get_data(picks=["C3"]), 2e-6)
    np.testing.assert_array_equal(
        mne.find_events(raw, initial_event=True, shortest_event=1)[:, 2], [1, 2]
    )


def test_daly_duplicate_runs_rejected(monkeypatch):
    ds = Daly2020()
    monkeypatch.setattr(
        ds, "bids_paths", lambda subject: [SimpleNamespace(task="run2")] * 2
    )
    with pytest.raises(ValueError, match="duplicate"):
        ds._get_single_subject_data(1)


def test_damm_block_onsets_first_last_and_empty(tmp_path):
    file = tmp_path / "events.tsv"
    pd.DataFrame(
        {"onset": np.arange(6) / 100**2, "trial_type": [3, 3, 1, 7, 7, 7]}
    ).to_csv(file, sep="\t", index=False)
    ds = Damm2026()
    np.testing.assert_array_equal(ds._read_events(file, 100, 6), [[0, 0, 3], [3, 0, 7]])
    pd.DataFrame(columns=["onset", "trial_type"]).to_csv(file, sep="\t", index=False)
    assert ds._read_events(file, 100, 6).shape == (0, 3)


def test_damm_loader_native_stim_and_si_units(monkeypatch):
    ds = Damm2026()
    monkeypatch.setattr(ds, "data_path", lambda subject: ["test_eeg.edf"] * 4)
    monkeypatch.setattr(
        "moabb.datasets.damm2026.mne.io.read_raw_edf",
        lambda *args, **kwargs: _raw(["ignored", "ignored"]),
    )
    monkeypatch.setattr(
        ds, "_read_events", lambda *args: np.array([[0, 0, 3], [400, 0, 7]])
    )
    runs = ds._get_single_subject_data(1)["0"]
    assert len(runs) == 4
    for raw in runs.values():
        assert raw.ch_names == ["C3", "STIM"]
        assert raw.info["bads"] == ["C3"]
        np.testing.assert_allclose(raw.get_data(picks=["C3"]), 2e-6)
        events = mne.find_events(raw, initial_event=True, shortest_event=1)
        np.testing.assert_array_equal(events[:, [0, 2]], [[0, 3], [400, 7]])
        epochs = mne.Epochs(
            raw,
            events,
            ds.event_id,
            0,
            6,
            baseline=None,
            picks=["C3"],
            preload=True,
            on_missing="ignore",
            verbose=False,
        )
        assert len(epochs) == 2
