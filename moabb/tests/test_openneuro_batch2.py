"""Synthetic, no-download regressions for the second OpenNeuro batch.

The shared ``OpenNeuroMirrorMixin`` contract (SDK raw selection with a mocked
HTTP transport, no sourcedata prefetch) is owned by ``test_openneuro_mirror.py``
in PR #1186. ``test_provider_policy`` is a single-case copy kept only so this
branch covers its own copy of ``_openneuro_mirror.py``; delete it when rebasing
on #1186.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mne
import numpy as np
import pandas as pd
import pytest

from moabb.datasets import Daly2020, Damm2026, Peterson2022
from moabb.datasets.base import BaseBIDSDataset
from moabb.datasets.download import NemarDownloadError


CASES = [
    (Daly2020, "on002720", 1, "01"),
    (Damm2026, "on008446", 1, "01"),
    (Peterson2022, "on003810", 2, "02"),
]


@pytest.mark.parametrize("cls,nemar_id,subject,label", CASES)
def test_mirror_flags(cls, nemar_id, subject, label, monkeypatch, tmp_path):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    transport = Mock()
    monkeypatch.setattr("moabb.datasets.download.nemar.download", transport)
    ds = cls()
    assert ds.nemar_id == nemar_id
    ds.download([subject], tmp_path, True, False, verbose="ERROR")
    transport.assert_called_once_with(
        dataset=nemar_id,
        target_dir=tmp_path / f"MNE-{ds.code.lower()}-data" / nemar_id,
        subject=label,
        trust_existing=False,
        scope="raw",
        datatype="eeg",
    )
    # Raw mirror loaders must never prefetch converted sourcedata.
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    ds._prefetch_nemar_sourcedata([subject])


def test_provider_policy(monkeypatch, tmp_path):
    cls, _, subject, _ = CASES[0]
    ds = cls()
    transport = Mock(side_effect=NemarDownloadError("offline"))
    monkeypatch.setattr(ds, "_download_nemar", transport)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    with pytest.raises(NemarDownloadError):
        ds._mirror_root(subject, tmp_path, False, False, None)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    with pytest.warns(RuntimeWarning, match="OpenNeuro"):
        assert ds._mirror_root(subject, tmp_path, False, False, None) is None
    transport.reset_mock()
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    assert ds._mirror_root(subject, tmp_path, False, False, None) is None
    transport.assert_not_called()


@pytest.mark.parametrize(
    "module,cls,subject,n_files",
    [("peterson2022", Peterson2022, 2, 12), ("daly2020", Daly2020, 1, 45)],
)
def test_bids_transport_only_mock(module, cls, subject, n_files, monkeypatch, tmp_path):
    """Exercise the real download manifest and root creation, mocking transport only."""
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    transport = Mock()
    monkeypatch.setattr(f"moabb.datasets.{module}.data_dl", transport)
    root = cls()._download_subject(subject, tmp_path, True, False, "ERROR")
    assert (Path(root) / "dataset_description.json").exists()
    assert transport.call_count == n_files
    for call in transport.call_args_list:
        assert call.kwargs["path"] == tmp_path
        assert call.kwargs["force_update"] is True
        assert call.kwargs["verbose"] == "ERROR"
        assert call.kwargs["fname"].startswith(f"sub-{subject:02d}/eeg/")


def test_damm_download_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    transport = Mock(return_value="dummy.edf")
    monkeypatch.setattr("moabb.datasets.damm2026.dl.data_dl", transport)
    assert len(Damm2026().data_path(1, tmp_path, True, False, "ERROR")) == 4
    assert transport.call_count == 8
    for call in transport.call_args_list:
        assert call.kwargs == {"path": tmp_path, "force_update": True, "verbose": "ERROR"}


def _raw(descriptions):
    """Raw with a bad EEG channel, a native stim channel and one marker per 6 s."""
    raw = mne.io.RawArray(
        np.vstack([np.full(1001, 2e-6), np.full(1001, 7)]),
        mne.create_info(["C3", "native"], 100, ["eeg", "stim"]),
        verbose=False,
    )
    raw.info["bads"] = ["C3"]
    onsets = 6.0 * np.arange(len(descriptions))
    raw.set_annotations(mne.Annotations(onsets, np.zeros_like(onsets), descriptions))
    return raw


def _daly_paths(*run_numbers):
    return [SimpleNamespace(task=f"run{n}") for n in run_numbers]


def _assert_loaded(raw, expected, event_id=None, tmax=None):
    """Native stim replaced by STIM, bads and SI volts kept, boundary trials epoch."""
    assert raw.ch_names == ["C3", "STIM"]
    assert raw.info["bads"] == ["C3"]
    np.testing.assert_allclose(raw.get_data(picks=["C3"]), 2e-6)
    events = mne.find_events(raw, initial_event=True, shortest_event=1)
    np.testing.assert_array_equal(events[:, [0, 2]], expected)
    if tmax is not None:
        kwargs = {"baseline": None, "picks": ["C3"], "on_missing": "ignore"}
        epochs = mne.Epochs(raw, events, event_id, 0, tmax, preload=True, **kwargs)
        assert len(epochs) == len(expected)


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
    _assert_loaded(result, [[0, 1], [600, 2]], ds.event_id, tmax=4)
    assert ds._get_path_search_params(2)["runs"] == ["1", "2", "3", "4"]


def test_peterson_get_data_builds_pipeline_without_sourcedata(monkeypatch):
    """Peterson overrides the process pipeline (cache version); get_data must
    still build it through the raw mirror path without touching sourcedata."""
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    ds = Peterson2022()
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    monkeypatch.setattr(ds, "_get_selected_subject_data", lambda *args: {})
    assert ds.get_data([2]) == {2: {}}


def test_daly_native_stim_units_and_missing_runs(monkeypatch):
    ds = Daly2020()
    monkeypatch.setattr(ds, "bids_paths", lambda subject: _daly_paths(9))
    monkeypatch.setattr(ds, "_read_raw_bids", lambda path: _raw(["1", "2"]))
    _assert_loaded(ds._get_single_subject_data(1)["0"]["0"], [[0, 1], [600, 2]])


def test_daly_duplicate_runs_rejected(monkeypatch):
    ds = Daly2020()
    monkeypatch.setattr(ds, "bids_paths", lambda subject: _daly_paths(2, 2))
    with pytest.raises(ValueError, match="duplicate"):
        ds._get_single_subject_data(1)


def test_daly_skips_only_targetless_noncalibration_runs(monkeypatch):
    """A released empty events.tsv is skipped and valid runs stay re-indexed."""
    ds = Daly2020()
    raws = {"run2": _raw([]), "run3": _raw(["1", "2"]), "run4": _raw(["2"])}
    monkeypatch.setattr(ds, "bids_paths", lambda subject: _daly_paths(1, 2, 3, 4))
    monkeypatch.setattr(ds, "_read_raw_bids", lambda path: raws[path.task])
    runs = ds._get_single_subject_data(5)["0"]
    assert list(runs) == ["0", "1"]
    assert list(runs["0"].annotations.description) == ["right_hand", "relax"]
    assert list(runs["1"].annotations.description) == ["relax"]


def test_daly_raises_when_no_noncalibration_run_has_targets(monkeypatch):
    ds = Daly2020()
    monkeypatch.setattr(ds, "bids_paths", lambda subject: _daly_paths(1, 2, 3))
    # Also cover a release file without a native stim channel.
    monkeypatch.setattr(
        ds, "_read_raw_bids", lambda path: _raw([]).drop_channels(["native"])
    )
    with pytest.raises(ValueError, match="subject 5 has no usable motor-imagery runs"):
        ds._get_single_subject_data(5)


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
        _assert_loaded(raw, [[0, 3], [400, 7]], ds.event_id, tmax=6)
