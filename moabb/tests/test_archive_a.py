"""Offline regressions for the BrainVision/GDF/EEGLAB archive loaders."""

import zipfile
from pathlib import Path
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from moabb.datasets import (
    DFKI2023,
    Batista2022,
    Farabbi2020,
    Han2026,
    Kodera2023,
    download,
    farabbi2020,
    kodera2023,
)


def raw(channels, descriptions, onsets=None):
    data = mne.io.RawArray(
        np.full((len(channels), 1001), 2e-6),
        mne.create_info(channels, 100, "eeg"),
        verbose=False,
    )
    data.info["bads"] = [channels[0]]
    data.set_annotations(
        mne.Annotations(onsets if onsets is not None else [0, 6], 0, descriptions)
    )
    return data


def _assert_invalid_subject_rejected(dataset):
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(999)


@pytest.mark.parametrize(
    "cls,folder",
    [(Batista2022, "sub-01"), (Farabbi2020, "01"), (DFKI2023, "EEG_dataset")],
)
def test_download_flags_and_extraction(cls, folder, tmp_path, monkeypatch):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(folder + "/payload.txt", "synthetic")
    data_dl = Mock(return_value=str(archive))
    monkeypatch.setattr(download, "data_dl", data_dl)
    ds = cls()
    ds.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert data_dl.call_args.args[2:] == (tmp_path, True, "ERROR")
    data_dir = tmp_path / f"MNE-{ds.code.lower()}-data"
    assert (data_dir / folder / "payload.txt").read_text() == "synthetic"
    _assert_invalid_subject_rejected(ds)


def test_batista_excludes_execution(tmp_path, monkeypatch):
    session = tmp_path / "ses-lab1"
    session.mkdir()
    for task in ["grazMI", "grazME", "neurowMIMOVR"]:
        (session / f"sub-01_task-{task}.vhdr").touch()
    ds = Batista2022()
    monkeypatch.setattr(ds, "data_path", Mock(return_value=[str(tmp_path)]))
    reader = Mock(return_value="raw")
    monkeypatch.setattr(ds, "_load_run", reader)
    runs = ds._get_single_subject_data(1)["0lab1"]
    assert len(runs) == 2
    assert all(not name.endswith("ME") for name in runs)


def test_batista_annotations_units_and_bads(monkeypatch):
    data = raw(["C3", "C4", "Aux1"], ["Stimulus/S  7", "Stimulus/S  8"])
    reader = Mock(return_value=data)
    monkeypatch.setattr(mne.io, "read_raw_brainvision", reader)
    result = Batista2022()._load_run(
        Path("test.vhdr"), mne.channels.make_standard_montage("colin27_1020")
    )
    assert reader.call_args.kwargs["overrides"] == {
        "data_fname": "test.eeg",
        "marker_fname": "test.vmrk",
    }
    assert list(result.annotations.description) == ["left_hand", "right_hand"]
    assert result.ch_names == ["C3", "C4"]
    assert result.info["bads"] == ["C3"]
    np.testing.assert_allclose(result.get_data(), 2e-6)


def test_dfki_exact_markers_and_run_classes(monkeypatch):
    ds = DFKI2023()
    monkeypatch.setattr(
        ds,
        "data_path",
        Mock(return_value=["unilateral/test.vhdr", "bilateral/test.vhdr"]),
    )
    markers = ["Stimulus/S 100", "Stimulus/S 1000", "Stimulus/S 101"]
    monkeypatch.setattr(
        mne.io,
        "read_raw_brainvision",
        Mock(side_effect=[raw(["C3"], markers, [0, 3, 6]) for _ in range(2)]),
    )
    runs = ds._get_single_subject_data(1)["0"]
    for name, onset in [("0unilateral", 0), ("1bilateral", 6)]:
        assert runs[name].annotations.onset.tolist() == [onset]
        np.testing.assert_allclose(runs[name].get_data(), 2e-6)
        assert runs[name].info["bads"] == ["C3"]


def test_farabbi_gdf_event_mapping_and_modalities(monkeypatch):
    channels = farabbi2020._EEG_CHANNELS + farabbi2020._ACC_CHANNELS
    monkeypatch.setattr(
        mne.io, "read_raw_gdf", Mock(return_value=raw(channels, ["769", "770"]))
    )
    result = Farabbi2020()._load_gdf(
        "test.gdf", mne.channels.make_standard_montage("colin27_1020")
    )
    assert len(result.ch_names) == 32
    assert result.annotations.description.tolist() == ["left_hand", "right_hand"]
    np.testing.assert_allclose(result.get_data(), 2e-6)
    assert result.info["bads"] == [channels[0]]


def test_han_deduplicates_only_exact_trial_markers(monkeypatch):
    ds = Han2026()
    monkeypatch.setattr(ds, "data_path", Mock(return_value=["synthetic.set"]))
    monkeypatch.setattr(
        mne.io,
        "read_raw_eeglab",
        Mock(
            return_value=raw(["C3", "BIP1"], ["A11", "A11", "A22", "A11"], [0, 0, 6, 6])
        ),
    )
    result = ds._get_single_subject_data(1)["0"]["0"]
    assert result.annotations.description.tolist() == [
        "motor_observation",
        "motor_imagery",
        "motor_observation",
    ]
    assert result.ch_names == ["C3"]
    assert result.annotations.onset.tolist() == [0, 6, 6]
    np.testing.assert_allclose(result.get_data(), 2e-6)


def test_han_transport_flags(tmp_path, monkeypatch):
    data_dl = Mock(return_value=str(tmp_path / "synthetic.set"))
    monkeypatch.setattr(download, "data_dl", data_dl)
    ds = Han2026()
    ds.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert data_dl.call_args.args[2:] == (tmp_path, True, "ERROR")
    assert "ds007327/sub-001/sub-001_task-dribble_eeg.set" in data_dl.call_args.args[0]
    assert ds.nemar_id is None
    _assert_invalid_subject_rejected(ds)


def test_kodera_shared_channels_preserve_units_and_bads(monkeypatch):
    channels = list(kodera2023._COMMON_EEG_CHANNELS)
    monkeypatch.setattr(
        mne.io,
        "read_raw_brainvision",
        Mock(return_value=raw(channels + ["Fp1"], ["Stimulus/S  1", "Stimulus/S  1"])),
    )
    result = Kodera2023()._read_run("synthetic_lh1.vhdr")
    assert result.ch_names == channels
    assert result.info["bads"] == [channels[0]]
    assert result.annotations.description.tolist() == ["left_hand", "left_hand"]
    np.testing.assert_allclose(result.get_data(), 2e-6)


def test_missing_sessions_fail_explicitly(tmp_path, monkeypatch):
    for ds in [Batista2022(), Farabbi2020()]:
        monkeypatch.setattr(ds, "data_path", Mock(return_value=[str(tmp_path)]))
        with pytest.raises(FileNotFoundError):
            ds._get_single_subject_data(1)


def test_kodera_transport_flags(tmp_path, monkeypatch):
    archive = tmp_path / "data.zip"
    # Subject 1 is ("01_12_2020", "1z"): short-name cohort, left-hand run 1.
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("data/01_12_2020/1z01122020lh1.vhdr", "synthetic")
        z.writestr("data/01_12_2020/2z01122020lh1.vhdr", "other subject")
    data_dl = Mock(return_value=str(archive))
    monkeypatch.setattr(download, "data_dl", data_dl)
    ds = Kodera2023()
    paths = ds.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert data_dl.call_args.args[2:] == (tmp_path, True, "ERROR")
    assert [Path(p).read_text() for p in paths] == ["synthetic"]
    _assert_invalid_subject_rejected(ds)
