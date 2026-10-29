"""Offline regressions for the BrainVision/GDF/EEGLAB archive loaders."""

import zipfile
from pathlib import Path
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from moabb.datasets import (
    Batista2022,
    Farabbi2020,
    Han2026,
    Kodera2023,
    Kueper2024,
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
    [(Batista2022, "sub-01"), (Farabbi2020, "01"), (Kueper2024, "EEG_dataset")],
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


def test_kueper2024_exact_markers_and_run_classes(monkeypatch):
    ds = Kueper2024()
    monkeypatch.setattr(
        ds,
        "data_path",
        Mock(return_value=["unilateral/test.vhdr", "bilateral/test.vhdr"]),
    )
    # Each onset is a correct trial (S128 follows it); S1000 must not match S100.
    markers = [
        "Stimulus/S 100",
        "Stimulus/S128",
        "Stimulus/S 1000",
        "Stimulus/S 101",
        "Stimulus/S128",
    ]
    monkeypatch.setattr(
        mne.io,
        "read_raw_brainvision",
        Mock(side_effect=[raw(["C3"], markers, [0, 1, 3, 6, 7]) for _ in range(2)]),
    )
    runs = ds._get_single_subject_data(1)["0"]
    for name, onset in [("0unilateral", 0), ("1bilateral", 6)]:
        assert runs[name].annotations.onset.tolist() == [onset]
        np.testing.assert_allclose(runs[name].get_data(), 2e-6)
        assert runs[name].info["bads"] == ["C3"]


def _kueper_annotations(events):
    onsets, descriptions = zip(*events)
    return mne.Annotations(list(onsets), 0.0, [f"Stimulus/{d}" for d in descriptions])


def test_kueper2024_keeps_correct_trials_and_places_rest():
    from moabb.datasets.kueper2024 import _trial_events

    annotations = _kueper_annotations(
        [
            (2.0, "S 23"),  # hands on the switches
            (10.0, "S100"),  # correct trial: 8 s of rest before it
            (10.05, "S 22"),
            (11.0, "S 21"),
            (12.0, "S 23"),
            (12.5, "S128"),
            (15.0, "S100"),  # error trial: only 3 s of rest, no S128
            (15.5, "S  3"),
            (16.0, "S 23"),
            (20.6, "S100"),  # correct, but 4.6 s of rest: no rest event
            (22.0, "S 23"),
            (22.5, "S128"),
            (30.0, "S100"),  # additional trial (S27), no S128
            (30.2, "S 27"),
        ]
    )
    movement, rest = _trial_events(annotations, "unilateral")
    assert movement == [10.0, 20.6]
    assert rest == [7.0]


def test_kueper2024_second_study_uses_hand_switch_release():
    from moabb.datasets.kueper2024 import _trial_events

    # Second-study recordings have no motion-tracking marker (S100).
    annotations = _kueper_annotations(
        [(1.0, "S 23"), (8.0, "S 22"), (9.0, "S 23"), (9.5, "S128"), (13.0, "S 22")]
    )
    movement, rest = _trial_events(annotations, "unilateral")
    assert movement == [8.0]  # the last release has no S128
    assert rest == [5.0]


def test_kueper2024_downloads_subjects_missing_from_a_partial_cache(
    tmp_path, monkeypatch
):
    # A version-1 or NEMAR-seeded cache has subjects 1-8 only.
    data_dir = tmp_path / "MNE-kueper2024-data"
    av82 = data_dir / "EEG_dataset" / "EEG" / "unilateral" / "AV82"
    av82.mkdir(parents=True)
    (av82 / "set1.vhdr").touch()
    extract = Mock()
    monkeypatch.setattr(
        "moabb.datasets.kueper2024.download_and_extract_subject_zip", extract
    )
    ds = Kueper2024()
    assert ds.data_path(1, path=tmp_path) == [str(av82 / "set1.vhdr")]
    extract.assert_not_called()
    ds.data_path(9, path=tmp_path)  # AF64D, second study
    extract.assert_called_once()
    assert extract.call_args.kwargs["skip_existing"] is True


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
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    data_dl = Mock(return_value=str(tmp_path / "synthetic.set"))
    monkeypatch.setattr(download, "data_dl", data_dl)
    ds = Han2026()
    ds.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert data_dl.call_args.args[2:] == (tmp_path, True, "ERROR")
    assert "ds007327/sub-001/sub-001_task-dribble_eeg.set" in data_dl.call_args.args[0]
    assert ds.nemar_id == "on007327"
    _assert_invalid_subject_rejected(ds)


def test_han_reads_the_recording_from_the_nemar_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    data_dl = Mock(side_effect=AssertionError("must not reach OpenNeuro"))
    monkeypatch.setattr(download, "data_dl", data_dl)
    ds = Han2026()
    monkeypatch.setattr(ds, "_download_nemar", Mock(return_value=str(tmp_path)))
    paths = ds.data_path(3, path=tmp_path, verbose="ERROR")
    assert paths == [str(tmp_path / "sub-003" / "eeg" / "sub-003_task-dribble_eeg.set")]
    ds._download_nemar.assert_called_once_with(3, tmp_path, False, None, "ERROR")


def test_han_falls_back_to_openneuro_when_the_mirror_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    data_dl = Mock(return_value=str(tmp_path / "synthetic.set"))
    monkeypatch.setattr(download, "data_dl", data_dl)
    ds = Han2026()
    monkeypatch.setattr(
        ds, "_download_nemar", Mock(side_effect=download.NemarDownloadError("offline"))
    )
    with pytest.warns(RuntimeWarning, match="on007327"):
        ds.data_path(1, path=tmp_path, verbose="ERROR")
    assert "ds007327/sub-001/sub-001_task-dribble_eeg.set" in data_dl.call_args.args[0]


def test_kodera_shared_channels_preserve_units_and_bads(monkeypatch):
    channels = list(kodera2023._COMMON_EEG_CHANNELS)
    # The loader only accepts the archive's rates; 500 Hz is kept as is.
    run = mne.io.RawArray(
        np.full((len(channels) + 1, 5001), 2e-6),
        mne.create_info(channels + ["Fp1"], 500, "eeg"),
        verbose=False,
    )
    run.info["bads"] = [channels[0]]
    run.set_annotations(mne.Annotations([0, 6], 0, ["Stimulus/S  1"] * 2))
    monkeypatch.setattr(mne.io, "read_raw_brainvision", Mock(return_value=run))
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
