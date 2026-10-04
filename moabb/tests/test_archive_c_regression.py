"""Offline signal and transport contracts for the archive-C datasets."""

import zipfile
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest
from scipy.io import savemat

from moabb.datasets import Leelakittisin2025, PerezBlanco2026, Vagaja2023
from moabb.datasets.leelakittisin2025 import EEG_CHANNELS
from moabb.datasets.perezblanco2026 import _EEG_NAMES, _EMG_NAMES
from moabb.datasets.vagaja2023 import _EEG_CHANNELS


def test_perez_execution_first_last_events_and_si_units():
    names = _EEG_NAMES + _EMG_NAMES + ["CurrentTarget", "TrialInitMovStam"]
    data = np.full((18, 4000), 25e-6)
    data[-2:] = 0
    for sample, code in [(0, 1), (500, 2), (1000, 3), (2000, 4)]:
        data[-2, sample] = code
        data[-1, sample] = 1
    raw = mne.io.RawArray(data, mne.create_info(names, 512, "eeg"), verbose=False)
    result = PerezBlanco2026(return_all_modalities=True)._build_run(raw)
    events = mne.find_events(result, initial_event=True, shortest_event=1)
    np.testing.assert_array_equal(events[:, 0], [0, 500, 1000, 2000])
    np.testing.assert_array_equal(events[:, 2], [1, 2, 3, 4])
    np.testing.assert_allclose(result.get_data(picks="eeg"), 25e-6)
    assert result.get_channel_types().count("emg") == 8


def test_leelakittisin2025_physiological_units_trigger_and_trial_boundaries(tmp_path):
    data = np.full((63, 12001), 25.0)
    data[-1] = 0
    data[-1, 0] = 21
    data[-1, 7200] = 32
    path = tmp_path / "S01_S1.mat"
    savemat(path, {"eeg": data, "eeg_fs": 1200})
    dataset = Leelakittisin2025()
    raw = dataset._build_raw(path)
    np.testing.assert_allclose(raw.get_data(picks="eeg"), 25e-6)
    np.testing.assert_allclose(raw.get_data(picks="eog"), 25e-6)
    events = mne.find_events(raw, initial_event=True, shortest_event=1)
    np.testing.assert_array_equal(events[:, 2], [21, 32])
    epochs = mne.Epochs(
        raw,
        events,
        dataset.event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert len(epochs) == 2
    assert raw.ch_names[:60] == EEG_CHANNELS


def test_vagaja_annotation_mapping_units_and_bad_channel_preserved(monkeypatch):
    raw = mne.io.RawArray(
        np.full((32, 12001), 12e-6),
        mne.create_info(_EEG_CHANNELS, 500, "eeg"),
        verbose=False,
    )
    raw.info["bads"] = ["C3"]
    raw.set_annotations(
        mne.Annotations([0, 14], [0, 0], ["Stimulus/S 7", "Stimulus/S 8"])
    )
    reader = Mock(return_value=raw)
    monkeypatch.setattr(mne.io, "read_raw_brainvision", reader)
    dataset = Vagaja2023()
    result = dataset._load_run(
        "unused.vhdr", mne.channels.make_standard_montage("colin27_1020")
    )
    assert reader.call_args.kwargs["overrides"] == {
        "data_fname": "unused.eeg",
        "marker_fname": "unused.vmrk",
    }
    assert list(result.annotations.description) == ["left_hand", "right_hand"]
    assert result.info["bads"] == ["C3"]
    np.testing.assert_allclose(result.get_data(), 12e-6)
    events, _ = mne.events_from_annotations(result, event_id=dataset.event_id)
    epochs = mne.Epochs(
        result,
        events,
        dataset.event_id,
        tmin=0,
        tmax=10,
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert len(epochs) == 2


@pytest.mark.parametrize(
    "dataset,subject",
    [(PerezBlanco2026(), 0), (Leelakittisin2025(), 5), (Vagaja2023(), 1)],
)
def test_invalid_subject_rejected_without_transport(dataset, subject):
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(subject)


def test_leelakittisin2025_transport_flags_missing_and_duplicate_sessions(tmp_path):
    archive = tmp_path / "subject.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/S01_S1.mat", "first")
    with patch(
        "moabb.datasets.leelakittisin2025.dl.data_dl", return_value=str(archive)
    ) as download:
        with pytest.raises(FileNotFoundError):
            Leelakittisin2025().data_path(
                1, path=tmp_path, force_update=True, verbose="ERROR"
            )
        assert download.call_args.kwargs == {
            "path": tmp_path,
            "force_update": True,
            "verbose": "ERROR",
        }
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ["a/S01_S1.mat", "b/S01_S1.mat", "a/S01_S2.mat"]:
            zf.writestr(name, "duplicate")
    with patch("moabb.datasets.leelakittisin2025.dl.data_dl", return_value=str(archive)):
        with pytest.raises(ValueError, match="Duplicate session"):
            Leelakittisin2025().data_path(1, force_update=True)


def test_vagaja_force_update_reextracts_archive(tmp_path):
    archive = tmp_path / "GROUPS.zip"
    target = tmp_path / "MNE-vagaja2023-data/Embodied/SUB03/SUB03_MI.vhdr"
    target.parent.mkdir(parents=True)
    target.write_text("old")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Embodied/SUB03/SUB03_MI.vhdr", "new")
    with patch("moabb.datasets.download.data_dl", return_value=str(archive)) as download:
        Vagaja2023().data_path(3, path=tmp_path, force_update=True, verbose="ERROR")
        assert download.call_args.args[2:] == (tmp_path, True, "ERROR")
    assert target.read_text() == "new"


def test_perez_figshare_transport_flags(tmp_path):
    archive = tmp_path / "sub-01.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sub-01/eeg/sub-01_task-wrist_run-01_eeg.edf", "synthetic")
    with (
        patch("moabb.datasets.perezblanco2026.dl.fs_get_file_list", return_value=[]),
        patch(
            "moabb.datasets.perezblanco2026.dl.fs_get_file_id",
            return_value={"sub-01.zip": "123"},
        ),
        patch(
            "moabb.datasets.perezblanco2026.dl.data_dl", return_value=str(archive)
        ) as download,
    ):
        paths = PerezBlanco2026().data_path(
            1, path=tmp_path, force_update=True, verbose="ERROR"
        )
    assert len(paths) == 1
    assert download.call_args.args[2:] == (tmp_path, True, "ERROR")


def test_leelakittisin2025_nested_archive_is_extracted_only_once(tmp_path):
    """The real Zenodo layout (extra ``v1_raw_s<ID>/``) must be reusable offline."""
    archive = tmp_path / "v1_raw_S01.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for session in (1, 2):
            zf.writestr(f"v1_raw_s1/S01_S{session}.mat", b"placeholder")
    with (
        patch.object(
            zipfile.ZipFile, "extractall", autospec=True, wraps=zipfile.ZipFile.extractall
        ) as extractall,
        patch("moabb.datasets.leelakittisin2025.dl.data_dl", return_value=str(archive)),
    ):
        paths = [Leelakittisin2025().data_path(1) for _ in range(2)]
    expected = [str(tmp_path / f"S01/v1_raw_s1/S01_S{s}.mat") for s in (1, 2)]
    assert paths == [expected, expected]
    assert extractall.call_count == 1


def test_vagaja_missing_subject_or_run_fails_explicitly(tmp_path, monkeypatch):
    archive = tmp_path / "GROUPS.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Embodied/SUB03/notes.txt", "no MI run")
    ds = Vagaja2023()
    monkeypatch.setenv("MNE_DATASETS_VAGAJA2023_PATH", str(tmp_path))
    with patch("moabb.datasets.download.data_dl", return_value=str(archive)):
        with pytest.raises(FileNotFoundError, match="Missing subject directory"):
            ds.data_path(5)
        with pytest.raises(FileNotFoundError, match="No motor-imagery BrainVision"):
            ds._get_single_subject_data(3)
    subject_dir = tmp_path / "MNE-vagaja2023-data/Embodied/SUB03"
    (subject_dir / "SUB03_MI.vhdr").touch()
    monkeypatch.setattr(ds, "data_path", Mock(return_value=[subject_dir]))
    load_run = Mock(return_value="run")
    monkeypatch.setattr(ds, "_load_run", load_run)
    assert ds._get_single_subject_data(3) == {"0": {"0": "run"}}
    assert load_run.call_args.args[0] == subject_dir / "SUB03_MI.vhdr"
