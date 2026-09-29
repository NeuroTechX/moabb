"""Offline transport and scientific contracts for the large-recording batch."""

import zipfile

import mne
import numpy as np
import pytest

from moabb.datasets import (
    MIND2026,
    MOVING2024,
    Garro2025,
    Thapa2025,
    garro2025,
    mind2026,
    moving2024,
    thapa2025,
)


@pytest.mark.parametrize("dataset", [Garro2025, MIND2026, MOVING2024, Thapa2025])
def test_invalid_subject_does_not_download(dataset):
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset().data_path(999)


@pytest.mark.parametrize("force", [False, True])
def test_mind_transport_flags_and_cached_subject(tmp_path, monkeypatch, force):
    root = tmp_path / "MNE-mind2026-data"
    header = root / "MIND_BIDS/sub-01/eeg/sub-01_task-mi2d_run-01_eeg.vhdr"
    header.parent.mkdir(parents=True)
    header.touch()
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(str(header.relative_to(root)), "updated")
        zf.writestr("MIND_BIDS/sub-01/nirs/ignored.txt", "not EEG")
    calls = []

    def transport(url, sign, **kwargs):
        calls.append(kwargs)
        return archive

    monkeypatch.setattr(mind2026.dl, "get_dataset_path", lambda *args: tmp_path)
    monkeypatch.setattr(mind2026.dl, "data_dl", transport)
    MIND2026().data_path(1, path=str(tmp_path), force_update=force, verbose=False)
    assert len(calls) == int(force)
    if force:
        assert calls == [{"path": str(tmp_path), "force_update": True, "verbose": False}]
        assert header.read_text() == "updated"
    assert not (root / "MIND_BIDS/sub-01/nirs").exists()


@pytest.mark.parametrize("force", [False, True])
def test_thapa_transport_and_missing_sessions(tmp_path, monkeypatch, force):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dataset_description.json", "{}")
        for subject, session in [(1, 1), (2, 2)]:
            zf.writestr(
                f"sub-{subject:02}/ses-{session:02}/eeg/"
                f"sub-{subject:02}_ses-{session:02}_task-reachingandgrasping_run-01_eeg.vhdr",
                "synthetic header",
            )
    calls = []

    def transport(url, sign, **kwargs):
        calls.append(kwargs)
        return archive

    monkeypatch.setattr(thapa2025.dl, "get_dataset_path", lambda *args: tmp_path)
    monkeypatch.setattr(thapa2025.dl, "data_dl", transport)
    dataset = Thapa2025()
    paths = dataset.data_path(1, path=str(tmp_path), verbose=False)
    assert len(paths) == 1
    assert paths[0].session == "01"
    dataset.data_path(1, path=str(tmp_path), force_update=force, verbose=False)
    assert len(calls) == 1 + int(force)
    assert calls[-1] == {"path": str(tmp_path), "force_update": force, "verbose": False}


def test_garro_transport_and_subject_local_tasks(tmp_path, monkeypatch):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sub-01/eeg/sub-01_task-free_eeg.vhdr", "header")
    metadata = tmp_path / "metadata"
    metadata.write_text("{}")
    calls = []

    def transport(url, sign, **kwargs):
        calls.append(kwargs)
        return archive if url.endswith("49987455") else metadata

    monkeypatch.setattr(garro2025.dl, "get_dataset_path", lambda *args: tmp_path)
    monkeypatch.setattr(garro2025.dl, "data_dl", transport)
    paths = Garro2025().data_path(1, path=str(tmp_path), force_update=True, verbose=False)
    assert [p.task for p in paths] == ["free"]
    assert len(calls) == 4
    assert all(
        c == {"path": str(tmp_path), "force_update": True, "verbose": False}
        for c in calls
    )


@pytest.mark.parametrize(
    "execution,triggers", [(False, [1, 3, 9, 15]), (True, [1, 5, 11, 17])]
)
def test_moving_units_first_last_and_modality(monkeypatch, execution, triggers):
    raw = mne.io.RawArray(
        np.full((4, 4001), 2e-6),
        mne.create_info(["Cz", "X", "Y", "Z"], 100, "eeg"),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations([0, 10, 20, 30], 0, [f"Trigger#{t}" for t in triggers])
    )
    dataset = MOVING2024(execution=execution)
    monkeypatch.setattr(dataset, "data_path", lambda subject: ["synthetic.edf"])
    monkeypatch.setattr(moving2024.mne.io, "read_raw_edf", lambda *a, **kw: raw)
    loaded = dataset._get_single_subject_data(1)["0"]["0"]
    np.testing.assert_allclose(loaded.get_data(), 2e-6)
    assert loaded.get_channel_types() == ["eeg", "misc", "misc", "misc"]
    events, ids = mne.events_from_annotations(loaded, event_id=dataset.event_id)
    epochs = mne.Epochs(loaded, events, ids, tmin=0, tmax=6, baseline=None, preload=True)
    assert len(epochs) == 4
    np.testing.assert_array_equal(events[:, 0], [0, 1000, 2000, 3000])


def test_target_parser_does_not_match_substrings():
    assert thapa2025._target_from_description("Tgt1") == "Tgt1"
    assert thapa2025._target_from_description("Tgt10") is None
    assert thapa2025._target_from_description("not_target1_end") is None


def test_moving_duplicate_subject_recordings_fail(tmp_path):
    for prefix in ("first", "second"):
        (tmp_path / f"{prefix}_Subj_01_bci_32_gesture.edf").touch()
    with pytest.raises(ValueError, match="Multiple"):
        MOVING2024._find_subject_file(tmp_path, 1)


def test_mind_duplicate_run_headers_fail(tmp_path):
    for directory in ("copy1", "copy2"):
        folder = tmp_path / directory
        folder.mkdir()
        (folder / "sub-01_task-mi2d_run-01_eeg.vhdr").touch()
    with pytest.raises(ValueError, match="Duplicate"):
        MIND2026._find_subject_vhdrs(tmp_path, 1)


def test_mind_keeps_reader_units_and_bad_channels(tmp_path, monkeypatch):
    header = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    sidecar = tmp_path / "sub-01_task-mi2d_run-01_events.tsv"
    codes = [4, 5] * 10 + [6, 7] * 10
    sidecar.write_text(
        "onset\tvalue\n" + "".join(f"{i}\t{c}\n" for i, c in enumerate(codes))
    )
    raw = mne.io.RawArray(
        np.full((1, 6001), 2e-6), mne.create_info(["Cz"], 100, "eeg"), verbose=False
    )
    raw.info["bads"] = ["Cz"]
    dataset = MIND2026()
    monkeypatch.setattr(dataset, "_subject_vhdrs", lambda subject: [header])
    monkeypatch.setattr(mind2026.mne.io, "read_raw_brainvision", lambda *a, **kw: raw)
    loaded = dataset._get_single_subject_data(1)["0"]["0"]
    np.testing.assert_allclose(loaded.get_data(), 2e-6)
    assert loaded.info["bads"] == ["Cz"]
    np.testing.assert_allclose(loaded.annotations.onset[[0, -1]], [0, 39])
