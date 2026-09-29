"""Offline transport and scientific contracts for the large-recording batch."""

import zipfile
from pathlib import Path

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


def test_mind_transport_flags_and_cached_subject(tmp_path, monkeypatch):
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
    dataset = MIND2026()
    dataset.data_path(1, path=str(tmp_path), verbose=False)
    assert calls == []
    dataset.data_path(1, path=str(tmp_path), force_update=True, verbose=False)
    assert calls == [{"path": str(tmp_path), "force_update": True, "verbose": False}]
    assert header.read_text() == "updated"
    assert not (root / "MIND_BIDS/sub-01/nirs").exists()


def test_thapa_transport_and_missing_sessions(tmp_path, monkeypatch):
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
    assert calls == [{"path": str(tmp_path), "force_update": False, "verbose": False}]
    dataset.data_path(1, path=str(tmp_path), verbose=False)
    assert len(calls) == 1
    dataset.data_path(1, path=str(tmp_path), force_update=True, verbose=False)
    assert len(calls) == 2
    assert calls[-1] == {"path": str(tmp_path), "force_update": True, "verbose": False}


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


# --- MIND2026 event-stream guards

_VALID_RUN_CODES = [4, 5] * 10 + [6, 7] * 10


def _write_events(path, coded_events):
    rows = ["onset\tduration\ttrial_type\tvalue"]
    rows.extend(f"{onset:.3f}\t0.0\tevent_{code}\t{code}" for onset, code in coded_events)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _raw_with_codes(codes):
    sfreq = 100.0
    raw = mne.io.RawArray(
        np.zeros((1, (len(codes) + 1) * int(sfreq))),
        mne.create_info(["Cz"], sfreq=sfreq, ch_types="eeg"),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=np.arange(len(codes), dtype=float),
            duration=np.zeros(len(codes)),
            description=[f"Comment/{code}" for code in codes],
        )
    )
    return raw


_LABELS = {4: "left_to_right", 5: "up_to_down"}
_LABELS.update({6: "upperleft_to_lowerright", 7: "upperright_to_lowerleft"})
_SUB19_PREFIX = [1, 3] + [4, 5] * 9 + [800000, 800001, 1, 2, 3]


@pytest.mark.parametrize(
    "source, codes",
    [
        # Balanced run: accepted on class counts alone; the code-3 cue is dropped.
        ("tsv", [3] + [4, 5, 6, 7] * 10),
        # Subject 4: aborted MI markers before the final code-1 restart (sidecar).
        ("tsv", [1, 1, 3, 5, 7, 15, 63, 255, 7, 1, 3] + _VALID_RUN_CODES),
        # Subject 19: same restart guard on the BrainVision-marker fallback.
        ("markers", _SUB19_PREFIX + _VALID_RUN_CODES),
    ],
    ids=["balanced", "sub04-tsv-restart", "sub19-marker-restart"],
)
def test_mi_events_keep_one_complete_run(tmp_path, source, codes):
    vhdr = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    if source == "tsv":
        _write_events(tmp_path / "sub-01_task-mi2d_run-01_events.tsv", enumerate(codes))
    annotations = MIND2026._mi_annotations(
        vhdr, _raw_with_codes(codes if source == "markers" else [])
    )
    assert annotations.description.tolist() == [_LABELS[c] for c in codes[-40:]]
    np.testing.assert_allclose(annotations.onset, np.arange(len(codes) - 40, len(codes)))


@pytest.mark.parametrize(
    "codes, match",
    [
        (_VALID_RUN_CODES + [4], "final code 1 restart marker"),
        ([4, 1] + [4, 5, 6, 7] * 10, "20 codes 4/5 followed by 20 codes 6/7"),
    ],
    ids=["no-restart", "half-order"],
)
def test_overfull_run_fails_closed(tmp_path, codes, match):
    """Extra MI markers are only trimmed after a restart, in released task order."""
    vhdr = tmp_path / "sub-01_task-mi2d_run-01_eeg.vhdr"
    with pytest.raises(RuntimeError, match=match):
        MIND2026._mi_annotations(vhdr, _raw_with_codes(codes))


# --- Thapa2025 published BIDS irregularities


def test_stale_brainvision_references_use_same_stem_bids_siblings(tmp_path, monkeypatch):
    vhdr = tmp_path / "sub-09_task-reachingandgrasping_run-0009_eeg.vhdr"
    vhdr.write_text(
        "[Common Infos]\nDataFile=correct.eeg\nMarkerFile=stale_acquisition_name.vmrk\n",
        encoding="utf-8",
    )
    vhdr.with_suffix(".eeg").touch()
    vhdr.with_suffix(".vmrk").touch()
    seen = {}

    def fake_reader(path, **kwargs):
        temporary = Path(path)
        seen["path"] = temporary
        seen["header"] = temporary.read_text(encoding="utf-8")
        return object()

    monkeypatch.setattr(
        "moabb.datasets.thapa2025.mne.io.read_raw_brainvision", fake_reader
    )

    assert Thapa2025._read_brainvision(vhdr) is not None
    assert seen["path"] != vhdr
    assert f"DataFile={vhdr.with_suffix('.eeg').name}" in seen["header"]
    assert f"MarkerFile={vhdr.with_suffix('.vmrk').name}" in seen["header"]
    assert not seen["path"].exists()


def test_irregular_optional_events_column_preserves_all_protocol_events(tmp_path):
    events = tmp_path / "sub-13_events.tsv"
    events.write_text(
        "onset\tduration\ttrial_type\tstim_file\n"
        "1.0\tn/a\ttrial_start\tstimuli/Start.wav\n"
        "2.0\tn/a\ttarget 4\n"
        "3.0\tn/a\ttrial_end\tstimuli/End.wav\t\n"
        "4.0\tn/a\tTgt10\n",
        encoding="utf-8",
    )

    annotations = Thapa2025._annotations_from_events(events)

    # Target variants normalise to Tgt{n}; "Tgt10" is not a substring match.
    assert annotations.description.tolist() == [
        "trial_start",
        "Tgt4",
        "trial_end",
        "Tgt10",
    ]
    assert annotations.onset.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert annotations.duration.tolist() == [0.0] * 4


def test_events_sidecar_stays_beside_brainvision_header(tmp_path):
    eeg_dir = tmp_path / "sub-09" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    header = eeg_dir / "sub-09_ses-01_task-reachingandgrasping_run-0001_eeg.vhdr"

    events = Thapa2025._events_path_for_header(header)

    assert events == eeg_dir / (
        "sub-09_ses-01_task-reachingandgrasping_run-0001_events.tsv"
    )
