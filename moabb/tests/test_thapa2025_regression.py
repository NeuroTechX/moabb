"""Regression tests for Thapa2025 published BIDS irregularities."""

from pathlib import Path

from moabb.datasets.thapa2025 import Thapa2025


def test_stale_brainvision_references_use_same_stem_bids_siblings(tmp_path, monkeypatch):
    vhdr = tmp_path / "sub-09_task-reachingandgrasping_run-0009_eeg.vhdr"
    vhdr.write_text(
        "[Common Infos]\n"
        "DataFile=correct.eeg\n"
        "MarkerFile=stale_acquisition_name.vmrk\n",
        encoding="utf-8",
    )
    vhdr.with_suffix(".eeg").touch()
    vhdr.with_suffix(".vmrk").touch()
    seen = {}

    def fake_reader(path, *, preload, verbose):
        temporary = Path(path)
        seen["path"] = temporary
        seen["header"] = temporary.read_text(encoding="utf-8")
        assert preload is True
        assert verbose is False
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
        "2.0\tn/a\tTgt4\n"
        "3.0\tn/a\ttrial_end\tstimuli/End.wav\t\n",
        encoding="utf-8",
    )

    annotations = Thapa2025._annotations_from_events(events)

    assert annotations.description.tolist() == ["trial_start", "Tgt4", "trial_end"]
    assert annotations.onset.tolist() == [1.0, 2.0, 3.0]
    assert annotations.duration.tolist() == [0.0, 0.0, 0.0]


def test_events_sidecar_stays_beside_brainvision_header(tmp_path):
    eeg_dir = tmp_path / "sub-09" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    header = eeg_dir / "sub-09_ses-01_task-reachingandgrasping_run-0001_eeg.vhdr"

    events = Thapa2025._events_path_for_header(header)

    assert events == eeg_dir / (
        "sub-09_ses-01_task-reachingandgrasping_run-0001_events.tsv"
    )
