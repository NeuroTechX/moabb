"""Regression tests for Batista2022 data-borne BrainVision inconsistencies."""

from pathlib import Path

from moabb.datasets.batista2022 import Batista2022


def test_bad_marker_reference_uses_sibling_marker_without_mutating_source(
    tmp_path, monkeypatch
):
    """Pilot headers with a stale marker name are read from a temporary header."""
    vhdr = tmp_path / "sub-02_ses-pilot1_task-neurowMIMO.vhdr"
    vmrk = vhdr.with_suffix(".vmrk")
    eeg = vhdr.with_suffix(".eeg")
    original = "[Common Infos]\nDataFile=stale-name.eeg\nMarkerFile=stale-name.vmrk\n"
    vhdr.write_text(original, encoding="utf-8")
    vmrk.write_text("Brain Vision Data Exchange Marker File, Version 1.0\n")
    eeg.touch()

    seen = {}

    def fake_reader(path, *, preload, verbose):
        temporary = Path(path)
        seen["path"] = temporary
        seen["header"] = temporary.read_text(encoding="utf-8")
        assert preload is True
        assert verbose is False
        return object()

    monkeypatch.setattr(
        "moabb.datasets.batista2022.mne.io.read_raw_brainvision", fake_reader
    )

    result = Batista2022._read_brainvision(vhdr)

    assert result is not None
    assert seen["path"] != vhdr
    assert "DataFile=sub-02_ses-pilot1_task-neurowMIMO.eeg" in seen["header"]
    assert "MarkerFile=sub-02_ses-pilot1_task-neurowMIMO.vmrk" in seen["header"]
    assert vhdr.read_text(encoding="utf-8") == original
    assert not seen["path"].exists()


def test_existing_marker_reference_reads_original_header(tmp_path, monkeypatch):
    """Well-formed recordings keep MNE's standard read path."""
    vhdr = tmp_path / "recording.vhdr"
    vhdr.write_text("MarkerFile=recording.vmrk\n", encoding="utf-8")
    vhdr.with_suffix(".vmrk").touch()
    seen = []

    def fake_reader(path, *, preload, verbose):
        seen.append(Path(path))
        return object()

    monkeypatch.setattr(
        "moabb.datasets.batista2022.mne.io.read_raw_brainvision", fake_reader
    )

    Batista2022._read_brainvision(vhdr)

    assert seen == [vhdr]
