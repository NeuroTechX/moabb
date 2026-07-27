"""Regression test for the stale Vagaja2023 subject-31 BrainVision header."""

from pathlib import Path

from moabb.datasets.vagaja2023 import Vagaja2023


def test_stale_subject31_brainvision_references_use_bids_siblings(tmp_path, monkeypatch):
    vhdr = tmp_path / "SUB31_MI.vhdr"
    original = "[Common Infos]\nDataFile=SUB31_mi_.eeg\nMarkerFile=SUB31_mi_.vmrk\n"
    vhdr.write_text(original, encoding="utf-8")
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
        "moabb.datasets.vagaja2023.mne.io.read_raw_brainvision", fake_reader
    )

    assert Vagaja2023._read_brainvision(vhdr) is not None
    assert seen["path"] != vhdr
    assert "DataFile=SUB31_MI.eeg" in seen["header"]
    assert "MarkerFile=SUB31_MI.vmrk" in seen["header"]
    assert vhdr.read_text(encoding="utf-8") == original
    assert not seen["path"].exists()
