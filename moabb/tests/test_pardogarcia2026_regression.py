"""Regression tests for PardoGarcia2026 BrainVision references."""

from pathlib import Path

from moabb.datasets.pardogarcia2026 import PardoGarcia2026


def test_stale_marker_reference_uses_unambiguous_sibling(tmp_path, monkeypatch):
    vhdr = tmp_path / "PAC03-POST.vhdr"
    original = (
        "[Common Infos]\n"
        "DataFile=PAC03-POST.eeg\n"
        "MarkerFile=PAC02-POST.vmrk\n"
    )
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
        "moabb.datasets.pardogarcia2026.mne.io.read_raw_brainvision",
        fake_reader,
    )

    assert PardoGarcia2026._read_brainvision(vhdr) is not None
    assert seen["path"] != vhdr
    assert "MarkerFile=PAC03-POST.vmrk" in seen["header"]
    assert vhdr.read_text(encoding="utf-8") == original
    assert not seen["path"].exists()
