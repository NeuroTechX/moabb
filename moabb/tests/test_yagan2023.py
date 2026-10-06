"""Offline regressions for the Yagan2023 P300 speller archive loader."""

import zipfile
from pathlib import Path
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from moabb.datasets import Yagan2023, download


def raw(descriptions, onsets, channels=("C3", "C4")):
    data = mne.io.RawArray(
        np.full((len(channels), 1001), 2e-6),
        mne.create_info(list(channels), 1000, "eeg"),
        verbose=False,
    )
    data.info["bads"] = [channels[0]]
    data.set_annotations(mne.Annotations(onsets, 0.0, descriptions))
    return data


def write_header(directory, stem, declared_marker=None, create_declared=True):
    """Write a minimal BrainVision header set that the loader can inspect.

    ``declared_marker`` is the ``MarkerFile=`` entry; by default the block
    declares its own ``.vmrk`` and that file exists, i.e. a self-consistent
    header that needs no override.
    """
    directory = Path(directory)
    declared = f"{stem}.vmrk" if declared_marker is None else declared_marker
    vhdr = directory / f"{stem}.vhdr"
    vhdr.write_text(f"MarkerFile={declared}\n")
    if create_declared:
        (directory / declared).write_text("")
    return vhdr


def _assert_invalid_subject_rejected(dataset):
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(999)


def test_s14_pairing_window_and_marker_filtering(tmp_path, monkeypatch):
    """Flashes pair with an S14 only inside the forward 10 ms window."""
    data = raw(
        [
            "Stimulus/S  1",  # flash, S14 3 ms later -> Target
            "Stimulus/S 14",
            "Stimulus/S  2",  # flash, S14 15 ms later (outside window) -> NonTarget
            "Stimulus/S 14",
            "Stimulus/S  3",  # flash without S14 -> NonTarget
            "Stimulus/S  4",  # flash with S14 1 ms *before* it -> NonTarget
            "Stimulus/S 14",
            "Stimulus/S 15",  # ISI marker -> dropped
            "Stimulus/S 14",  # unpaired target marker -> dropped
        ],
        [0.0, 0.003, 0.1, 0.115, 0.2, 0.3, 0.299, 0.4, 0.5],
    )
    monkeypatch.setattr(mne.io, "read_raw_brainvision", Mock(return_value=data))
    result = Yagan2023._get_single_run_data(str(write_header(tmp_path, "s1b1")))
    assert result.annotations.onset.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert list(result.annotations.description) == [
        "Target",
        "NonTarget",
        "NonTarget",
        "NonTarget",
    ]
    stim = result.get_data(picks="STIM")[0]
    assert stim[[0, 100, 200, 300]].tolist() == [2, 1, 1, 1]


def test_run_without_flashes_rejected(tmp_path, monkeypatch):
    data = raw(["Stimulus/S 15"], [0.0])
    monkeypatch.setattr(mne.io, "read_raw_brainvision", Mock(return_value=data))
    with pytest.raises(ValueError, match="intensification markers"):
        Yagan2023._get_single_run_data(str(write_header(tmp_path, "s1b1")))


def test_misdeclared_marker_file_gets_an_override(tmp_path, monkeypatch):
    """A header pointing at another subject's ``.vmrk`` is redirected to its own."""
    vhdr = write_header(
        tmp_path, "s12b10", declared_marker="s20b10.vmrk", create_declared=False
    )
    reader = Mock(return_value=raw(["Stimulus/S  1", "Stimulus/S 14"], [0.0, 0.003]))
    monkeypatch.setattr(mne.io, "read_raw_brainvision", reader)
    Yagan2023._get_single_run_data(str(vhdr))
    assert reader.call_args.kwargs["overrides"] == {"marker_fname": "s12b10.vmrk"}


def test_self_consistent_header_gets_no_override(tmp_path, monkeypatch):
    vhdr = write_header(tmp_path, "s1b1")
    reader = Mock(return_value=raw(["Stimulus/S  1", "Stimulus/S 14"], [0.0, 0.003]))
    monkeypatch.setattr(mne.io, "read_raw_brainvision", reader)
    Yagan2023._get_single_run_data(str(vhdr))
    assert reader.call_args.kwargs["overrides"] is None


def test_marker_file_disabled_header_gets_no_override(tmp_path, monkeypatch):
    vhdr = write_header(tmp_path, "s1b1", declared_marker="false", create_declared=False)
    reader = Mock(return_value=raw(["Stimulus/S  1", "Stimulus/S 14"], [0.0, 0.003]))
    monkeypatch.setattr(mne.io, "read_raw_brainvision", reader)
    Yagan2023._get_single_run_data(str(vhdr))
    assert reader.call_args.kwargs["overrides"] is None


def test_data_path_skips_duplicate_exports(tmp_path, monkeypatch):
    """``s5b7(1).vhdr`` duplicates ``s5b7`` and is skipped; blocks sort numerically."""
    extracted = tmp_path / "MNE-yagan2023-data" / "extracted" / "Subject_1"
    extracted.mkdir(parents=True)
    for name in ["s1b1.vhdr", "s1b10.vhdr", "s1b2.vhdr", "s1b2(1).vhdr"]:
        (extracted / name).touch()
    monkeypatch.setattr(
        download, "data_dl", Mock(return_value=str(tmp_path / "dataset.zip"))
    )
    ds = Yagan2023()
    paths = ds.data_path(1, path=tmp_path, verbose="ERROR")
    assert [Path(p).name for p in paths] == ["s1b1.vhdr", "s1b2.vhdr", "s1b10.vhdr"]
    assert download.data_dl.call_args.kwargs["fname"] == "dataset.zip"
    _assert_invalid_subject_rejected(ds)


def test_download_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("dataset/Subject_1/s1b1.vhdr", "synthetic")
    data_dl = Mock(return_value=str(archive))
    monkeypatch.setattr(download, "data_dl", data_dl)
    ds = Yagan2023()
    paths = ds.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert data_dl.call_args.args[2:] == (tmp_path,)
    assert data_dl.call_args.kwargs == {
        "force_update": True,
        "verbose": "ERROR",
        "fname": "dataset.zip",
    }
    assert [Path(p).name for p in paths] == ["s1b1.vhdr"]
    extracted = tmp_path / "MNE-yagan2023-data" / "extracted"
    assert (extracted / "dataset" / "Subject_1" / "s1b1.vhdr").read_text() == "synthetic"


def test_unreadable_blocks_are_skipped_not_fatal(tmp_path, monkeypatch):
    ds = Yagan2023()
    headers = [write_header(tmp_path, f"s1b{i}") for i in (1, 2, 3)]
    monkeypatch.setattr(ds, "data_path", Mock(return_value=[str(p) for p in headers]))
    first = raw(["Stimulus/S  1", "Stimulus/S 14"], [0.0, 0.003])
    third = raw(["Stimulus/S  2"], [0.0])
    monkeypatch.setattr(
        mne.io,
        "read_raw_brainvision",
        Mock(side_effect=[first, RuntimeError("overflow"), third]),
    )
    runs = ds._get_single_subject_data(1)["0"]
    assert set(runs) == {"0", "2"}
    assert runs["0"].annotations.description.tolist() == ["Target"]
    assert runs["2"].annotations.description.tolist() == ["NonTarget"]


def test_all_blocks_unreadable_raises(tmp_path, monkeypatch):
    ds = Yagan2023()
    headers = [write_header(tmp_path, f"s1b{i}") for i in (1, 2)]
    monkeypatch.setattr(ds, "data_path", Mock(return_value=[str(p) for p in headers]))
    monkeypatch.setattr(
        mne.io, "read_raw_brainvision", Mock(side_effect=RuntimeError("empty"))
    )
    with pytest.raises(FileNotFoundError, match="No loadable blocks"):
        ds._get_single_subject_data(1)
