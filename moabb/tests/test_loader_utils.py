"""Offline tests of the shared loader helpers (fake data, no network)."""

import configparser
import json
import warnings
import zipfile
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from moabb.datasets import utils as dsu
from moabb.datasets._openneuro_mirror import (
    OpenNeuroMirrorMixin,
    drop_native_stim,
    relabel_annotations,
    write_dataset_description,
)
from moabb.datasets.download import NemarDownloadError
from moabb.datasets.preprocessing import _is_preserved_annotation


# --------------------------------------------------------------------------- montage
def test_montage_helpers_emit_no_deprecation_warning():
    raw = mne.io.RawArray(
        np.zeros((3, 10)), mne.create_info(["FP1", "CZ", "STI"], 100.0, "eeg")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dsu.set_neuroscan_montage(raw)
        built = dsu.build_raw_from_epochs(
            np.ones((2, 2, 4)), ["C3", "Cz"], 100.0, [1, 2], "colin27_1020"
        )
    assert raw.ch_names[:2] == ["Fp1", "Cz"]
    assert not np.isnan(raw.get_montage().get_positions()["ch_pos"]["Cz"]).any()
    assert built.get_montage() is not None


# ----------------------------------------------------------------------------- zip
def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


@pytest.fixture
def fake_dl(monkeypatch, tmp_path):
    """Replace ``data_dl`` by a queue of prepared archives; record the calls."""
    calls, queue = [], []

    def data_dl(url, sign, path, force_update, verbose, fname=None):
        calls.append((url, sign, path, force_update, fname))
        return str(queue.pop(0) if len(queue) > 1 else queue[0])

    monkeypatch.setattr(dsu.dl, "data_dl", data_dl)
    return calls, queue


def test_download_and_extract_zip_extracts_once(fake_dl, tmp_path):
    calls, queue = fake_dl
    queue.append(_zip(tmp_path / "a.zip", {"root/x.txt": "1"}))
    target = dsu.download_and_extract_zip("u", "Code", "root", path="p", fname="a.zip")
    assert target == tmp_path / "root" and (target / "x.txt").read_text() == "1"

    (target / "x.txt").write_text("kept")  # a second call must not re-extract
    assert dsu.download_and_extract_zip("u", "Code", "root") == target
    assert (target / "x.txt").read_text() == "kept"
    dsu.download_and_extract_zip("u", "Code", "root", force_update=True)
    assert (target / "x.txt").read_text() == "1"
    assert calls == [
        ("u", "Code", "p", False, "a.zip"),
        ("u", "Code", None, False, None),
        ("u", "Code", None, True, None),
    ]


def test_download_and_extract_zip_into_subfolder(fake_dl, tmp_path):
    _, queue = fake_dl
    queue.append(_zip(tmp_path / "a.zip", {"EXP/x.txt": "1"}))
    target = dsu.download_and_extract_zip("u", "C", "EXP", extract_to="sub")
    assert target == tmp_path / "sub" / "EXP" and target.is_dir()
    assert dsu.download_and_extract_zip("u", "C", ".", extract_to="sub") == target.parent


def test_download_and_extract_zip_corrupted_archive(fake_dl, tmp_path):
    calls, queue = fake_dl
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    queue.extend([bad, bad])
    with pytest.raises(zipfile.BadZipFile):  # opt-in recovery only
        dsu.download_and_extract_zip("u", "C", "root")

    queue[:] = [bad, _zip(tmp_path / "good.zip", {"root/x.txt": "1"})]
    calls.clear()
    with pytest.warns(UserWarning, match="Corrupted zip file detected"):
        target = dsu.download_and_extract_zip("u", "C", "root", redownload_corrupted=True)
    assert (target / "x.txt").is_file()
    assert [c[3] for c in calls] == [False, True]  # forced re-download

    queue[:] = [bad, bad]
    with pytest.warns(UserWarning), pytest.raises(zipfile.BadZipFile):
        dsu.download_and_extract_zip("u", "C", "other", redownload_corrupted=True)


# ---------------------------------------------------------------------- BrainVision
def _write_brainvision(folder, stem, data_file=None, marker_file=None, fields=True):
    """Write a minimal 2-channel float32 BrainVision triplet named ``stem``."""
    folder.mkdir(parents=True, exist_ok=True)
    data = np.arange(20, dtype="<f4").reshape(10, 2)  # multiplexed samples
    (folder / f"{stem}.eeg").write_bytes(data.tobytes())
    (folder / f"{stem}.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n\n[Common Infos]\n"
        f"Codepage=UTF-8\nDataFile={stem}.eeg\n\n[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0\nMk2=Stimulus,S  7,3,1,0\nMk3=Stimulus,S 12,5,1,0\n",
        encoding="utf-8",
    )
    refs = (
        f"DataFile={data_file or stem + '.eeg'}\n"
        f"MarkerFile={marker_file or stem + '.vmrk'}\n"
        if fields
        else ""
    )
    vhdr = folder / f"{stem}.vhdr"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n\n[Common Infos]\n"
        f"Codepage=UTF-8\n{refs}DataFormat=BINARY\nDataOrientation=MULTIPLEXED\n"
        "NumberOfChannels=2\nSamplingInterval=4000\n\n[Binary Infos]\n"
        "BinaryFormat=IEEE_FLOAT_32\n\n[Channel Infos]\nCh1=C3,,1,µV\nCh2=C4,,1,µV\n",
        encoding="utf-8",
    )
    return vhdr


@pytest.mark.parametrize("strict", [False, True])
def test_read_raw_brainvision_repaired_reads_intact_header(tmp_path, strict):
    vhdr = _write_brainvision(tmp_path, "run")
    raw = dsu.read_raw_brainvision_repaired(vhdr, strict=strict)
    assert raw.ch_names == ["C3", "C4"] and raw.n_times == 10
    assert raw.info["sfreq"] == 250.0


@pytest.mark.parametrize("strict", [False, True])
def test_read_raw_brainvision_repaired_fixes_stale_references(tmp_path, strict):
    vhdr = _write_brainvision(tmp_path, "sub-01_run", "Old_Name.eeg", "Old_Name.vmrk")
    original = vhdr.read_text(encoding="utf-8")
    raw = dsu.read_raw_brainvision_repaired(str(vhdr), strict=strict)
    np.testing.assert_allclose(raw.get_data()[0, :2], [0, 2e-6])
    assert "Stimulus/S  7" in raw.annotations.description
    assert vhdr.read_text(encoding="utf-8") == original  # download untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "sub-01_run.eeg",
        "sub-01_run.vhdr",
        "sub-01_run.vmrk",
    ]


def test_read_raw_brainvision_repaired_strict_errors(tmp_path):
    no_fields = _write_brainvision(tmp_path / "a", "run", fields=False)
    with pytest.raises(ValueError, match="Missing DataFile entry"):
        dsu.read_raw_brainvision_repaired(no_fields, strict=True)
    with pytest.raises(configparser.Error):  # lenient: MNE reports the bad header
        dsu.read_raw_brainvision_repaired(no_fields)

    orphan = _write_brainvision(tmp_path / "b", "run", "Old.eeg")
    (tmp_path / "b" / "run.eeg").unlink()
    with pytest.raises(FileNotFoundError, match="sibling run.eeg is also absent"):
        dsu.read_raw_brainvision_repaired(orphan, strict=True)
    with pytest.raises(OSError):  # lenient mode leaves the failure to MNE
        dsu.read_raw_brainvision_repaired(orphan)


def test_rename_stimulus_codes(tmp_path):
    raw = dsu.read_raw_brainvision_repaired(_write_brainvision(tmp_path, "run"))
    dsu.rename_stimulus_codes(raw, {7: "left_hand", 8: "right_hand"})
    assert list(raw.annotations.description) == ["left_hand", "Stimulus/S 12"]


# ------------------------------------------------------------------- EDGE boundary
def test_edge_boundary_annotations_survive_event_rederivation():
    ann = dsu.edge_boundary_annotations([1.0, 2.5])
    np.testing.assert_array_equal(ann.onset, [1.0, 2.5])
    np.testing.assert_array_equal(ann.duration, [0.0, 0.0])
    assert list(ann.description) == ["EDGE boundary"] * 2
    assert len(dsu.edge_boundary_annotations([])) == 0
    assert _is_preserved_annotation("EDGE boundary")
    assert _is_preserved_annotation("edge boundary")
    assert not _is_preserved_annotation("left_hand")


# ------------------------------------------------------------- OpenNeuro mirror
def _raw(ch_types, annotations=()):
    names = [f"ch{i}" for i in range(len(ch_types))]
    raw = mne.io.RawArray(
        np.zeros((len(names), 100)),
        mne.create_info(names, 100.0, ch_types),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([0.1] * len(annotations), 0, list(annotations)))
    return raw


def test_mirror_raw_helpers(tmp_path):
    raw = drop_native_stim(_raw(["eeg", "stim", "eog", "stim"]))
    assert raw.get_channel_types() == ["eeg", "eog"]

    raw = _raw(["eeg"], ["S  1", "S  2", "boundary"])
    relabel_annotations(raw, {"S  1": "left_hand", "S  2": "right_hand"})
    assert list(raw.annotations.description) == ["left_hand", "right_hand", "boundary"]

    write_dataset_description(tmp_path, "Name", "1.9.0", "10.1/x", ("A", "B"))
    written = json.loads((tmp_path / "dataset_description.json").read_text())
    assert written == {
        "Name": "Name",
        "BIDSVersion": "1.9.0",
        "License": "CC0",
        "Authors": ["A", "B"],
        "DatasetDOI": "10.1/x",
    }
    write_dataset_description(tmp_path, "Other", "1.0", "d", ())  # never overwrites
    assert json.loads((tmp_path / "dataset_description.json").read_text()) == written


class _Mirror(OpenNeuroMirrorMixin):
    subject_list = [1, 2]
    nemar_id = "on000001"

    def __init__(self):
        self.data_path = Mock()
        self._download_nemar = Mock(return_value="/mirror/root")


def test_mirror_mixin_provider_policy(monkeypatch):
    ds = _Mirror()
    assert ds.nemar_bids_filters == {"scope": "raw", "datatype": "eeg"}
    assert ds._prefetch_nemar_sourcedata([1]) is None
    with pytest.raises(ValueError, match="Invalid subject"):
        ds._mirror_root(3, None, False, None, None)

    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    assert ds._mirror_root(1, "p", True, False, "v") == "/mirror/root"
    ds._download_nemar.assert_called_once_with(1, "p", True, False, "v")
    ds._download_nemar.side_effect = NemarDownloadError("offline")
    with pytest.raises(NemarDownloadError):
        ds._mirror_root(1, None, False, None, None)

    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    with pytest.warns(RuntimeWarning, match="using its OpenNeuro source"):
        assert ds._mirror_root(1, None, False, None, None) is None

    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    ds._download_nemar.reset_mock()
    assert ds._mirror_root(1, None, False, None, None) is None
    ds._download_nemar.assert_not_called()


def test_mirror_mixin_download_iterates_data_path():
    ds = _Mirror()
    ds.download(path="p", force_update=True, verbose="v")
    assert [c.args for c in ds.data_path.call_args_list] == [
        (1, "p", True, None, "v"),
        (2, "p", True, None, "v"),
    ]
    ds.data_path.reset_mock()
    ds.download([2])
    ds.data_path.assert_called_once_with(2, None, False, None, None)
