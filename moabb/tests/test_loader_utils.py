"""Offline tests of the shared loader helpers (fake data, no network)."""

import warnings
import zipfile
from unittest.mock import Mock, call

import mne
import numpy as np
import pytest

from moabb.datasets import utils as dsu
from moabb.datasets._openneuro_mirror import OpenNeuroMirrorMixin
from moabb.datasets.download import NemarDownloadError
from moabb.datasets.preprocessing import _is_preserved_annotation


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


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return str(path)


def test_download_and_extract_subject_zip(monkeypatch, tmp_path):
    # An extension-less download (".../content") is renamed to .zip first.
    data_dl = Mock(return_value=_zip(tmp_path / "content", {"root/x.txt": "1"}))
    monkeypatch.setattr(dsu.dl, "data_dl", data_dl)
    dsu.download_and_extract_subject_zip("u", "C", tmp_path / "out", "p", fname="a.zip")
    assert (tmp_path / "out" / "root" / "x.txt").read_text() == "1"
    assert (tmp_path / "content.zip").is_file()
    data_dl.assert_called_once_with("u", "C", "p", False, None, fname="a.zip")


def test_download_and_extract_subject_zip_corrupted(monkeypatch, tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    monkeypatch.setattr(dsu.dl, "data_dl", Mock(return_value=str(bad)))
    with pytest.raises(zipfile.BadZipFile):  # opt-in recovery only
        dsu.download_and_extract_subject_zip("u", "C", tmp_path / "out")

    good = _zip(tmp_path / "good.zip", {"root/x.txt": "1"})
    data_dl = Mock(side_effect=[str(bad), good])
    monkeypatch.setattr(dsu.dl, "data_dl", data_dl)
    with pytest.warns(UserWarning, match="Corrupted zip file detected"):
        dsu.download_and_extract_subject_zip(
            "u", "C", tmp_path / "out", redownload_corrupted=True
        )
    assert (tmp_path / "out" / "root" / "x.txt").is_file()
    assert data_dl.call_args_list == [
        call("u", "C", None, False, None, fname=None),
        call("u", "C", None, True, None, fname=None),  # forced re-download
    ]

    bad.write_bytes(b"not a zip")
    other = tmp_path / "other.zip"
    other.write_bytes(b"still not a zip")
    monkeypatch.setattr(dsu.dl, "data_dl", Mock(side_effect=[str(bad), str(other)]))
    with pytest.warns(UserWarning), pytest.raises(zipfile.BadZipFile):
        dsu.download_and_extract_subject_zip(
            "u", "C", tmp_path / "out", redownload_corrupted=True
        )


def test_rename_stimulus_codes():
    raw = mne.io.RawArray(np.zeros((1, 100)), mne.create_info(["C3"], 100.0, "eeg"))
    raw.set_annotations(
        mne.Annotations([0.1, 0.2, 0.3], 0, ["Stimulus/S  7", "Stimulus/S 12", "S 8"])
    )
    dsu.rename_stimulus_codes(raw, {7: "left_hand", 8: "right_hand"})
    assert list(raw.annotations.description) == [
        "left_hand",
        "Stimulus/S 12",
        "right_hand",
    ]


def test_edge_boundary_is_preserved():
    assert _is_preserved_annotation("EDGE boundary")
    assert _is_preserved_annotation("edge boundary")
    assert not _is_preserved_annotation("left_hand")


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
