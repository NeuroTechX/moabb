"""Regression tests for dataset-loader fixes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from moabb.datasets import download as _dl
from moabb.datasets.bnci.bnci_2020 import _convert_attention_shift
from moabb.datasets.braininvaders import BI2015b
from moabb.datasets.hefmi_ich2025 import HefmiIch2025
from moabb.datasets.kojima2024a import Kojima2024A
from moabb.datasets.mainsah2025 import _parse_manifest
from moabb.datasets.ssvep_chen2017 import Chen2017SingleFlicker
from moabb.datasets.ssvep_mamem import MAMEM1
from moabb.datasets.yang2025 import Yang2025


def _fake_bciexp(n_channels=3, n_samples=4, n_trials=5):
    rng = np.random.default_rng(0)
    data = np.asfortranarray(
        rng.standard_normal((n_channels, n_samples, n_trials)).astype(np.float64)
    )
    return SimpleNamespace(
        data=data,
        heog=rng.standard_normal((n_samples, n_trials)).astype(np.float64),
        veog=rng.standard_normal((n_samples, n_trials)).astype(np.float64),
        srate=250.0,
        label=[f"E{i + 1}" for i in range(n_channels)],
        intention=np.array(["yes", "no", "yes", "no", "yes"], dtype=object),
    )


def test_bnci2020_002_layout():
    bciexp = _fake_bciexp()
    with patch("moabb.datasets.bnci.bnci_2020.loadmat", return_value={"bciexp": bciexp}):
        raw, event_id = _convert_attention_shift("dummy.mat")

    n_channels, n_samples, n_trials = bciexp.data.shape
    expected = bciexp.data.transpose(0, 2, 1).reshape(n_channels, -1) * 1e-6
    np.testing.assert_allclose(
        raw.copy().pick("eeg").get_data(), expected, rtol=0, atol=1e-15
    )

    stim = raw.copy().pick("stim").get_data()[0]
    nonzero_idx = np.flatnonzero(stim)
    np.testing.assert_array_equal(nonzero_idx, np.arange(n_trials) * n_samples)
    assert event_id == {"NonTarget": 1, "Target": 2}
    np.testing.assert_array_equal(stim[nonzero_idx], [2, 1, 2, 1, 2])


def test_fs_get_file_list_caches():
    _dl.fs_get_file_list.cache_clear()
    side_effect = [
        [{"id": 1, "name": "a.mat", "supplied_md5": "0" * 32}],
        [{"id": 2, "name": "b.mat", "supplied_md5": "1" * 32}],
    ]
    with patch.object(_dl, "_fs_paginated_file_list", side_effect=side_effect) as page:
        first = _dl.fs_get_file_list(123456)
        second = _dl.fs_get_file_list(123456)
        third = _dl.fs_get_file_list(123456, version=2)
    assert page.call_count == 2
    assert first is second
    assert third is not first
    _dl.fs_get_file_list.cache_clear()


def test_mamem_filelist_disk_cache(tmp_path: Path):
    _dl.fs_get_file_list.cache_clear()
    ds = MAMEM1()
    sentinel = [{"id": 99, "name": "U001ai.mat", "supplied_md5": "0" * 32}]
    cache_path = Path(ds._filelist_cache_path(str(tmp_path)))

    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list", return_value=sentinel
    ) as api:
        assert ds._load_or_fetch_filelist(str(tmp_path)) == sentinel
    assert api.call_count == 1
    assert json.loads(cache_path.read_text()) == sentinel

    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list",
        side_effect=AssertionError("API hit when cache present"),
    ):
        assert ds._load_or_fetch_filelist(str(tmp_path)) == sentinel

    with patch(
        "moabb.datasets.ssvep_mamem.fs_get_file_list", return_value=sentinel
    ) as api:
        ds._load_or_fetch_filelist(str(tmp_path), force_update=True)
    assert api.call_count == 1
    _dl.fs_get_file_list.cache_clear()


def test_mamem_already_downloaded_does_not_ping_figshare(tmp_path: Path):
    _dl.fs_get_file_list.cache_clear()
    ds = MAMEM1()

    filelist = []
    for sub_id in (1, 2, 3):
        for run_letter in "ai":
            payload = f"S{sub_id:02d}{run_letter}".encode()
            file_id = sub_id * 100 + ord(run_letter)
            filelist.append(
                {
                    "id": file_id,
                    "name": f"U0{sub_id:02d}{run_letter}i.mat",
                    "supplied_md5": hashlib.md5(payload).hexdigest(),
                }
            )
            (tmp_path / str(file_id)).write_bytes(payload)

    Path(ds._filelist_cache_path(str(tmp_path))).write_text(json.dumps(filelist))

    with (
        patch.object(ds, "_dataset_root", return_value=str(tmp_path)),
        patch(
            "moabb.datasets.ssvep_mamem.fs_get_file_list",
            side_effect=AssertionError("API hit when dataset already downloaded"),
        ),
    ):
        for sub_id in (1, 2, 3):
            paths = ds.data_path(sub_id)
            assert paths
            assert all(Path(p).exists() for p in paths)
    _dl.fs_get_file_list.cache_clear()


def test_mainsah_manifest_includes_hyphenated_path_components(tmp_path: Path):
    manifest = tmp_path / "SHA256SUMS.txt"
    paths = [
        "bigP3BCI-data/StudyQ/Q_01/SE001/Train/Grey-to-White/run-1.edf",
        "bigP3BCI-data/StudyQ/Q_01/SE001/Train/Grey-to-Color/run-2.edf",
        "bigP3BCI-data/StudyQ/Q_01/SE001/Train/ColorIntensification/run-3.edf",
    ]
    manifest.write_text("\n".join(f"deadbeef  {path}" for path in reversed(paths)))

    parsed = _parse_manifest(manifest)

    assert parsed["Q"]["Q_01"][1] == sorted(paths)


def test_chen2017_data_path_finds_prefixed_xdf_files(tmp_path: Path, monkeypatch):
    dataset = Chen2017SingleFlicker()
    xdf_file = (
        tmp_path
        / "MNE-chen2017singleflicker-data"
        / "3.raw_data"
        / "1.training_data"
        / "sub_1_1.xdf"
    )
    xdf_file.parent.mkdir(parents=True)
    xdf_file.touch()
    monkeypatch.setattr(
        "moabb.datasets.ssvep_chen2017.dl.get_dataset_path", lambda *args: tmp_path
    )
    monkeypatch.setattr(
        "moabb.datasets.ssvep_chen2017.dl.data_dl",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    paths = dataset.data_path(1)

    assert paths == {"mat": [], "xdf": [str(xdf_file)]}


def test_yang2025_recognizes_extracted_bdf_files(tmp_path: Path, monkeypatch):
    data_file = (
        tmp_path
        / "MNE-yang2025-data"
        / "WBCIC_SHU Motor Imagery dataset"
        / "sourcedata"
        / "2C dataset"
        / "sub-001"
        / "ses-01"
        / "eeg"
        / "data.bdf"
    )
    data_file.parent.mkdir(parents=True)
    data_file.touch()
    monkeypatch.setattr(
        "moabb.datasets.yang2025.dl.get_dataset_path", lambda *args: tmp_path
    )
    monkeypatch.setattr(
        "moabb.datasets.yang2025.dl.data_dl",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert Yang2025().data_path(1) == str(tmp_path / "MNE-yang2025-data")


def test_hefmi_ich_data_path_downloads_subject_files(tmp_path: Path, monkeypatch):
    dataset = HefmiIch2025()
    monkeypatch.setattr(
        "moabb.datasets.hefmi_ich2025.dl.get_dataset_path", lambda *args: tmp_path
    )
    monkeypatch.setattr(
        "moabb.datasets.hefmi_ich2025._MANIFEST", {1: [(42, "sub01_1_epo.mat")]}
    )

    def fake_download(url, sign, root, force_update, verbose):
        downloaded = tmp_path / "MNE-hefmiich2025-data" / "42"
        downloaded.write_bytes(b"data")
        return downloaded

    monkeypatch.setattr("moabb.datasets.hefmi_ich2025.dl.data_dl", fake_download)

    assert dataset.data_path(1) == [
        str(tmp_path / "MNE-hefmiich2025-data" / "sub01_1_epo.mat")
    ]


def test_bi2015b_data_path_returns_existing_mat_paths(tmp_path: Path, monkeypatch):
    group_dir = tmp_path / "zenodo" / "3268762" / "group_01"
    group_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "moabb.datasets.braininvaders.dl.data_dl",
        lambda *args, **kwargs: str(tmp_path / "zenodo" / "3268762" / "group_01_mat.zip"),
    )

    paths = BI2015b().data_path(1)

    assert all(path.endswith(".mat") for path in paths)


def test_kojima2024_removes_invalid_cached_manifest(tmp_path: Path, monkeypatch):
    dataset = Kojima2024A()
    manifest_path = tmp_path / f"MNE-{dataset.code}-data" / "kojima2024_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("<html>upstream error</html>")
    monkeypatch.setattr(
        "moabb.datasets.kojima2024a.dl.download_if_missing", lambda *args: None
    )

    with pytest.raises(RuntimeError, match="invalid and has been removed"):
        dataset.download_by_subject(1, tmp_path)

    assert not manifest_path.exists()
