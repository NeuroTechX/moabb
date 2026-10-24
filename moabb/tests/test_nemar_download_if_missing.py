"""download_if_missing serves files from the NEMAR store (offline).

Loaders that fetch with :func:`moabb.datasets.download.download_if_missing`
instead of ``data_dl`` (Mainsah2025's PhysioNet ``SHA256SUMS.txt`` index, the
Kojima, ErpCore2021, Dreyer2023 and Lee2021Mobile files) reached the upstream
host although the deposit's ``sourcedata/`` held the file.
"""

import os
import socket

import pytest

import moabb.datasets.download as dl
from moabb.datasets import mainsah2025
from moabb.datasets.mainsah2025 import Mainsah2025_A


EDF = "bigP3BCI-data/StudyA/A_01/SE001/Test/CB/A_01_SE001_CB_Test06.edf"


@pytest.fixture
def mne_data(tmp_path, monkeypatch):
    fake_home = tmp_path / "mne-home"
    fake_home.mkdir()
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(fake_home))
    for key in tuple(os.environ):
        if key.startswith("MNE_DATASETS_"):
            monkeypatch.delenv(key)
    root = tmp_path / "mne_data"
    root.mkdir()
    monkeypatch.setenv("MNE_DATA", str(root))
    return root


def _forbid_connect(self, address):
    raise AssertionError(f"network touched: {address}")


def _no_retrieve(url, *args, **kwargs):
    raise AssertionError(f"downloaded {url}")


def test_download_if_missing_links_the_store_file(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    (store / "SHA256SUMS.txt").write_text("index")
    monkeypatch.setattr(dl, "_retrieve_with_physionet_fallback", _no_retrieve)
    target = tmp_path / "cache" / "SHA256SUMS.txt"

    with dl.active_sourcedata_store(store):
        dl.download_if_missing(
            str(target),
            "https://physionet.org/files/bigp3bci/1.0.0/SHA256SUMS.txt",
            warn_missing=False,
        )

    assert target.stat().st_ino == (store / "SHA256SUMS.txt").stat().st_ino


def test_download_if_missing_force_update_skips_the_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    (store / "SHA256SUMS.txt").write_text("stale")
    calls = []

    def fake_retrieve(url, fname, path, **kwargs):
        calls.append(url)
        with open(os.path.join(path, fname), "w") as fout:
            fout.write("fresh")
        return os.path.join(path, fname)

    monkeypatch.setattr(dl, "_retrieve_with_physionet_fallback", fake_retrieve)
    target = tmp_path / "cache" / "SHA256SUMS.txt"
    url = "https://physionet.org/files/bigp3bci/1.0.0/SHA256SUMS.txt"

    with dl.active_sourcedata_store(store):
        dl.download_if_missing(str(target), url, warn_missing=False, force_update=True)

    assert calls == [url]
    assert target.read_text() == "fresh"
    assert (store / "SHA256SUMS.txt").read_text() == "stale"


def test_download_if_missing_without_store_downloads(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        dl, "_retrieve_with_physionet_fallback", lambda url, **kwargs: calls.append(url)
    )
    dl.download_if_missing(
        str(tmp_path / "a.txt"), "https://example.org/a.txt", warn_missing=False
    )
    assert calls == ["https://example.org/a.txt"]


def test_mainsah2025_reads_sha256sums_from_the_deposit(mne_data, tmp_path, monkeypatch):
    """Every Mainsah2025 deposit carries the index at sourcedata/SHA256SUMS.txt."""
    store = tmp_path / "store"
    store.mkdir()
    (store / "SHA256SUMS.txt").write_text(f"{'0' * 64}  {EDF}\n")
    monkeypatch.setattr(mainsah2025, "_manifest_cache", None)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    monkeypatch.setattr(dl, "_retrieve_with_physionet_fallback", _no_retrieve)

    with dl.active_sourcedata_store(store):
        manifest = Mainsah2025_A()._get_subject_manifest(1)

    assert manifest == {1: [EDF]}
    cached = mne_data / "MNE-Mainsah2025-data" / "SHA256SUMS.txt"
    assert cached.stat().st_ino == (store / "SHA256SUMS.txt").stat().st_ino
