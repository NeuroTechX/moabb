"""Files a NEMAR deposit keeps under another name than the URL (offline).

AguileraRodriguez2025's deposit (nm000174) keeps Mendeley's extension-less
``<uuid>/file_downloaded`` as ``file_downloaded.edf``/``.xdf``; HefmiIch2025's
deposit (nm000347) keeps the figshare file list and the epoch files under the
loader's local names.
"""

import json
import os
import socket

import pytest

import moabb.datasets.download as dl
from moabb.datasets import hefmi_ich2025
from moabb.datasets.aguilera_rodriguez2025 import (
    _SUBJECT_URLS,
    _SUBJECT_URLS_GAMIFIED,
    AguileraRodriguez2025,
)
from moabb.datasets.hefmi_ich2025 import HefmiIch2025


MENDELEY = "https://data.mendeley.com/public-files/datasets/57g8z63tmy/files"


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


def _uuid(url):
    return url.rstrip("/").split("/")[-2]


def test_store_lookup_accepts_a_single_added_extension(tmp_path):
    store = tmp_path / "store"
    (store / "abc").mkdir(parents=True)
    (store / "abc" / "file_downloaded.xdf").write_text("x")
    with dl.active_sourcedata_store(store):
        found = dl._store_lookup(f"{MENDELEY}/abc/file_downloaded")
    assert found == str(store / "abc" / "file_downloaded.xdf")


def test_store_lookup_prefers_the_exact_name(tmp_path):
    store = tmp_path / "store"
    (store / "abc").mkdir(parents=True)
    (store / "abc" / "file_downloaded").write_text("exact")
    (store / "abc" / "file_downloaded.xdf").write_text("variant")
    with dl.active_sourcedata_store(store):
        found = dl._store_lookup(f"{MENDELEY}/abc/file_downloaded")
    assert found == str(store / "abc" / "file_downloaded")


@pytest.mark.parametrize(
    "stored,url",
    [
        # two candidates: ambiguous
        (["abc/file_downloaded.xdf", "abc/file_downloaded.edf"], "abc/file_downloaded"),
        # one-segment tails never take a variant
        (["file_downloaded.xdf"], "abc/file_downloaded"),
        # a URL name that has an extension is matched exactly only
        (["abc/data.mat.gz"], "abc/data.mat"),
    ],
)
def test_store_lookup_variant_misses(tmp_path, stored, url):
    store = tmp_path / "store"
    for name in stored:
        (store / name).parent.mkdir(parents=True, exist_ok=True)
        (store / name).write_text("x")
    with dl.active_sourcedata_store(store):
        assert dl._store_lookup(f"{MENDELEY}/{url}") is None


def test_aguilera_rodriguez2025_reads_the_deposit_names(mne_data, tmp_path, monkeypatch):
    store = tmp_path / "store"
    stored = [
        store / _uuid(_SUBJECT_URLS[1]) / "file_downloaded.edf",
        store / _uuid(_SUBJECT_URLS_GAMIFIED[1]) / "file_downloaded.xdf",
    ]
    for path in stored:
        path.parent.mkdir(parents=True)
        path.write_bytes(path.suffix.encode())
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)

    with dl.active_sourcedata_store(store):
        returned = AguileraRodriguez2025().data_path(1)

    assert [os.path.splitext(p)[1] for p in returned] == [".edf", ".xdf"]
    assert [os.stat(p).st_ino for p in returned] == [p.stat().st_ino for p in stored]


def test_hefmi_ich2025_reads_filelist_and_epochs_from_the_store(
    mne_data, tmp_path, monkeypatch
):
    store = tmp_path / "store"
    store.mkdir()
    filelist = [
        {"id": 11, "name": "1_epo.mat", "size": 41_000_000},
        {"id": 12, "name": "2_epo.mat", "size": 42_000_000},
        {"id": 13, "name": "1_epo.mat", "size": 28_000_000},  # fNIRS
        {"id": 21, "name": "1_epo.mat", "size": 41_500_000},
    ]
    (store / "figshare_filelist_28955456.json").write_text(json.dumps(filelist))
    for name in ("sub01_1_epo.mat", "sub01_2_epo.mat"):
        (store / name).write_text(name)
    monkeypatch.setattr(hefmi_ich2025, "_MANIFEST", None)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)

    def _no_filelist(*args, **kwargs):
        raise AssertionError("figshare API called")

    monkeypatch.setattr(dl, "fs_get_file_list", _no_filelist)

    with dl.active_sourcedata_store(store):
        returned = HefmiIch2025().data_path(1)

    base = mne_data / "MNE-hefmiich2025-data"
    assert returned == [str(base / "sub01_1_epo.mat"), str(base / "sub01_2_epo.mat")]
    for path in returned:
        assert os.stat(path).st_ino == (store / os.path.basename(path)).stat().st_ino
    assert hefmi_ich2025._MANIFEST[2] == [(21, "sub02_1_epo.mat")]


def test_hefmi_ich2025_caches_the_api_filelist(mne_data, monkeypatch):
    filelist = [{"id": 11, "name": "1_epo.mat", "size": 41_000_000}]
    calls = []

    def fake_filelist(article_id, version=None):
        calls.append(article_id)
        return filelist

    monkeypatch.setattr(dl, "fs_get_file_list", fake_filelist)
    base = mne_data / "MNE-hefmiich2025-data"

    assert HefmiIch2025._figshare_filelist(base) == filelist
    assert HefmiIch2025._figshare_filelist(base) == filelist
    assert calls == [28955456]
    assert json.loads((base / "figshare_filelist_28955456.json").read_text()) == filelist
