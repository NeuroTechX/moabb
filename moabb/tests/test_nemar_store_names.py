"""Files a NEMAR deposit keeps under another name than the URL (offline).

AguileraRodriguez2025's deposit (nm000174) keeps Mendeley's extension-less
``<uuid>/file_downloaded`` as ``file_downloaded.edf``/``.xdf``.
"""

import os
import socket

import pytest

import moabb.datasets.download as dl
from moabb.datasets.aguilera_rodriguez2025 import (
    _SUBJECT_URLS,
    _SUBJECT_URLS_GAMIFIED,
    AguileraRodriguez2025,
)


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
