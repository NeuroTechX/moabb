"""Chang2025: the store lookup must match the deposit's MNE-chang2025-data/ prefix."""

import os
import socket
from pathlib import Path

import pytest

import moabb.datasets.download as dl
from moabb.datasets import Chang2025


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


def test_chang2025_store_lookup_uses_deposit_prefix(tmp_path, monkeypatch):
    store = tmp_path / "store"
    stored = store / "MNE-chang2025-data" / "files" / "57432889"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(b"zip-bytes")
    monkeypatch.setattr(
        socket.socket, "connect", lambda s, a: (_ for _ in ()).throw(AssertionError(a))
    )
    with dl.active_sourcedata_store(store):
        hit = dl._store_lookup(
            "https://ndownloader.figshare.com/files/57432889",
            fname="MNE-chang2025-data/files/57432889",
        )
    assert hit == str(stored)


def test_chang2025_data_path_serves_zip_from_store(mne_data, tmp_path, monkeypatch):
    import zipfile

    store = tmp_path / "store"
    zip_path = store / "MNE-chang2025-data" / "files" / "57432889"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("S201/session/run.set", "set")
    monkeypatch.setattr(dl, "data_dl", dl.data_dl)  # real data_dl, store-backed
    called = []
    monkeypatch.setattr(
        dl,
        "_retrieve_with_physionet_fallback",
        lambda *a, **k: called.append(a) or str(Path(k["path"]) / k["fname"]),
    )
    monkeypatch.setattr(
        socket.socket, "connect", lambda s, a: (_ for _ in ()).throw(AssertionError(a))
    )
    with dl.active_sourcedata_store(store):
        returned = Chang2025().data_path(1)
    # data_dl still calls retrieve (pooch returns the cached file without
    # network when it exists); the store file must be the one extracted.
    assert list(Path(returned, "S201").rglob("*.set"))
