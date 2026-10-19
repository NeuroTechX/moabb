"""Mainsah2025: build the file list from the store's provenance, not PhysioNet."""

import hashlib
import json
import os
import socket

import pytest

import moabb.datasets.download as dl
import moabb.datasets.mainsah2025 as mainsah
from moabb.datasets import Mainsah2025_B


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
    mainsah._manifest_cache = None
    yield root
    mainsah._manifest_cache = None


REL = "bigP3BCI-data/StudyB/B_05/SE001/Test/B_05/a.edf"


def test_manifest_from_store_provenance(mne_data, tmp_path, monkeypatch):
    store = tmp_path / "store"
    data = b"edf-bytes"
    (store / REL).parent.mkdir(parents=True)
    (store / REL).write_bytes(data)
    prov = {
        "files": [
            {
                "name": REL,
                "url": "https://physionet.org/files/bigp3bci/1.0.0/" + REL,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ]
    }
    (store / "sourcedata_provenance.json").write_text(json.dumps(prov))
    monkeypatch.setattr(
        socket.socket, "connect", lambda s, a: (_ for _ in ()).throw(AssertionError(a))
    )
    monkeypatch.setattr(
        dl,
        "download_if_missing",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("upstream manifest")),
    )
    with dl.active_sourcedata_store(store):
        manifest = Mainsah2025_B._get_manifest()
    assert manifest["B"]["B_05"][1] == [REL]
