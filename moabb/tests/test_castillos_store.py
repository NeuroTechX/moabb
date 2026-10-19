"""Castillos: the shared 4Class-CVEP tree comes from the store, once."""

import os
import socket
from pathlib import Path

import pytest

import moabb.datasets.download as dl
from moabb.datasets import CastillosCVEP40


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


def test_castillos_reads_shared_tree_from_store(mne_data, tmp_path, monkeypatch):
    store = tmp_path / "store"
    member = store / "4Class-CVEP" / "P1" / "P1_mseq40.set"
    member.parent.mkdir(parents=True)
    # A real (tiny) EEGLAB-style .set whose data is stored inline, so
    # _missing_companions finds no external .fdt reference.
    from scipy.io import savemat

    savemat(member, {"data": ""})
    monkeypatch.setattr(
        socket.socket, "connect", lambda s, a: (_ for _ in ()).throw(AssertionError(a))
    )
    monkeypatch.setattr(
        dl, "data_dl", lambda *a, **k: (_ for _ in ()).throw(AssertionError("data_dl"))
    )
    ds = CastillosCVEP40()
    with dl.active_sourcedata_store(store):
        paths = ds.data_path(1, ds.paradigm_type)
    assert paths and Path(paths[0]).stat().st_ino == member.stat().st_ino
