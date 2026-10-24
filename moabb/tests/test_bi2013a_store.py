"""BI2013a: use the extracted runs in the NEMAR deposit instead of the zips."""

import os
import socket
from pathlib import Path

import pytest

import moabb.datasets.download as dl
from moabb.datasets import BI2013a


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


META = """runs:
  - filename: run_1.gdf
    experimental_condition: nonadaptive
    type: training
"""


def test_bi2013a_reads_extracted_runs_from_store(mne_data, tmp_path, monkeypatch):
    store = tmp_path / "store"
    # Deposit layout (nm000264): subject_01/Session1/<run>.mat + meta.yml
    run = store / "subject_08" / "Session1" / "run_1.mat"
    run.parent.mkdir(parents=True)
    run.write_text("mat")
    (store / "subject_08" / "meta.yml").write_text(META)
    monkeypatch.setattr(
        socket.socket, "connect", lambda s, a: (_ for _ in ()).throw(AssertionError(a))
    )
    monkeypatch.setattr(
        dl, "data_dl", lambda *a, **k: (_ for _ in ()).throw(AssertionError("data_dl"))
    )
    # Subject 8 uses the single-ZIP layout; the store tree exercises the
    # extracted-before-archive path without 8 session ZIPs.
    ds = BI2013a()
    # Only the extracted run the deposit holds; default flags ask training+...
    with dl.active_sourcedata_store(store):
        paths = ds.data_path(8)
    assert paths and Path(paths[0]).stat().st_ino == run.stat().st_ino
