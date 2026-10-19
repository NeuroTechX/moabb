"""Per-subject-ZIP loaders take the extracted subject from the store (offline).

Tavakolan2017, Liu2025, Zhou2020 and Wu2020 fetch a per-subject ZIP whose
extracted tree is exactly what their NEMAR deposits keep, so the archive URL
never matches the store and the loaders downloaded gigabytes per subject.
"""

import os
import socket
from pathlib import Path

import pytest

import moabb.datasets.download as dl
from moabb.datasets import Liu2025, Tavakolan2017, Wu2020, Zhou2020


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


def _no_download(*args, **kwargs):
    raise AssertionError(f"data_dl called: {args[:1]}")


CASES = [
    pytest.param(
        Tavakolan2017,
        [f"P01/P01_Se{s:02d}.DAT" for s in (1, 2, 3, 4)],
        id="Tavakolan2017",
    ),
    pytest.param(
        Liu2025,
        ["sub-01/sub-01_task-test_eeg.vhdr"],
        id="Liu2025",
    ),
    pytest.param(Zhou2020, ["S01/session/file.npz"], id="Zhou2020"),
    pytest.param(Wu2020, ["subject1/subject1.dat"], id="Wu2020"),
]


@pytest.mark.parametrize("cls,members", CASES)
def test_subject_zip_loader_reads_extracted_subject_from_store(
    mne_data, tmp_path, monkeypatch, cls, members
):
    store = tmp_path / "store"
    for member in members:
        (store / member).parent.mkdir(parents=True, exist_ok=True)
        (store / member).write_text(member)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    monkeypatch.setattr(dl, "data_dl", _no_download)

    with dl.active_sourcedata_store(store):
        returned = cls().data_path(1)

    stored = {os.stat(store / m).st_ino for m in members}
    found = [p for p in Path(returned).rglob("*") if p.is_file()]
    assert found and all(p.stat().st_ino in stored for p in found)
