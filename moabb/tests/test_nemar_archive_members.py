"""Archive loaders take the extracted members from the NEMAR store (offline).

NEMAR deposits mirror what an archive extracts to, not the archive itself, so
the URL lookup in ``data_dl`` never matches the ``.zip`` these loaders fetch.
"""

import os
import socket
from pathlib import Path

import pytest

import moabb.datasets.download as dl
from moabb.datasets import (
    BI2012,
    BNCI2019_001,
    BI2014a,
    BI2014b,
    BI2015a,
    BI2015b,
    Hinss2021,
)
from moabb.datasets.bnci.bnci_2025 import _load_data_001_2025


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


def _bnci2025(subject):
    return _load_data_001_2025(subject, only_filenames=True)


# Member paths exactly as the deposits list them for subject 1 (provenance of
# nm000260, nm000244, nm000215, nm000216, nm000217, nm000149, nm000162,
# nm000343). BI2015a's odd directory is real: the loader derives it with
# str.strip, and the deposit was taken from that cache.
CASES = [
    pytest.param(BI2012().data_path, ["subject_01/training.mat"], id="BI2012"),
    pytest.param(BI2014a().data_path, ["subject_01/subject_01.mat"], id="BI2014a"),
    pytest.param(BI2014b().data_path, ["group_01/group_01_sujet_01.mat"], id="BI2014b"),
    pytest.param(
        BI2015a().data_path,
        [f"subject_01_masubject_01/subject_01_session_0{s}.mat" for s in (1, 2, 3)],
        id="BI2015a",
    ),
    pytest.param(
        BI2015b().data_path,
        [f"group_01/group_01_s{s}.mat" for s in (1, 2, 3, 4)],
        id="BI2015b",
    ),
    pytest.param(
        BNCI2019_001().data_path,
        [f"P01 Run {r}.gdf" for r in BNCI2019_001._MOVEMENT_RUNS],
        id="BNCI2019_001",
    ),
    pytest.param(
        _bnci2025,
        ["p001/p001v2-trialblocks.set", "p001/p001v2-trialblocks.fdt"],
        id="BNCI2025_001",
    ),
    pytest.param(
        Hinss2021().data_path,
        [
            f"P01/S{s}/eeg/alldata_sbj01_sess{s}_{t}.set"
            for s in (1, 2)
            for t in ("RS", "MATBeasy", "MATBmed", "MATBdiff")
        ],
        id="Hinss2021",
    ),
]


@pytest.mark.parametrize("data_path,members", CASES)
def test_archive_loader_reads_members_from_the_store(
    mne_data, tmp_path, monkeypatch, data_path, members
):
    store = tmp_path / "store"
    for member in members:
        (store / member).parent.mkdir(parents=True, exist_ok=True)
        (store / member).write_text(member)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    monkeypatch.setattr(dl, "data_dl", _no_download)

    with dl.active_sourcedata_store(store):
        returned = data_path(1)

    returned = returned if isinstance(returned, list) else [returned]
    assert returned
    stored = {os.stat(store / m).st_ino for m in members}
    for path in returned:
        path = Path(path)
        if path.is_dir():  # Hinss2021 returns the subject directory
            assert any(p.stat().st_ino in stored for p in path.rglob("*.set"))
        else:
            assert path.stat().st_ino in stored


def test_bnci2025_needs_the_fdt_too(mne_data, tmp_path, monkeypatch):
    """nm000162 mirrors the .set without its .fdt: fetch the ZIP as before."""
    store = tmp_path / "store"
    (store / "p001").mkdir(parents=True)
    (store / "p001" / "p001v2-trialblocks.set").write_text("header")
    monkeypatch.setattr(dl, "data_dl", _no_download)
    with (
        dl.active_sourcedata_store(store),
        pytest.raises(AssertionError, match="data_dl"),
    ):
        _bnci2025(1)


def test_archive_loader_downloads_when_the_store_lacks_a_member(
    mne_data, tmp_path, monkeypatch
):
    """All or nothing: one absent member means the archive is fetched as before."""
    store = tmp_path / "store"
    (store / "group_01").mkdir(parents=True)
    for s in (1, 2, 3):  # s4 is missing
        (store / "group_01" / f"group_01_s{s}.mat").write_text("x")
    calls = []

    def fake_data_dl(url, sign, *args, **kwargs):
        calls.append(url)
        raise RuntimeError("stop after the download decision")

    monkeypatch.setattr(dl, "data_dl", fake_data_dl)
    with dl.active_sourcedata_store(store), pytest.raises(RuntimeError, match="stop"):
        BI2015b().data_path(1)

    assert calls and calls[0].endswith("group_01_mat.zip")
    # Nothing was linked into the cache, so no half-extracted group is left.
    cache = dl.data_dl_destination(calls[0], "BRAININVADERS2015B").parent
    assert not (cache / "group_01").exists()


def test_fill_from_store_without_an_active_store(tmp_path):
    present = tmp_path / "a.mat"
    present.write_text("x")
    assert dl.fill_from_store(tmp_path, [present])
    assert not dl.fill_from_store(tmp_path, [tmp_path / "missing.mat"])


def test_fill_from_store_refuses_files_outside_base_dir(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    with dl.active_sourcedata_store(store):
        assert not dl.fill_from_store(tmp_path / "base", [tmp_path / "elsewhere.mat"])


@pytest.mark.parametrize(
    "url,fname",
    [
        ("https://zenodo.org/record/2649069/files/subject_01.zip", None),
        ("https://lampx.tugraz.at/~bci/database/001-2019/P01.zip", None),
        ("https://example.org/api/datafile/123", "sub-01/run.mat"),
    ],
)
def test_data_dl_destination_matches_data_dl(mne_data, monkeypatch, url, fname):
    def fake_retrieve(url, known_hash, fname, path, **kwargs):
        return str(Path(path) / fname)

    monkeypatch.setattr(dl, "retrieve", fake_retrieve)
    returned = dl.data_dl(url, "SIGN", fname=fname)
    assert Path(returned) == dl.data_dl_destination(url, "SIGN", fname=fname)
