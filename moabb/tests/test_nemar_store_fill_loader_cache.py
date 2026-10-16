"""Loaders whose cache the seeding cannot reach take their files from the store.

MIMED2024 (Mendeley ZIP, nm000296), Weibo2014 (Dataverse ZIPs, nm000146) and
Shin2017A/B (TU Berlin ZIPs, nm000267/nm000268) check their cache before
downloading, but either the cache test (MIMED2024: "is the extracted tree
there?") or the cache directory (``MNE-weibo-2014``; the licence gate of
Shin2017) rules out ``nemar_seed_cache``. They now place the subject's files
from the active NEMAR store themselves (offline tests).
"""

import importlib
import os
import socket
import zipfile

import pytest

import moabb.datasets.download as dl
from moabb.datasets import MIMED2024, Shin2017A, Shin2017B, Weibo2014
from moabb.datasets.utils import download_and_extract_subject_zip


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


class _Downloaded(Exception):
    pass


def _no_download(*args, **kwargs):
    raise _Downloaded(args[:1])


def _forbid_connect(self, address):
    raise AssertionError(f"network touched: {address}")


def _store(tmp_path, names):
    store = tmp_path / "store"
    for name in names:
        path = store / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return store


_MIMED = [
    f"{scenario}/P{{:02d}}.mat"
    for scenario in (
        "Left Hand Up-Down Imagine",
        "Right Hand Up-Down Imagine",
        "Stand Up-Down Imagine",
    )
]
_SHIN = [
    "subject {:02d}/with occular artifact/cnt.mat",
    "subject {:02d}/with occular artifact/mrk.mat",
]

# (make dataset, subject, deposit names of that subject, of another subject)
CASES = [
    (MIMED2024, 2, [n.format(2) for n in _MIMED], [n.format(1) for n in _MIMED]),
    (Weibo2014, 6, ["subject_6.mat"], ["subject_1.mat"]),
    (
        lambda: Shin2017A(accept=True),
        29,
        [n.format(29) for n in _SHIN],
        [n.format(1) for n in _SHIN],
    ),
    (
        lambda: Shin2017B(accept=True),
        29,
        [n.format(29) for n in _SHIN],
        [n.format(1) for n in _SHIN],
    ),
]
IDS = ["MIMED2024", "Weibo2014", "Shin2017A", "Shin2017B"]


@pytest.mark.parametrize("make,subject,files,others", CASES, ids=IDS)
def test_reads_the_subject_from_the_store(
    mne_data, tmp_path, monkeypatch, make, subject, files, others
):
    import moabb.datasets.bbci_eeg_fnirs as shin

    weibo = importlib.import_module("moabb.datasets.Weibo2014")

    store = _store(tmp_path, files + others)
    monkeypatch.setattr(dl, "data_dl", _no_download)
    monkeypatch.setattr(weibo, "retrieve", _no_download)
    monkeypatch.setattr(shin, "retrieve", _no_download)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    dataset = make()
    with dl.active_sourcedata_store(store):
        got = dataset.data_path(subject)
    got = got if isinstance(got, list) else [got]
    assert len(got) == len(files)
    for path, name in zip(got, files):
        assert path.endswith(name.replace("/", os.sep))
        assert os.stat(path).st_ino == (store / name).stat().st_ino
    # Only this subject's files were placed.
    for name in others:
        assert not any(os.path.exists(p.replace(n, name)) for p, n in zip(got, files))


@pytest.mark.parametrize("make,subject,files,others", CASES, ids=IDS)
def test_downloads_when_the_store_lacks_the_subject(
    mne_data, tmp_path, monkeypatch, make, subject, files, others
):
    """A store holding only another subject must not hide the download."""
    import moabb.datasets.bbci_eeg_fnirs as shin
    import moabb.datasets.mimed2024 as mimed

    weibo = importlib.import_module("moabb.datasets.Weibo2014")

    store = _store(tmp_path, others)
    monkeypatch.setattr(mimed, "download_and_extract_subject_zip", _no_download)
    monkeypatch.setattr(weibo, "retrieve", _no_download)
    monkeypatch.setattr(shin, "retrieve", _no_download)
    dataset = make()
    # The other subject is already in the cache, from the store.
    with dl.active_sourcedata_store(store):
        dataset.data_path(int("".join(c for c in others[0] if c.isdigit())[:2]))
        with pytest.raises(_Downloaded):
            dataset.data_path(subject)


def test_shin2017_store_needs_the_licence(mne_data, tmp_path, monkeypatch):
    store = _store(tmp_path, [n.format(1) for n in _SHIN])
    with (
        dl.active_sourcedata_store(store),
        pytest.raises(AttributeError, match="licence"),
    ):
        Shin2017A().data_path(1)


def test_extract_skips_existing_members(mne_data, tmp_path, monkeypatch):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("d/kept.mat", "from archive")
        zf.writestr("d/new.mat", "from archive")
    out = tmp_path / "out"
    (out / "d").mkdir(parents=True)
    linked = out / "d" / "kept.mat"
    linked.write_text("from store")
    monkeypatch.setattr(dl, "data_dl", lambda *a, **k: str(archive))
    download_and_extract_subject_zip("u", "s", out, skip_existing=True)
    assert linked.read_text() == "from store"
    assert (out / "d" / "new.mat").read_text() == "from archive"
