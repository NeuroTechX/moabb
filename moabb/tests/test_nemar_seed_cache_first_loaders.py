"""Cache-first loaders read a subject seeded from its NEMAR deposit (offline).

GuttmannFlury2025_SSVEP, Nguyen2017_L, Zhang2025 and Zheng2020 check their own
cache before downloading, and their deposits (nm000310, nm000252, nm000211,
nm000205) are snapshots of that cache holding every file the loaders read for
every subject. With ``nemar_seed_cache`` the subject's files are linked there,
so ``data_path`` returns them without contacting the upstream host.
"""

import json
import os
import socket

import pytest

import moabb.datasets.download as dl


_STAGING = "/scratch/someone/sourcedata_staging/mne_data"


@pytest.fixture
def mne_data(tmp_path, monkeypatch):
    """A fresh MNE_DATA with MNE's config redirected to a temporary home."""
    fake_home = tmp_path / "mne-home"
    fake_home.mkdir()
    monkeypatch.setenv("_MNE_FAKE_HOME_DIR", str(fake_home))
    for key in tuple(os.environ):
        if key.startswith("MNE_DATASETS_"):
            monkeypatch.delenv(key)
    root = tmp_path / "mne_data"
    root.mkdir()
    monkeypatch.setenv("MNE_DATA", str(root))
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    return root


def _no_download(*args, **kwargs):
    raise AssertionError(f"download attempted: {args[:1]}")


def _forbid_connect(self, address):
    raise AssertionError(f"network touched: {address}")


# (class, subject, upstream_cache_root recorded by the deposit, files the
#  deposit files under that subject -- names as in the real manifests --, and
#  what data_path returns, relative to that cache root)
CASES = [
    (
        "GuttmannFlury2025_SSVEP",
        2,
        f"{_STAGING}/MNE-guttmannflury2025-data/SSVEP",
        [
            f"S02/Sess0{s}/SSVEP02{s}{suffix}"
            for s in (1, 2, 3)
            for suffix in (".bdf", "_annotations.json", "_sync.csv")
        ],
        "S02",
    ),
    (
        "Nguyen2017_L",
        2,
        f"{_STAGING}/MNE-nguyen2017l-data/Long_words",
        ["sub-02.mat"],
        "sub-02.mat",
    ),
    ("Zhang2025", 15, f"{_STAGING}/MNE-Zhang2025-data", ["S15.mat"], "S15.mat"),
    # nm000205 records no subject: every group is filed for every subject.
    (
        "Zheng2020",
        14,
        f"{_STAGING}/MNE-Zheng2020-data",
        [f"G{g}/G{g}D{d}.mat" for g in range(1, 8) for d in (1, 2)],
        "G7",
    ),
]


@pytest.mark.parametrize(
    "name,subject,cache_root,files,returned", CASES, ids=[case[0] for case in CASES]
)
def test_seeded_subject_loads_without_download(
    mne_data, monkeypatch, name, subject, cache_root, files, returned
):
    import moabb.datasets as datasets

    cls = getattr(datasets, name)
    assert cls.nemar_seed_cache
    dataset = cls()
    store = dl.nemar_store(dataset.code, dataset.nemar_id)
    entries = []
    for member in files:
        path = store / "sourcedata" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(member)
        entry = {"file": member}
        if name != "Zheng2020":
            entry["subject"] = subject
        entries.append(entry)
    record = {"upstream_cache_root": cache_root, "files": entries}
    (store / dl.SOURCEDATA_PROVENANCE).write_text(json.dumps(record))

    assert dataset._seed_loader_cache(subject) == len(files)
    monkeypatch.setattr(dl, "data_dl", _no_download)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)

    cache = dl._loader_cache_root(record)
    assert dataset.data_path(subject) == str(cache / returned)
    for member in files:
        if member == returned or member.startswith(returned + "/"):
            linked = (cache / member).stat().st_ino
            assert linked == (store / "sourcedata" / member).stat().st_ino
