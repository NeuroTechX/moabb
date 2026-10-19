"""Seeded NEMAR deposits serve the remaining archive loaders (offline).

Chen2017SingleFlicker, EPFLP300, Nguyen2017_V, Vagaja2023, Jeong2020,
Stieger2021, Yi2025, Yang2025, Schrag2026Pediatric and Brandl2020 downloaded
their upstream archive (or file list) on every load even though their NEMAR
deposits hold the extracted files under the subject's provenance label.
With ``nemar_seed_cache`` the subject's files are linked into the loader's
own cache first, and the loaders that fetched before checking their cache
(EPFLP300, Schrag2026Pediatric, Stieger2021) now check it first.
GuttmannFlury2025's deposit nests subjects 5 and 9 one level deeper
(``S05/S05/...``); its loader accepts that tree too.
"""

import json
import os
import socket

import pytest

import moabb.datasets.download as dl


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
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    return root


def _no_download(*args, **kwargs):
    raise AssertionError(f"download attempted: {args[:1]}")


def _forbid_connect(self, address):
    raise AssertionError(f"network touched: {address}")


_STAGING = "/scratch/someone/sourcedata_staging/mne_data"

# (class, subject, upstream_cache_root, seeded files, expected data_path hit)
CASES = [
    (
        "Chen2017SingleFlicker",
        1,
        f"{_STAGING}/MNE-chen2017singleflicker-data/3.raw_data",
        ["1.training_data/sub_1_1.xdf"],
        "sub_1_1.xdf",
    ),
    (
        "EPFLP300",
        1,
        f"{_STAGING}/MNE-epflp300-data/groups/m/mm/mmspg/www/BCI/p300",
        ["subject1/session1/eeg_epochs.mat"],
        "eeg_epochs.mat",
    ),
    (
        "Nguyen2017_V",
        2,
        f"{_STAGING}/MNE-nguyen2017v-data/Vowels",
        ["sub-02.mat"],
        "sub-02.mat",
    ),
    (
        "Vagaja2023",
        17,
        f"{_STAGING}/MNE-vagaja2023-data",
        ["Control/SUB17/SUB17_MI.vhdr"],
        "SUB17",
    ),
    ("Jeong2020", 1, f"{_STAGING}/MNE-jeong2020-data", ["sub1/session1.vhdr"], "sub1"),
    (
        "Stieger2021",
        1,
        f"{_STAGING}/MNE-Stieger2021-data",
        ["S1_Session_1.mat"],
        "S1_Session_1.mat",
    ),
    (
        "Yi2025",
        1,
        f"{_STAGING}/MNE-yi2025-data/FineMI",
        ["subject1/EEG/block1.cnt"],
        "subject1",
    ),
    (
        "Yang2025",
        1,
        f"{_STAGING}/MNE-yang2025-data/WBCIC_SHU Motor Imagery dataset",
        ["2C dataset/sub-001/ses-01/eeg/data.bdf"],
        "MNE-yang2025-data",
    ),
    (
        "Schrag2026Pediatric",
        1,
        f"{_STAGING}/MNE-schrag2026-data",
        ["P001/EEG/sub-P001_eeg.xdf"],
        "P001",
    ),
]


@pytest.mark.parametrize(
    "name,subject,cache_root,files,hit", CASES, ids=[c[0] for c in CASES]
)
def test_seeded_subject_loads_without_download(
    mne_data, monkeypatch, name, subject, cache_root, files, hit
):
    import moabb.datasets as datasets

    cls = getattr(datasets, name)
    assert cls.nemar_seed_cache, name
    dataset = cls()
    store = dl.nemar_store(dataset.code, dataset.nemar_id)
    entries = []
    for member in files:
        path = store / "sourcedata" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(member)
        entries.append({"file": member, "subject": subject})
    record = {"upstream_cache_root": cache_root, "files": entries}
    (store / dl.SOURCEDATA_PROVENANCE).write_text(json.dumps(record))

    assert dataset._seed_loader_cache(subject) == len(files)
    monkeypatch.setattr(dl, "data_dl", _no_download)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    if name == "Stieger2021":
        monkeypatch.setattr(dl, "fs_get_file_list", _no_download, raising=False)

    returned = dataset.data_path(subject)
    flat = (
        [str(p) for p in returned]
        if isinstance(returned, (list, tuple))
        else [str(returned)]
    )
    assert any(hit in p for p in flat), (name, returned)


def test_guttmann_p300_accepts_double_nested_seed(mne_data, monkeypatch):
    import moabb.datasets as datasets

    dataset = datasets.GuttmannFlury2025_P300()
    store = dl.nemar_store(dataset.code, dataset.nemar_id)
    member = "S05/S05/Sess01/P3004L051.bdf"
    path = store / "sourcedata" / member
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(member)
    record = {
        "upstream_cache_root": f"{_STAGING}/MNE-guttmannflury2025-data/P3004L",
        "files": [{"file": member, "subject": 5}],
    }
    (store / dl.SOURCEDATA_PROVENANCE).write_text(json.dumps(record))

    assert dataset._seed_loader_cache(5) == 1
    monkeypatch.setattr(dl, "data_dl", _no_download)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    returned = dataset.data_path(5)
    assert returned.endswith(os.path.join("S05", "S05"))
