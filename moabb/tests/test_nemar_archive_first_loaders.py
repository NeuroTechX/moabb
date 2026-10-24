"""Archive-first loaders take the extracted subject from the NEMAR store (offline).

Leelakittisin2025, MOVING2024, Perdikis2018 and Shin2022 called ``data_dl`` on
their archive before looking at what it extracts to. Their deposits (nm000330,
nm000293, nm000307, nm000315) keep the extracted files, so the archive URL never
matched the store and every load fetched the archive from Zenodo/figshare.
"""

import os
import socket
from pathlib import Path

import pytest

import moabb.datasets.download as dl
from moabb.datasets import MOVING2024, Leelakittisin2025, Perdikis2018, Shin2022


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


_SHIN_RUNS = [
    f"{kind}/{kind}{n}.dat"
    for kind, ns in (("BW", (30, 60)), ("CV", (200, 250)), ("NT", (0, 24)))
    for n in ns
]

# (dataset, subject, deposit files of that subject -- names as in the real
#  manifests --, files of another subject also in the store, and what
#  data_path returns, relative to the store root)
CASES = [
    (
        Leelakittisin2025,
        2,
        ["S02/v1_raw_s2/S02_S1.mat", "S02/v1_raw_s2/S02_S2.mat"],
        ["S01/v1_raw_s1/S01_S1.mat"],
        ["S02/v1_raw_s2/S02_S1.mat", "S02/v1_raw_s2/S02_S2.mat"],
    ),
    (
        MOVING2024,
        2,
        ["edf/20231220093707_Subj_02_bci_32_gesture.edf"],
        ["edf/20231219170321_Subj_01_bci_32_gesture.edf"],
        ["edf/20231220093707_Subj_02_bci_32_gesture.edf"],
    ),
    (
        Perdikis2018,
        2,
        [
            "AN14VE/20160922/AN14VE.20160922.101010.offline.mi.mi_bhbf.gdf",
            "AN14VE/BrainRunnersLogs/v1225/result.csv",
        ],
        ["MA25VE/20160707/MA25VE.20160707.101010.offline.mi.mi_bhbf.gdf"],
        "AN14VE",
    ),
    (
        Shin2022,
        2,
        [f"S02_20220303/S02_20220303_LR_S1001/{run}" for run in _SHIN_RUNS],
        ["S01_20220302/S01_20220302_LR_S1001/BW/BW30.dat"],
        "S02_20220303/S02_20220303_LR_S1001",
    ),
]
IDS = [case[0].__name__ for case in CASES]


def _store(tmp_path, names):
    store = tmp_path / "store"
    for name in names:
        path = store / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return store


@pytest.mark.parametrize("cls,subject,files,others,returned", CASES, ids=IDS)
def test_reads_the_extracted_subject_from_the_store(
    mne_data, tmp_path, monkeypatch, cls, subject, files, others, returned
):
    store = _store(tmp_path, files + others)
    monkeypatch.setattr(dl, "data_dl", _no_download)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    dataset = cls()
    with dl.active_sourcedata_store(store):
        got = dataset.data_path(subject)

    got = got if isinstance(got, list) else [got]
    if isinstance(returned, list):
        assert [os.path.basename(p) for p in got] == [
            os.path.basename(name) for name in returned
        ]
    else:
        assert len(got) == 1 and Path(got[0]).as_posix().endswith(returned)
    for path in got:
        local = (
            [path]
            if os.path.isfile(path)
            else [os.path.join(root, f) for root, _d, fs in os.walk(path) for f in fs]
        )
        assert local
        for name in local:
            posix = Path(name).as_posix()
            member = posix[posix.index("/" + files[0].split("/")[0] + "/") + 1 :]
            assert os.stat(name).st_ino == (store / member).stat().st_ino
    # A second load finds the same files without a download; Shin2022 can
    # only tell a complete LIVE directory from the archive or the store.
    with dl.active_sourcedata_store(store if cls is Shin2022 else None):
        again = dataset.data_path(subject)
    assert (again if isinstance(again, list) else [again]) == got


@pytest.mark.parametrize("cls,subject,files,others,returned", CASES, ids=IDS)
def test_downloads_the_archive_when_the_store_lacks_the_subject(
    mne_data, tmp_path, monkeypatch, cls, subject, files, others, returned
):
    store = _store(tmp_path, others)
    monkeypatch.setattr(dl, "data_dl", _no_download)
    with dl.active_sourcedata_store(store), pytest.raises(_Downloaded):
        cls().data_path(subject)
    with pytest.raises(_Downloaded):
        cls().data_path(subject)


def test_store_glob_lists_matching_store_names(tmp_path):
    store = _store(tmp_path, ["S01_20220302/a.dat", "SIMULATED_S01_x/b.dat"])
    assert dl.store_glob("S01_*") == []
    with dl.active_sourcedata_store(store):
        assert dl.store_glob("S01_*") == ["S01_20220302"]
