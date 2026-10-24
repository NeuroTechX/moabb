"""Thielen2021 and Kaneshiro2015 load from the NEMAR store without upstream (offline).

Thielen2021's deposit (nm000214) keeps ``offline/sub-NN/block_k/<file>``, four
segments of the URL path, while the store lookup probed three; Kaneshiro2015
took ``S<n>.mat`` from the store (nm000263) but still sent a Range request to
stacks.stanford.edu on every load to detect truncated downloads.
"""

import os
import socket

import pytest

import moabb.datasets.download as dl
from moabb.datasets import Kaneshiro2015, Thielen2021
from moabb.datasets.thielen2021 import SESSIONS


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


class _Resp:
    headers = {}


def _store(tmp_path, names):
    store = tmp_path / "store"
    for name in names:
        path = store / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return store


def test_lookup_matches_four_trailing_segments(tmp_path):
    store = _store(tmp_path, ["offline/sub-01/block_1/trainlabels.mat"])
    url = (
        "https://public.data.ru.nl/dcc/DSC_2018.00122_448_v3/sourcedata/"
        "offline/sub-01/block_1/trainlabels.mat"
    )
    with dl.active_sourcedata_store(store):
        assert dl._store_lookup(url) == str(
            store / "offline/sub-01/block_1/trainlabels.mat"
        )
        # The longest tail wins: another block's file is never taken.
        assert dl._store_lookup(url.replace("block_1", "block_2")) is None


def test_thielen2021_loads_a_subject_from_the_store(mne_data, tmp_path, monkeypatch):
    subject = 30
    sub = f"sub-{subject:02d}"
    names = []
    for block in range(1, 6):
        names += [
            f"offline/{sub}/block_{block}/{sub}_{SESSIONS[subject - 1]}_block_{block}_main_eeg.gdf",
            f"offline/{sub}/block_{block}/trainlabels.mat",
        ]
    names += ["resources/mgold_61_6521_flip_balanced_20.mat", "resources/nt_cap8.loc"]
    store = _store(tmp_path, names)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    with dl.active_sourcedata_store(store):
        paths = Thielen2021().data_path(subject)
    assert len(paths) == len(names)
    for path, name in zip(paths, names):
        assert os.stat(path).st_ino == (store / name).stat().st_ino


def test_kaneshiro2015_skips_the_range_probe_for_a_store_copy(
    mne_data, tmp_path, monkeypatch
):
    import moabb.datasets.kaneshiro2015 as kaneshiro

    store = _store(tmp_path, ["S3.mat"])
    probed = []
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    # data_path swallows errors of the probe: record it instead of raising.
    monkeypatch.setattr(
        kaneshiro.requests, "get", lambda url, **kw: probed.append(url) or _Resp()
    )
    with dl.active_sourcedata_store(store):
        path = Kaneshiro2015().data_path(3)
    assert os.stat(path).st_ino == (store / "S3.mat").stat().st_ino
    assert probed == []


def test_kaneshiro2015_still_probes_a_file_not_from_the_store(mne_data, monkeypatch):
    import moabb.datasets.kaneshiro2015 as kaneshiro

    local = mne_data / f"MNE-{kaneshiro._SIGN}-data" / "S3.mat"
    local.parent.mkdir(parents=True)
    local.write_text("cached upstream")
    probed = []
    monkeypatch.setattr(
        kaneshiro.requests, "get", lambda url, **kw: probed.append(url) or _Resp()
    )
    assert Kaneshiro2015().data_path(3) == str(local)
    assert probed == [f"{kaneshiro._BASE_URL}/S3.mat"]
