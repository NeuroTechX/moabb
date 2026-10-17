"""A sourcedata file may belong to several subjects (offline).

Zheng2020's upstream files ``G{g}/G{g}D{1,2}.mat`` each hold both members of
pair ``g`` (MOABB subjects ``2g-1`` and ``2g``). Its deposit (nm000205) files
them with ``"subjects": ["2g-1", "2g"]`` and no ``"subject"`` key, so that MOABB
releases that only know ``"subject"`` keep fetching the whole deposit, while
this version prefetches, checks and seeds only the pair's two files.
"""

import glob
import json
import os
import socket

import pytest

import moabb.datasets.download as dl


_STAGING = "/scratch/someone/sourcedata_staging/mne_data"
_ZHENG = [
    {"file": f"G{g}/G{g}D{d}.mat", "subjects": [str(2 * g - 1), str(2 * g)]}
    for g in range(1, 8)
    for d in (1, 2)
]


def _store(tmp_path, entries, present=(), **extra):
    target = tmp_path / "NEMAR" / "nm000205"
    provenance = target / dl.SOURCEDATA_PROVENANCE
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({"dataset": "nm000205", "files": entries, **extra}))
    for name in present:
        (target / "sourcedata" / name).parent.mkdir(parents=True, exist_ok=True)
        (target / "sourcedata" / name).write_text(name)
    return target


@pytest.mark.parametrize("subject", [13, 14, "13", [13, "S13"]])
def test_pair_file_belongs_to_both_members(tmp_path, subject):
    target = _store(tmp_path, _ZHENG)

    patterns = dl._sourcedata_files_for_subject(target, "nm000205", subject, False)

    assert patterns == ["sourcedata/" + glob.escape(n) for n in ("G7/G7D1.mat", "G7/G7D2.mat")]


def test_list_in_subject_field_matches_too(tmp_path):
    entries = [{"file": e["file"], "subject": e["subjects"]} for e in _ZHENG]

    assert dl._manifest_subject_files({"files": entries}, 1) == ["G1/G1D1.mat", "G1/G1D2.mat"]
    assert dl._manifest_shared_files({"files": entries}) == []


def test_multi_subject_entries_are_not_shared(tmp_path):
    entries = _ZHENG + [
        {"file": "montage.loc", "subject": None},
        {"file": "README"},
        {"file": "empty.txt", "subjects": []},
    ]
    record = {"files": entries}

    assert dl._manifest_shared_files(record) == ["montage.loc", "README", "empty.txt"]
    assert dl._manifest_subject_files(record, 2) == ["G1/G1D1.mat", "G1/G1D2.mat"]


def test_subject_outside_every_list_is_unknown(tmp_path):
    target = _store(tmp_path, _ZHENG)

    with pytest.raises(dl.NemarDownloadError, match=r"Known subjects: \['1', '10',"):
        dl._sourcedata_files_for_subject(target, "nm000205", 15, False)


def test_is_local_per_subject(tmp_path):
    target = _store(tmp_path, _ZHENG, present=("G7/G7D1.mat", "G7/G7D2.mat"))

    assert dl.nemar_sourcedata_is_local(target, 13)
    assert dl.nemar_sourcedata_is_local(target, 14)
    assert not dl.nemar_sourcedata_is_local(target, 1)


def test_scalar_subject_manifests_unchanged(tmp_path):
    entries = [
        {"file": "pp1.mat", "subject": "1"},
        {"file": "pp12.mat", "subject": "12"},
        {"file": "mnt.mat", "subject": None},
    ]
    record = {"files": entries}

    assert dl._manifest_subject_files(record, 1) == ["pp1.mat"]
    assert dl._manifest_subject_files(record, [12]) == ["pp12.mat"]
    assert dl._manifest_shared_files(record) == ["mnt.mat"]
    assert dl._manifest_subject_files({"files": [{"file": "a"}]}, 1) is None


def test_zheng2020_seeds_only_its_pair(tmp_path, monkeypatch):
    import moabb.datasets as datasets

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

    dataset = datasets.Zheng2020()
    store = dl.nemar_store(dataset.code, dataset.nemar_id)
    for entry in _ZHENG:
        path = store / "sourcedata" / entry["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["file"])
    record = {"upstream_cache_root": f"{_STAGING}/MNE-Zheng2020-data", "files": _ZHENG}
    (store / dl.SOURCEDATA_PROVENANCE).write_text(json.dumps(record))

    assert dataset._seed_loader_cache(14) == 2

    def _no_download(*args, **kwargs):
        raise AssertionError(f"download attempted: {args[:1]}")

    def _forbid_connect(self, address):
        raise AssertionError(f"network touched: {address}")

    monkeypatch.setattr(dl, "data_dl", _no_download)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    cache = dl._loader_cache_root(record)
    assert dataset.data_path(14) == str(cache / "G7")
    assert sorted(p.name for p in (cache).rglob("*.mat")) == ["G7D1.mat", "G7D2.mat"]
