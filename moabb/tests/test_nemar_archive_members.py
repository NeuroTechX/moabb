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
    Huebner2017,
    Huebner2018,
    Liu2024,
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
    pytest.param(
        Liu2024().data_path,
        ["sub-01/eeg/sub-01_task-motor-imagery_eeg.edf"],
        id="Liu2024",
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


def _brainvision_run(directory, stem, *, companions=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.vhdr").write_text(
        f"DataFile={stem}.eeg\nMarkerFile={stem}.vmrk\n", encoding="latin-1"
    )
    if companions:
        (directory / f"{stem}.eeg").write_bytes(b"\0" * 8)
        (directory / f"{stem}.vmrk").write_text("markers")


@pytest.mark.parametrize("dataset", [Huebner2017, Huebner2018], ids=lambda d: d.__name__)
def test_huebner_reads_the_extracted_subject_from_the_store(
    mne_data, tmp_path, monkeypatch, dataset
):
    """nm000199/nm000195 keep subjectXX/ as extracted from subjectXX.zip."""
    store = tmp_path / "store"
    stems = [f"matrixSpeller_Block{b}_Run{r}" for b in (1, 2) for r in (1, 2)]
    for stem in stems:
        _brainvision_run(store / "subject01", stem)
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    monkeypatch.setattr(dl, "data_dl", _no_download)

    with dl.active_sourcedata_store(store):
        returned = dataset().data_path(1)

    assert [Path(p).name for p in returned] == [f"{stem}.vhdr" for stem in stems]
    for path in map(Path, returned):
        for suffix in (".vhdr", ".eeg", ".vmrk"):
            linked = path.with_suffix(suffix)
            stored = store / "subject01" / linked.name
            assert linked.stat().st_ino == stored.stat().st_ino


def test_huebner_downloads_when_the_store_has_headers_only(
    mne_data, tmp_path, monkeypatch
):
    """The deposits before v1.0.4 held the .vhdr without .eeg/.vmrk."""
    store = tmp_path / "store"
    _brainvision_run(store / "subject01", "matrixSpeller_Block1_Run1", companions=False)
    calls = []

    def fake_data_dl(url, sign, *args, **kwargs):
        calls.append(url)
        raise RuntimeError("stop after the download decision")

    monkeypatch.setattr(dl, "data_dl", fake_data_dl)
    with dl.active_sourcedata_store(store), pytest.raises(RuntimeError, match="stop"):
        Huebner2017().data_path(1)

    assert calls == ["https://zenodo.org/record/5831826/files/subject01.zip"]
    cache = dl.data_dl_destination(calls[0], "llp").parent
    assert not (cache / "subject01").exists()


def test_huebner_reuses_an_extracted_subject_without_the_zip(mne_data, monkeypatch):
    url = "https://zenodo.org/record/5831826/files/subject01.zip"
    subject_dir = dl.data_dl_destination(url, "llp").parent / "subject01"
    _brainvision_run(subject_dir, "matrixSpeller_Block1_Run1")
    monkeypatch.setattr(dl, "data_dl", _no_download)

    returned = Huebner2017().data_path(1)

    assert returned == [str(subject_dir / "matrixSpeller_Block1_Run1.vhdr")]


def test_liu2024_extracts_only_absent_members(mne_data, tmp_path, monkeypatch):
    """A subject linked from the store is not overwritten through its link."""
    import zipfile

    store = tmp_path / "store"
    member = "sub-01/eeg/sub-01_task-motor-imagery_eeg.edf"
    (store / member).parent.mkdir(parents=True)
    (store / member).write_text("from NEMAR")
    archive = tmp_path / "edffile.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for sub in ("01", "02"):
            zf.writestr(
                f"edffile/sub-{sub}/eeg/sub-{sub}_task-motor-imagery_eeg.edf",
                f"from figshare {sub}",
            )
    monkeypatch.setattr(dl, "data_dl", lambda *args, **kwargs: str(archive))

    with dl.active_sourcedata_store(store):
        Liu2024().data_path(1)
        (second,) = Liu2024().data_path(2)

    assert Path(second).read_text() == "from figshare 02"
    assert (store / member).read_text() == "from NEMAR"


def test_fill_tree_from_store_needs_an_active_store(tmp_path):
    (tmp_path / "subject01").mkdir()
    (tmp_path / "subject01" / "a.vhdr").write_text("x")
    assert not dl.fill_tree_from_store(tmp_path, "subject01")
