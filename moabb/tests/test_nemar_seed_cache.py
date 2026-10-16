"""Seeding a loader's cache from the NEMAR sourcedata store (offline)."""

import json
import os
import socket
from pathlib import Path

import pytest

import moabb.datasets.download as dl
from moabb.datasets.fake import FakeDataset


CACHE_ROOT = (
    "/expanse/lustre/scratch/someone/sourcedata_staging/mne_data/"
    "MNE-Simoes2020-data/BCIAUT_P300"
)


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
    monkeypatch.delenv("MOABB_DOWNLOAD_PROVIDER", raising=False)
    return root


def _forbid_connect(self, address):
    raise AssertionError(f"network touched: {address}")


def _write_store(target_dir, files, cache_root=CACHE_ROOT, subjects=True):
    """A deposit store as nemar_sourcedata_dl leaves it: manifest + files."""
    entries = []
    for name, (subject, content) in files.items():
        entry = {"file": name, "bytes": len(content or "")}
        if subjects:
            entry["subject"] = subject
        entries.append(entry)
        if content is not None:
            path = target_dir / "sourcedata" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    manifest = target_dir / dl.SOURCEDATA_PROVENANCE
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {"files": entries}
    if cache_root is not None:
        record["upstream_cache_root"] = cache_root
    manifest.write_text(json.dumps(record))


def test_cache_root_is_rebased_on_the_local_dataset_path(mne_data):
    root = dl._loader_cache_root({"upstream_cache_root": CACHE_ROOT})
    assert root == mne_data / "MNE-Simoes2020-data" / "BCIAUT_P300"


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"upstream_cache_root": ""},
        {"upstream_cache_root": 3},
        {"upstream_cache_root": "/scratch/somewhere/else"},
        {"upstream_cache_root": "/x/mne_data/MNE-foo-data/../../etc"},
    ],
)
def test_cache_root_unknown_or_unsafe_is_none(mne_data, record):
    assert dl._loader_cache_root(record) is None


def test_seed_links_only_the_subjects_files(mne_data, tmp_path):
    store = mne_data / "NEMAR" / "nm000210"
    _write_store(
        store,
        {
            "SBJ01/S01/Train/trainData.mat": ("1", "one"),
            "SBJ02/S01/Train/trainData.mat": ("2", "two"),
        },
    )
    assert dl.seed_loader_cache(store, 1) == 1

    cache = mne_data / "MNE-Simoes2020-data" / "BCIAUT_P300"
    seeded = cache / "SBJ01" / "S01" / "Train" / "trainData.mat"
    assert seeded.read_text() == "one"
    assert os.path.samefile(
        seeded, store / "sourcedata" / "SBJ01/S01/Train/trainData.mat"
    )
    assert not (cache / "SBJ02").exists()
    # Idempotent, and an existing cache file is never replaced.
    seeded.unlink()
    seeded.write_text("local")
    assert dl.seed_loader_cache(store, 1) == 0
    assert seeded.read_text() == "local"


def test_seed_accepts_alternative_subject_labels(mne_data):
    store = mne_data / "NEMAR" / "nm000210"
    _write_store(store, {"sub-01/a.edf": ("01", "x")})
    assert dl.seed_loader_cache(store, [1, "01"]) == 1


def test_seed_without_subject_field_places_the_whole_store(mne_data):
    store = mne_data / "NEMAR" / "nm000210"
    _write_store(store, {"a.mat": ("1", "a"), "b.mat": ("2", "b")}, subjects=False)
    assert dl.seed_loader_cache(store, 1) == 2


def test_seed_skips_missing_and_escaping_names(mne_data):
    store = mne_data / "NEMAR" / "nm000210"
    _write_store(store, {"SBJ01/missing.mat": ("1", None), "SBJ01/ok.mat": ("1", "ok")})
    record = json.loads((store / dl.SOURCEDATA_PROVENANCE).read_text())
    record["files"].append({"file": "../../../escape.mat", "subject": "1"})
    (store / dl.SOURCEDATA_PROVENANCE).write_text(json.dumps(record))
    # The name resolves to a real file outside the store; its cache destination
    # would land above MNE_DATA.
    (mne_data / "escape.mat").write_text("no")
    assert dl.seed_loader_cache(store, 1) == 1
    assert not (mne_data.parent / "escape.mat").exists()


@pytest.mark.parametrize("manifest", ["absent", "garbage", "no-root"])
def test_seed_is_a_no_op_without_a_usable_manifest(mne_data, manifest):
    store = mne_data / "NEMAR" / "nm000210"
    if manifest == "garbage":
        (store / "sourcedata").mkdir(parents=True)
        (store / dl.SOURCEDATA_PROVENANCE).write_text("{not json")
    elif manifest == "no-root":
        _write_store(store, {"a.mat": ("1", "a")}, cache_root=None)
    assert dl.seed_loader_cache(store, 1) == 0


class _ArchiveLoader(FakeDataset):
    """Mimics Simoes2020: return the extracted tree if present, else fetch a ZIP."""

    nemar_id = "nm000210"
    nemar_seed_cache = True
    downloads = []

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        base = (
            Path(dl.get_dataset_path("Simoes2020", path))
            / "MNE-Simoes2020-data"
            / "BCIAUT_P300"
            / f"SBJ{subject:02d}"
        )
        if base.exists():
            return str(base)
        self.downloads.append(subject)
        dl.data_dl(f"https://zenodo.org/records/1/files/SBJ{subject:02d}.zip", "x")
        return str(base)

    def _get_single_subject_data(self, subject):
        base = Path(self.data_path(subject))
        assert (base / "S01" / "Train" / "trainData.mat").read_text() == "mirrored"
        return super()._get_single_subject_data(subject)


def _archive_dataset(mne_data):
    dataset = _ArchiveLoader(n_subjects=1, paradigm="p300")
    dataset.downloads.clear()
    store = Path(dataset._sourcedata_store()).parent
    _write_store(store, {"SBJ01/S01/Train/trainData.mat": ("1", "mirrored")})
    return dataset


def test_get_data_serves_an_archive_loader_from_the_store(mne_data, monkeypatch):
    """The ZIP is not in the store; the extracted files are, so nothing is fetched."""
    monkeypatch.setattr(socket.socket, "connect", _forbid_connect)
    dataset = _archive_dataset(mne_data)

    dataset.get_data(subjects=[1])

    assert dataset.downloads == []


def test_upstream_provider_does_not_seed(mne_data, monkeypatch):
    dataset = _archive_dataset(mne_data)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    assert dataset._seed_loader_cache(1) == 0
    assert not (mne_data / "MNE-Simoes2020-data").exists()


def test_seed_places_shared_files_the_store_holds(mne_data):
    store = mne_data / "NEMAR" / "nm000329"
    _write_store(
        store,
        {
            "pp1.mat": ("1", "s1"),
            "mnt.mat": (None, "montage"),
            "absent.mat": (None, None),
        },
    )
    assert dl.seed_loader_cache(store, 1) == 2
    cache = mne_data / "MNE-Simoes2020-data" / "BCIAUT_P300"
    assert (cache / "mnt.mat").read_text() == "montage"


_VHDR = (
    "Brain Vision Data Exchange Header File Version 1.0\n"
    "[Common Infos]\nDataFile=run1.eeg\nMarkerFile=run1.vmrk\n"
)


@pytest.mark.parametrize("complete", [True, False])
def test_seed_skips_recordings_whose_headers_name_absent_files(mne_data, complete):
    """nm000195 (Huebner2018) mirrors only the .vhdr headers: seeding them would
    make the loader skip its ZIP and then fail on the missing .eeg."""
    store = mne_data / "NEMAR" / "nm000195"
    files = {"subject01/run1.vhdr": ("1", _VHDR)}
    if complete:
        files["subject01/run1.eeg"] = ("1", "data")
        files["subject01/run1.vmrk"] = ("1", "markers")
    _write_store(store, files)

    seeded = dl.seed_loader_cache(store, 1)

    cache = mne_data / "MNE-Simoes2020-data" / "BCIAUT_P300" / "subject01"
    assert seeded == (3 if complete else 0)
    assert cache.exists() is complete


def _eeglab_set(path, data):
    from scipy.io import savemat

    savemat(path, {"data": data, "nbchan": 1.0})


@pytest.mark.parametrize(
    "data,with_fdt,expected",
    [
        pytest.param("run.fdt", False, 0, id="fdt-referenced-absent"),
        pytest.param("run.fdt", True, 2, id="fdt-referenced-present"),
        pytest.param([[1.0, 2.0]], False, 1, id="samples-embedded"),
    ],
)
def test_seed_checks_eeglab_data_files(mne_data, data, with_fdt, expected):
    """nm000162 (BNCI2025_001) mirrors the .set but not the .fdt it points to."""
    store = mne_data / "NEMAR" / "nm000162"
    files = {"p001/run.set": ("1", None)}
    if with_fdt:
        files["p001/run.fdt"] = ("1", "samples")
    _write_store(store, files)
    (store / "sourcedata" / "p001").mkdir(parents=True, exist_ok=True)
    _eeglab_set(store / "sourcedata" / "p001" / "run.set", data)

    assert dl.seed_loader_cache(store, 1) == expected


def test_seed_does_not_place_an_unreadable_eeglab_header(mne_data):
    store = mne_data / "NEMAR" / "nm000162"
    _write_store(store, {"p001/run.set": ("1", "not a MAT file")})
    assert dl.seed_loader_cache(store, 1) == 0


@pytest.mark.parametrize("header", [None, ".dpa", ".dpo"])
def test_seed_requires_the_curry_header(mne_data, header):
    """nm000316 (IMUMIA2026) mirrors the .cdt samples without their header."""
    store = mne_data / "NEMAR" / "nm000316"
    files = {"task1/0101001.cdt": ("1", "samples")}
    if header:
        files[f"task1/0101001.cdt{header}"] = ("1", "header")
    _write_store(store, files)
    assert dl.seed_loader_cache(store, 1) == (2 if header else 0)


def test_seeding_is_opt_in(mne_data, monkeypatch):
    """Loaders not verified against their deposit keep develop's behaviour."""
    dataset = _archive_dataset(mne_data)
    monkeypatch.setattr(type(dataset), "nemar_seed_cache", False)
    assert dataset._seed_loader_cache(1) == 0


def test_every_seeding_dataset_declares_a_deposit():
    from moabb.datasets.utils import dataset_list

    seeding = [cls for cls in dataset_list if cls.nemar_seed_cache]
    assert seeding
    for cls in seeding:
        assert cls.nemar_id or cls.__subclasses__(), cls


# upstream_cache_root of each deposit's latest release and one subject-1 file,
# as their sourcedata_provenance.json records them, and the directory the
# loader itself reads (relative to MNE_DATA).
_STAGING = "/voyager/ceph/groups/sdp190/bpinto/moabb-nemar/mne_data"
COMPLETED_DEPOSITS = [
    (
        "NETBCI2026",
        f"{_STAGING}/NETBCI2026/MNE-netbci2026-data",
        "sub-01/ses-01/eeg/sub-01_ses-01_task-MotorImageryRest_run-01_eeg.json",
        "MNE-netbci2026-data",
    ),
    (
        "NeBULA2025",
        f"{_STAGING}/NeBULA2025/MNE-nebula2025-data",
        "sub-01/eeg/sub-01_task-free_eeg.json",
        "MNE-nebula2025-data",
    ),
    (
        "Kodera2023",
        f"{_STAGING}/Kodera2023/MNE-kodera2023-data/data",
        "01_12_2020/1z01122020lh1.vmrk",
        "MNE-kodera2023-data/data",
    ),
    (
        "IMUMIA2026",
        f"{_STAGING}/IMUMIA2026/MNE-imumia2026-data/MI_A_Dataset/MI_A_Dataset/Raw_data",
        "task1/0101001.cdt.dpa",
        "MNE-imumia2026-data/MI_A_Dataset/MI_A_Dataset/Raw_data",
    ),
    (
        "MAMEM1",
        "/expanse/lustre/scratch/x/sourcedata_staging/mne_data/MNE-mamem1-data",
        "27901941",
        "MNE-mamem1-data",
    ),
    (
        "MAMEM2",
        "/expanse/lustre/scratch/x/sourcedata_staging/mne_data/MNE-mamem2-data",
        "figshare_filelist_3153409.json",
        "MNE-mamem2-data",
    ),
    (
        "MAMEM3",
        "/expanse/lustre/scratch/x/sourcedata_staging/mne_data/MNE-mamem3-data",
        "figshare_filelist_3413851.json",
        "MNE-mamem3-data",
    ),
    (
        "Kojima2024A",
        "/home/bpinto/mne_data/MNE-Kojima2024A-data",
        "kojima2024_manifest.json",
        "MNE-Kojima2024A-data",
    ),
    (
        "Kojima2024B",
        "/home/bpinto/mne_data/MNE-Kojima2024B-data",
        "kojima2024_manifest.json",
        "MNE-Kojima2024B-data",
    ),
    (
        "Thapa2025",
        f"{_STAGING}/Thapa2025/MNE-Thapa2025-data",
        "sub-01/ses-01/eeg/sub-01_ses-01_task-reachingandgrasping_run-0001_events.tsv",
        "MNE-Thapa2025-data",
    ),
]


@pytest.mark.parametrize(
    "name,cache_root,member,loader_dir",
    COMPLETED_DEPOSITS,
    ids=[case[0] for case in COMPLETED_DEPOSITS],
)
def test_completed_deposits_seed_where_the_loader_reads(
    mne_data, monkeypatch, name, cache_root, member, loader_dir
):
    """Deposits completed for every subject (W42 patch) are served from NEMAR."""
    import moabb.datasets as datasets

    cls = getattr(datasets, name)
    assert cls.nemar_seed_cache
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    dataset = cls()
    store = dl.nemar_store(dataset.code, dataset.nemar_id)
    (store / "sourcedata" / member).parent.mkdir(parents=True)
    (store / "sourcedata" / member).write_text("stored")
    record = {
        "upstream_cache_root": cache_root,
        "files": [{"file": member, "subject": "1"}],
    }
    (store / dl.SOURCEDATA_PROVENANCE).write_text(json.dumps(record))

    assert dataset._seed_loader_cache(1) == 1
    placed = mne_data / loader_dir / member
    assert placed.stat().st_ino == (store / "sourcedata" / member).stat().st_ino
