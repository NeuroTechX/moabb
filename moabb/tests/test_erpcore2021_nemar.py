"""ErpCore2021 reads its recordings from NEMAR's raw BIDS deposit nm000132.

The deposit has no ``sourcedata/``: it is the ERP CORE BIDS dataset itself, one
dataset whose components are the ``task-`` entity, with the ERN and LRP
components sharing ``task-flankers``. These tests replace only
``nemar.download`` (the library call that would reach data.nemar.org) and the
OSF downloader; MOABB's provider routing and BIDS path building stay real.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from moabb.datasets import download as dl
from moabb.datasets.erpcore2021 import (
    ErpCore2021_ERN,
    ErpCore2021_LRP,
    ErpCore2021_N170,
    ErpCore2021_P3,
)


_ROOT_FILES = ("dataset_description.json", "participants.tsv")


def _write_nemar_files(target_dir, subject, task=None, suffix=None):
    """Materialize what nemar.download would select for these filters."""
    eeg = Path(target_dir) / f"sub-{subject}" / "eeg"
    eeg.mkdir(parents=True, exist_ok=True)
    for name in _ROOT_FILES:
        (Path(target_dir) / name).touch()
    if task is not None:
        for tail in ("eeg.set", "eeg.fdt", "eeg.json", "events.tsv", "channels.tsv"):
            (eeg / f"sub-{subject}_task-{task}_{tail}").touch()
    for name in suffix or ():
        ext = "tsv" if name == "electrodes" else "json"
        (eeg / f"sub-{subject}_{name}.{ext}").touch()


@pytest.fixture
def fake_nemar(monkeypatch, tmp_path):
    """Record nemar.download calls and write the files they would fetch."""
    calls = []

    def _download(*, dataset, target_dir, subject, trust_existing, **filters):
        calls.append({"dataset": dataset, "subject": subject, **filters})
        _write_nemar_files(
            target_dir, subject, filters.get("task"), filters.get("suffix")
        )

    monkeypatch.setattr(dl.nemar, "download", _download)
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    monkeypatch.delenv("MOABB_DOWNLOAD_PROVIDER", raising=False)
    return calls


@pytest.fixture
def no_osf(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("OSF downloader reached")

    monkeypatch.setattr(dl, "download_if_missing", _boom)


@pytest.mark.parametrize(
    ("cls", "task"),
    [
        (ErpCore2021_N170, "N170"),
        (ErpCore2021_P3, "P3"),
        (ErpCore2021_ERN, "flankers"),
        (ErpCore2021_LRP, "flankers"),
    ],
)
def test_auto_provider_reads_the_nemar_bids_recording(
    tmp_path, fake_nemar, no_osf, cls, task
):
    bids_path = cls().data_path(1, path=tmp_path)[0]

    root = tmp_path / "MNE-erpcore2021-data" / "nm000132"
    assert Path(bids_path.root) == root
    assert (bids_path.subject, bids_path.task) == ("001", task)
    assert (root / "sub-001/eeg" / f"sub-001_task-{task}_eeg.set").is_file()
    # The task query and the subject-level sidecars a task filter would drop.
    assert fake_nemar == [
        {"dataset": "nm000132", "subject": "001", "task": task},
        {
            "dataset": "nm000132",
            "subject": "001",
            "suffix": ["electrodes", "coordsystem"],
        },
    ]
    assert (root / "sub-001/eeg/sub-001_electrodes.tsv").is_file()


def test_ern_and_lrp_share_one_download_and_reuse_it_offline(
    tmp_path, fake_nemar, no_osf
):
    first = ErpCore2021_ERN().data_path(2, path=tmp_path)[0]
    calls_after_first = len(fake_nemar)
    second = ErpCore2021_LRP().data_path(2, path=tmp_path)[0]
    again = ErpCore2021_ERN().data_path(2, path=tmp_path)[0]

    assert first.fpath == second.fpath == again.fpath
    # A recording already on disk costs no request to NEMAR.
    assert len(fake_nemar) == calls_after_first == 2


def test_upstream_provider_never_contacts_nemar(tmp_path, fake_nemar, monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    roots = []
    monkeypatch.setattr(
        ErpCore2021_N170,
        "download_by_subject",
        lambda self, subject, path=None, force_update=False: (
            roots.append(subject) or tmp_path / "MNE-erpcore2021-data"
        ),
    )

    bids_path = ErpCore2021_N170().data_path(1, path=tmp_path)[0]

    assert fake_nemar == []
    assert roots == [1]
    assert bids_path.task == "N170"


def test_auto_provider_reuses_a_local_upstream_copy(tmp_path, fake_nemar, no_osf):
    upstream = tmp_path / "MNE-erpcore2021-data"
    _write_nemar_files(upstream, "001", task="ERN")
    (upstream / "erpcore_manifest.csv").write_text(
        "component,participant_id,local_path,url\n"
    )

    bids_path = ErpCore2021_ERN().data_path(1, path=tmp_path)[0]

    assert fake_nemar == []
    assert Path(bids_path.root) == upstream
    assert bids_path.task == "ERN"


def test_nemar_failure_falls_back_to_osf_under_auto(tmp_path, monkeypatch):
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    monkeypatch.delenv("MOABB_DOWNLOAD_PROVIDER", raising=False)

    def _fail(**kwargs):
        raise OSError("data.nemar.org unreachable")

    monkeypatch.setattr(dl.nemar, "download", _fail)
    fallback = []
    monkeypatch.setattr(
        ErpCore2021_P3,
        "download_by_subject",
        lambda self, subject, path=None, force_update=False: (
            fallback.append(subject) or tmp_path / "MNE-erpcore2021-data"
        ),
    )

    with pytest.warns(RuntimeWarning, match="from OSF instead"):
        bids_path = ErpCore2021_P3().data_path(3, path=tmp_path)[0]

    assert fallback == [3]
    assert bids_path.task == "P3"

    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    with pytest.raises(dl.NemarDownloadError):
        ErpCore2021_P3().data_path(4, path=tmp_path)
    assert fallback == [3]


def test_get_data_does_not_prefetch_sourcedata(tmp_path, fake_nemar, monkeypatch):
    """nm000132 has no sourcedata/: get_data must not ask NEMAR for it."""

    def _boom(*args, **kwargs):
        raise AssertionError("sourcedata requested")

    monkeypatch.setattr(dl, "nemar_sourcedata_dl", _boom)
    ErpCore2021_N170()._prefetch_nemar_sourcedata([1, 2])


def test_download_fetches_the_recordings_it_reads(tmp_path, fake_nemar, no_osf):
    ErpCore2021_LRP().download(subject_list=[1, 2], path=tmp_path)

    root = tmp_path / "MNE-erpcore2021-data" / "nm000132"
    for subject in ("001", "002"):
        assert (root / f"sub-{subject}/eeg/sub-{subject}_task-flankers_eeg.fdt").is_file()


def test_n170_labels_ignore_the_event_type_column_nemar_adds(tmp_path):
    """nm000132 v1.1 adds ``event_type`` to the N170 events; labels stay."""
    rows = pd.DataFrame(
        {
            "onset": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "duration": [0.0, 0.3, 0.3, 0.3, 0.3, 0.0],
            "sample": [1024, 2048, 3072, 4096, 5120, 6144],
            "trial_type": ["STATUS", "stimulus", "stimulus", "stimulus", "stimulus"]
            + ["response"],
            "value": [1, 12, 79, 120, 170, 201],
        }
    )
    with_type = rows.assign(
        event_type=["n/a", "face", "car", "scrambled_face", "scrambled_car", "correct"]
    )

    def labels(df):
        raw = mne.io.RawArray(
            np.zeros((1, 8000)), mne.create_info(1, 1024.0), verbose=False
        )
        dataset = ErpCore2021_N170()
        path = tmp_path / "events.tsv"
        df.to_csv(path, sep="\t", index=False)
        annotations = dataset.handle_events_reading(path, raw).annotations
        return list(annotations.onset), list(annotations.description)

    assert labels(rows) == labels(with_type)
    assert labels(rows)[1] == ["Target", "NonTarget", "NonTarget", "NonTarget"]
