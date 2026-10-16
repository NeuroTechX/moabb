"""Zhou2016 reads its BIDS release from NEMAR's raw deposit nm000115.

nm000115 has no ``sourcedata/``; its raw BIDS tree holds the files of the
Zenodo ``sub-N.zip`` archives. Only ``nemar.download`` (the call that would
reach data.nemar.org) and the Zenodo access are replaced here; MOABB's provider
routing and the BIDS path search stay real.
"""

from pathlib import Path

import pytest

from moabb.datasets import Zhou2016
from moabb.datasets import download as dl


class _ZenodoReached(Exception):
    pass


def _write_subject(target_dir, subject):
    root = Path(target_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").touch()
    for session in range(3):
        ses = root / f"sub-{subject}" / f"ses-{session}"
        (ses / "eeg").mkdir(parents=True, exist_ok=True)
        (ses / f"sub-{subject}_ses-{session}_scans.tsv").touch()
        for run in range(2):
            stem = f"sub-{subject}_ses-{session}_task-imagery_run-{run}"
            for tail in ("eeg.edf", "events.tsv", "channels.tsv", "eeg.json"):
                (ses / "eeg" / f"{stem}_{tail}").touch()


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = []

    def _download(*, dataset, target_dir, subject, trust_existing, **filters):
        calls.append({"dataset": dataset, "subject": subject, **filters})
        _write_subject(target_dir, subject)

    def _zenodo(self, path=None):
        raise _ZenodoReached

    monkeypatch.setattr(dl.nemar, "download", _download)
    monkeypatch.setattr(Zhou2016, "get_metainfo", _zenodo)
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    monkeypatch.delenv("MOABB_DOWNLOAD_PROVIDER", raising=False)
    return calls


def test_auto_provider_reads_the_nemar_deposit(tmp_path, env):
    paths = Zhou2016().bids_paths(1, path=tmp_path)

    root = tmp_path / "MNE-zhou2016-data" / "nm000115"
    assert env == [{"dataset": "nm000115", "subject": "1", "scope": "raw"}]
    assert len(paths) == 6
    assert {Path(p.root) for p in paths} == {root}
    assert sorted((p.session, p.run) for p in paths) == [
        (s, r) for s in ("0", "1", "2") for r in ("0", "1")
    ]


def test_upstream_provider_never_contacts_nemar(tmp_path, env, monkeypatch):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")

    with pytest.raises(_ZenodoReached):
        Zhou2016().bids_paths(2, path=tmp_path)

    assert env == []


def test_nemar_failure_falls_back_to_zenodo_under_auto(tmp_path, env, monkeypatch):
    def _fail(**kwargs):
        raise OSError("data.nemar.org unreachable")

    monkeypatch.setattr(dl.nemar, "download", _fail)

    with pytest.warns(RuntimeWarning, match="nm000115; using its Zenodo source"):
        with pytest.raises(_ZenodoReached):
            Zhou2016().bids_paths(3, path=tmp_path)

    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    with pytest.raises(dl.NemarDownloadError):
        Zhou2016().bids_paths(3, path=tmp_path)


def test_download_and_get_data_never_ask_for_sourcedata(tmp_path, env, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("sourcedata requested")

    monkeypatch.setattr(dl, "nemar_sourcedata_dl", _boom)

    Zhou2016()._prefetch_nemar_sourcedata([1, 2])
    Zhou2016().download(subject_list=[4], path=tmp_path)

    assert env == [{"dataset": "nm000115", "subject": "4", "scope": "raw"}]
