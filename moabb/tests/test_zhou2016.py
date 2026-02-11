import importlib
from pathlib import Path
from zipfile import ZipFile

from moabb.datasets.Zhou2016 import Zhou2016


def _build_minimal_subject_zip(zip_path: Path):
    """Create a minimal archive with one EEG run and its events file."""
    with ZipFile(zip_path, "w") as zf:
        base = "sub-1/ses-0/eeg"
        zf.writestr(
            f"{base}/sub-1_ses-0_task-imagery_run-0_desc-test_eeg.edf",
            "dummy-edf-content",
        )
        zf.writestr(
            f"{base}/sub-1_ses-0_task-imagery_run-0_desc-test_events.tsv",
            "onset\tduration\ttrial_type\n",
        )


def test_subject_has_downloaded_data_detects_incomplete_dir(tmp_path):
    subject_dir = tmp_path / "sub-1"
    subject_dir.mkdir()
    (subject_dir / "sub-1_desc-lockfile.json").write_text("{}")
    assert not Zhou2016._subject_has_downloaded_data(subject_dir)

    eeg_dir = subject_dir / "ses-0" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "run_eeg.edf").write_text("dummy")
    assert not Zhou2016._subject_has_downloaded_data(subject_dir)

    (eeg_dir / "run_events.tsv").write_text("onset\tduration\ttrial_type\n")
    assert Zhou2016._subject_has_downloaded_data(subject_dir)


def test_download_subject_repairs_incomplete_subject_dir(tmp_path, monkeypatch):
    dataset = Zhou2016()
    dataset_path = tmp_path / "MNE-BIDS-zhou2016"
    incomplete_subject_dir = dataset_path / "sub-1"
    incomplete_subject_dir.mkdir(parents=True)
    (incomplete_subject_dir / "sub-1_desc-lockfile.json").write_text("{}")

    source_zip = tmp_path / "source-sub-1.zip"
    _build_minimal_subject_zip(source_zip)

    monkeypatch.setattr(
        dataset,
        "get_metainfo",
        lambda path=None: {
            "files": [
                {
                    "key": "sub-1.zip",
                    "links": {"self": "https://example.org/sub-1.zip"},
                }
            ]
        },
    )

    def fake_download_if_missing(file_path, url, warn_missing=True, verbose=True):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(source_zip.read_bytes())
        return str(file_path)

    zhou_module = importlib.import_module("moabb.datasets.Zhou2016")
    monkeypatch.setattr(zhou_module, "download_if_missing", fake_download_if_missing)

    root = dataset._download_subject(
        subject=1,
        path=str(tmp_path),
        force_update=False,
        update_path=False,
        verbose=False,
    )

    assert Path(root) == dataset_path
    assert dataset._subject_has_downloaded_data(incomplete_subject_dir)
    assert any(incomplete_subject_dir.rglob("*_eeg.edf"))
    assert any(incomplete_subject_dir.rglob("*_events.tsv"))
