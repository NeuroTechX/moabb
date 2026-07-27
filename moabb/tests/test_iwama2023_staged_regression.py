"""Regression tests for Iwama2023 locally staged sparse sessions."""

from unittest.mock import patch

from moabb.datasets.iwama2023 import Iwama2023


def _stage_edf(root, subject, session):
    subject_name = f"sub-{subject:03d}"
    session_name = f"ses-{session:02d}"
    path = root / subject_name / session_name / "eeg"
    path.mkdir(parents=True)
    prefix = f"{subject_name}_{session_name}_task-smrbmi_"
    for suffix in ("eeg.edf", "events.tsv", "channels.tsv", "electrodes.tsv", "eeg.json"):
        (path / f"{prefix}{suffix}").touch()


def test_iwama2023_staged_subject_avoids_remote_probe_and_discovers_sessions(
    tmp_path,
):
    """Subjects 20 and 30 use staged EDFs without probing absent S3 sessions."""
    data_root = tmp_path / "MNE-iwama2023-data"
    for session in range(1, 9):
        _stage_edf(data_root, 20, session)
        _stage_edf(data_root, 30, session)
    _stage_edf(data_root, 30, 16)

    dataset = Iwama2023()
    with patch(
        "moabb.datasets.iwama2023.get_dataset_path", return_value=str(tmp_path)
    ), patch.object(
        dataset,
        "_download_root_files",
        side_effect=AssertionError("staged data must not trigger remote probes"),
    ):
        assert dataset._download_subject(20, None, False, None, None) == str(data_root)
        assert dataset._download_subject(30, None, False, None, None) == str(data_root)

    sessions20 = {path.parent.parent.name for path in dataset._staged_subject_edfs(data_root, 20)}
    sessions30 = {path.parent.parent.name for path in dataset._staged_subject_edfs(data_root, 30)}
    assert sessions20 == {f"ses-{session:02d}" for session in range(1, 9)}
    assert sessions30 == {f"ses-{session:02d}" for session in range(1, 9)} | {"ses-16"}


def test_iwama2023_absent_subject_keeps_download_fallback(tmp_path):
    """No local EDF leaves the existing bounded download path available."""
    dataset = Iwama2023()
    data_root = tmp_path / "MNE-iwama2023-data"
    with patch(
        "moabb.datasets.iwama2023.get_dataset_path", return_value=str(tmp_path)
    ), patch.object(dataset, "_download_root_files") as root_download, patch.object(
        dataset, "_download_file", return_value=False
    ) as download_file:
        dataset._download_subject(1, None, False, None, None)

    root_download.assert_called_once_with(data_root, False)
    assert download_file.called


def test_iwama2023_partial_staged_session_keeps_download_fallback(tmp_path):
    """An interrupted session is not mistaken for a complete staged subject."""
    dataset = Iwama2023()
    data_root = tmp_path / "MNE-iwama2023-data"
    subject_dir = data_root / "sub-001" / "ses-01" / "eeg"
    subject_dir.mkdir(parents=True)
    (subject_dir / "sub-001_ses-01_task-smrbmi_eeg.edf").touch()

    with patch(
        "moabb.datasets.iwama2023.get_dataset_path", return_value=str(tmp_path)
    ), patch.object(dataset, "_download_root_files") as root_download, patch.object(
        dataset, "_download_file", return_value=False
    ) as download_file:
        dataset._download_subject(1, None, False, None, None)

    root_download.assert_called_once_with(data_root, False)
    assert download_file.called
