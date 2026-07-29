import zipfile
from unittest.mock import patch

from moabb.datasets.sitstand2026 import SitStand2026


def test_nested_subject_archive_is_extracted_only_once(tmp_path, monkeypatch):
    """The real Zenodo layout must be reusable by offline compute jobs."""
    archive = tmp_path / "v1_raw_S01.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        for session in (1, 2):
            zip_file.writestr(
                f"v1_raw_s1/S01_S{session}.mat",
                b"placeholder",
            )

    extract_calls = 0
    original_extractall = zipfile.ZipFile.extractall

    def tracked_extractall(self, *args, **kwargs):
        nonlocal extract_calls
        extract_calls += 1
        return original_extractall(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "extractall", tracked_extractall)
    dataset = SitStand2026()

    with patch(
        "moabb.datasets.sitstand2026.dl.data_dl",
        return_value=str(archive),
    ):
        first = dataset.data_path(1)
        second = dataset.data_path(1)

    expected = [
        str(tmp_path / "S01" / "v1_raw_s1" / f"S01_S{session}.mat")
        for session in (1, 2)
    ]
    assert first == expected
    assert second == expected
    assert extract_calls == 1
