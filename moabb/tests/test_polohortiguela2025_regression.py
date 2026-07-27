"""Regression tests for PoloHortiguela2025 archive cache collisions."""

import zipfile

from moabb.datasets.polohortiguela2025 import PoloHortiguela2025


def _make_archive(path, folder):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{folder}/run01.mat", b"test")


def test_shared_content_cache_refreshes_for_each_condition(tmp_path, monkeypatch):
    static_archive = tmp_path / "static-content"
    motion_archive = tmp_path / "motion-content"
    _make_archive(static_archive, "B02_S1_STATIC")
    _make_archive(motion_archive, "B02_S1_MOTION")
    calls = []

    def data_dl(url, code, path, force_update):
        del code, path
        calls.append((url, force_update))
        if force_update:
            return str(motion_archive)
        return str(static_archive)

    monkeypatch.setattr(
        "moabb.datasets.polohortiguela2025.dl.data_dl",
        data_dl,
    )

    paths = PoloHortiguela2025().data_path(2, path=tmp_path)

    assert [path.rsplit("/", 1)[-1] for path in paths] == [
        "B02_S1_STATIC",
        "B02_S1_MOTION",
    ]
    assert (tmp_path / "B02_S1_STATIC" / "run01.mat").is_file()
    assert (tmp_path / "B02_S1_MOTION" / "run01.mat").is_file()
    assert sum(force for _, force in calls) == 1
