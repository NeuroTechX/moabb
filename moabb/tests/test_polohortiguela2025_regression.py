"""Regression tests for PoloHortiguela2025 archive cache collisions."""

import zipfile
from types import SimpleNamespace

import mne
import numpy as np

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


def test_static_and_motion_phase_codes_both_create_target_annotations(monkeypatch):
    """The motion archive shifts the task-code tens digit from 1 to 2."""

    n_samples = 40
    data = np.zeros((35, n_samples))

    for rest_code, mi_code in ((211, 311), (221, 321)):
        task = np.zeros(n_samples, dtype=np.uint16)
        task[5:15] = rest_code
        task[20:30] = mi_code
        session = SimpleNamespace(data_EEG=data, task_EEG=task)
        monkeypatch.setattr(
            "moabb.datasets.polohortiguela2025.loadmat",
            lambda *args, session=session, **kwargs: {"session": session},
        )

        raw = PoloHortiguela2025()._make_raw("unused.mat")
        events, event_id = mne.events_from_annotations(
            raw,
            event_id={"rest": 1, "motor_imagery": 2},
            verbose=False,
        )

        assert event_id == {"rest": 1, "motor_imagery": 2}
        assert events[:, 0].tolist() == [5, 20]
        assert events[:, 2].tolist() == [1, 2]
