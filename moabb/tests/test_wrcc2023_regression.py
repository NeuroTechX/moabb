"""Offline scientific and transport contracts for the WRCC/MIMED loaders."""

import mne
import numpy as np
import pytest
from scipy.io import savemat

from moabb.datasets import WRCC2023_MI_A, WRCC2023_MI_B, WRCC2023_MI_C, Wirawan2024
from moabb.datasets.preprocessing import SetRawAnnotations


WRCC = [WRCC2023_MI_A, WRCC2023_MI_B, WRCC2023_MI_C]


@pytest.mark.parametrize("cls", WRCC)
def test_stored_windows_preserve_samples_units_and_first_last(tmp_path, monkeypatch, cls):
    data = np.random.default_rng(42).normal(0, 10e-6, (3, 59, 4000))
    path = tmp_path / "subject.mat"
    savemat(path, {"data": data, "label": [1, 2, 3], "fs": 1000})
    ds = cls()
    monkeypatch.setattr(ds, "data_path", lambda subject: str(path))
    runs = ds._get_single_subject_data(1)["0"]
    assert len(runs) == 3
    for i, raw in enumerate(runs.values()):
        assert raw.n_times == 4000
        np.testing.assert_array_equal(raw.get_data(), data[i])
        raw.info["bads"] = ["Fp1"]
        raw = SetRawAnnotations(ds.event_id, ds.interval).transform(raw)
        assert raw.info["bads"] == ["Fp1"]
        events, _ = mne.events_from_annotations(raw, event_id=ds.event_id)
        epochs = mne.Epochs(
            raw,
            events,
            ds.event_id,
            tmin=0,
            tmax=ds.interval[1],
            baseline=None,
            preload=True,
            on_missing="ignore",
        )
        assert len(epochs) == 1
        assert epochs.get_data().shape[-1] == 4000
        assert events[0, 2] == i + 1
        # Filtering operates on one stored window, never its neighbour.
        filtered = raw.copy().filter(8, 30, verbose=False)
        expected = mne.filter.filter_data(data[i], 1000, 8, 30, verbose=False)
        np.testing.assert_allclose(filtered.get_data(), expected)


@pytest.mark.parametrize("cls", WRCC)
@pytest.mark.parametrize("labels", [[1], [1, 4], [1, 2, 3], [1, 2.5]])
def test_invalid_trial_labels_fail(tmp_path, cls, labels):
    path = tmp_path / "bad.mat"
    savemat(path, {"data": np.zeros((2, 59, 4000)), "label": labels, "fs": 1000})
    with pytest.raises(ValueError, match="labels"):
        cls._mat_to_raw(path)


@pytest.mark.parametrize("cls", WRCC)
def test_download_flags_and_subject_validation(tmp_path, monkeypatch, cls):
    calls = []
    monkeypatch.setattr(
        "moabb.datasets.download.data_dl", lambda *args: calls.append(args) or "file.mat"
    )
    ds = cls()
    ds.data_path(1, path=tmp_path, force_update=True, update_path=False, verbose=False)
    assert calls[0][1:] == (ds.code, tmp_path, True, False)
    with pytest.raises(ValueError):
        ds.data_path(0)


def test_mimed_sessions_runs_units_and_missing_blocks(tmp_path, monkeypatch):
    ds = Wirawan2024()
    files = []
    for scenario in range(3):
        joined = np.empty((1, 4), dtype=object)
        for i in range(4):
            joined[0, i] = np.full((1152, 14), 4200 + i + scenario)
        path = tmp_path / f"scenario{scenario}.mat"
        savemat(path, {"joined_data": joined, "Fs": 128})
        files.append(str(path))
    monkeypatch.setattr(ds, "data_path", lambda subject: files)
    sessions = ds._get_single_subject_data(1)
    assert list(sessions) == ["0", "1"]
    for day, runs in sessions.items():
        assert len(runs) == 6
        for run, raw in runs.items():
            expected = (4200 + int(day) * 2 + int(run) % 2 + int(run) // 2) * 1e-6
            np.testing.assert_allclose(raw.get_data(), expected)
            assert raw.annotations.onset[0] == 3
    savemat(files[0], {"joined_data": joined[:, :3], "Fs": 128})
    with pytest.raises(ValueError, match="four MIMED"):
        ds._get_single_subject_data(1)


def test_mimed_download_flags(tmp_path, monkeypatch):
    import zipfile

    from moabb.datasets.wirawan2024 import WIRAWAN2024_MI_URL

    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Motor Imagery/fixture.txt", "fixture")
    calls = []
    monkeypatch.setattr(
        "moabb.datasets.download.data_dl",
        lambda *args: calls.append(args) or str(archive),
    )
    paths = Wirawan2024().data_path(1, tmp_path, True, False, False)
    assert len(paths) == 3
    assert calls == [(WIRAWAN2024_MI_URL, "Wirawan2024", tmp_path, True, False)]


def test_mimed_duplicate_scenarios_rejected(monkeypatch):
    ds = Wirawan2024()
    monkeypatch.setattr(ds, "data_path", lambda subject: ["same.mat"] * 3)
    with pytest.raises(ValueError, match="Duplicate"):
        ds._get_single_subject_data(1)
