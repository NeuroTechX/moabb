"""Offline signal, event and transport contracts for the archive B loaders."""

import importlib
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from moabb.datasets import Leeuwis2021, MartinezPeon2024, OpenViBE, PardoGarcia2026
from moabb.datasets.leeuwis2021 import LEEUWIS2021_EEG_CHANNELS
from moabb.datasets.preprocessing import SetRawAnnotations


@pytest.mark.parametrize(
    "cls,count",
    [(Leeuwis2021, 4), (MartinezPeon2024, 6), (OpenViBE, 1), (PardoGarcia2026, 6)],
)
def test_transport_flags(cls, count, tmp_path, monkeypatch):
    calls = []
    module = importlib.import_module(cls.__module__)

    def transport(url, sign, path=None, force_update=False, verbose=None):
        calls.append((url, sign, path, force_update, verbose))
        return str(tmp_path / url.rsplit("/", 1)[-1])

    monkeypatch.setattr(module.dl, "data_dl", transport)
    dataset = cls()
    dataset.data_path(1, path=tmp_path, force_update=True, verbose="ERROR")
    assert len(calls) == count
    assert all(call[2:] == (tmp_path, True, "ERROR") for call in calls)
    with pytest.raises(ValueError, match="Invalid subject"):
        dataset.data_path(0)


def test_leeuwis_first_last_si_and_discontinuities(tmp_path):
    frame = pd.DataFrame(
        {ch: np.r_[np.zeros(2000), np.ones(2000) * 20] for ch in LEEUWIS2021_EEG_CHANNELS}
    )
    frame["trial"] = np.repeat([1, 2], 2000)
    frame["class"] = np.repeat([-1, 1], 2000)
    frame["TimeStamp"] = np.tile(np.arange(2000) / 250 - 3, 2)
    path = tmp_path / "run.csv"
    frame.to_csv(path, index=False)
    dataset = Leeuwis2021()
    raw = dataset._csv_to_raw(path)
    np.testing.assert_allclose(raw.get_data(picks="eeg")[:, -1], 20e-6)
    transformed = SetRawAnnotations(dataset.event_id, dataset.interval).transform(raw)
    assert "EDGE boundary" in transformed.annotations.description
    events, _ = mne.events_from_annotations(transformed, event_id=dataset.event_id)
    assert events[:, 0].tolist() == [750, 2750]
    epochs = mne.Epochs(
        transformed,
        events,
        dataset.event_id,
        tmin=0,
        tmax=dataset.interval[1],
        baseline=None,
        preload=True,
    )
    assert len(epochs) == 2
    assert epochs.get_data().shape[-1] == 1250
    filtered = transformed.copy().pick("eeg").filter(1, 30, verbose=False)
    independent = np.concatenate(
        [
            raw.copy()
            .pick("eeg")
            .crop(i * 8, (i + 1) * 8 - 1 / 250)
            .filter(1, 30, verbose=False)
            .get_data()
            for i in range(2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(filtered.get_data(), independent, atol=1e-15)


def test_martinez_units_and_first_last_cue(tmp_path):
    path = tmp_path / "record.txt"
    np.savetxt(path, np.full((5120, 20), 15.0))
    raw = MartinezPeon2024()._read_run(path, "level_70")
    np.testing.assert_allclose(raw.get_data(), 15e-6)
    np.testing.assert_allclose(raw.annotations.onset, [2.9, 10.9, 18.9, 26.9, 34.9])
    assert raw.annotations.description.tolist() == ["level_70"] * 5


@pytest.mark.parametrize(
    "subject,expected", [(1, ["0pre", "1post"]), (2, ["0pre"]), (11, ["0pre"])]
)
def test_pardo_missing_post_is_not_duplicated(subject, expected, monkeypatch):
    dataset = PardoGarcia2026()
    monkeypatch.setattr(
        dataset, "data_path", lambda subject: [str(i) for i in range(len(expected))]
    )
    monkeypatch.setattr(dataset, "_load_raw", lambda path: path)
    sessions = dataset._get_single_subject_data(subject)
    assert list(sessions) == expected
    assert [session["0"] for session in sessions.values()] == [
        Path(str(i)) for i in range(len(expected))
    ]
