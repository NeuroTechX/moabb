"""Offline source-format regression tests (no dataset downloads)."""

import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import mne
import numpy as np
import pytest

from moabb.datasets import Pan2023, Pan2025, PoloHortiguela2025
from moabb.datasets.pan2023 import PAN2023_CHANNELS
from moabb.datasets.preprocessing import SetRawAnnotations


@pytest.mark.parametrize(
    "cls,samples,offset", [(Pan2023, 1750, 750), (Pan2025, 1375, 375)]
)
def test_stored_windows(tmp_path, cls, samples, offset):
    path = tmp_path / "fixture.mat"
    data = np.ones((28, samples, 2)) * 12
    data[:, :, 1] = -7
    if cls is Pan2023:
        with h5py.File(path, "w") as f:
            f["data"] = data.transpose(2, 1, 0)
            f["label"] = [1, 2]
            f["fs"] = [250]
        raw = cls._mat_to_raw(path)
    else:
        obj = {
            "data": data,
            "label": [1, 2],
            "Info": SimpleNamespace(fs=250, period=[-1.5, 4], chaninfo=PAN2023_CHANNELS),
        }
        with patch("moabb.datasets.pan2025.sio.loadmat", return_value=obj):
            raw = cls._mat_to_raw(path)
    np.testing.assert_allclose(raw.get_data(picks="eeg")[:, 0], 12e-6)
    np.testing.assert_allclose(raw.get_data(picks="eeg")[:, -1], -7e-6)
    ds = cls()
    raw = SetRawAnnotations(ds.event_id, ds.interval).transform(raw)
    events = mne.find_events(raw, initial_event=True, verbose=False)
    np.testing.assert_array_equal(events[:, 0], [offset, samples + offset])
    epochs = mne.Epochs(
        raw,
        events,
        ds.event_id,
        tmin=0,
        tmax=ds.interval[1],
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert epochs.get_data().shape == (2, 29, 1000)
    # The non-rejecting boundary must sit exactly at the stored-trial join.
    edges = raw.annotations[raw.annotations.description == "EDGE boundary"]
    np.testing.assert_allclose(edges.onset, [samples / 250])


@pytest.mark.parametrize("cls", [Pan2023, Pan2025])
def test_download_flags(cls):
    with patch("moabb.datasets.download.data_dl", return_value="fixture") as download:
        assert cls().data_path(1, path="custom", force_update=True, verbose=False) == [
            "fixture",
            "fixture",
        ]
        assert all(
            call.args[2:] == ("custom", True, False) for call in download.call_args_list
        )
    with pytest.raises(ValueError):
        cls().data_path(0)


@pytest.mark.parametrize("rest,imagery", [(211, 311), (221, 321)])
def test_polo_units_and_both_conditions(rest, imagery):
    obj = SimpleNamespace(
        data_EEG=np.ones((35, 2500)) * 10,
        task_EEG=np.r_[np.full(1250, rest), np.full(1250, imagery)],
    )
    with patch(
        "moabb.datasets.polohortiguela2025.loadmat", return_value={"session": obj}
    ):
        raw = PoloHortiguela2025()._make_raw("fixture")
    np.testing.assert_allclose(raw.get_data()[:32], 10e-6)
    np.testing.assert_allclose(raw.get_data()[32:], 10)
    assert list(raw.annotations.description) == ["rest", "motor_imagery"]
    np.testing.assert_allclose(raw.annotations.onset, [0, 5])
    assert raw.get_channel_types().count("eog") == 4


def test_polo_transport_flags(tmp_path):
    archives = []
    for condition in ("STATIC", "MOTION"):
        archives.append(tmp_path / f"B01_S1_{condition}.zip")
        with zipfile.ZipFile(archives[-1], "w") as archive:
            archive.writestr(f"B01_S1_{condition}/run.mat", b"synthetic")
    with patch("moabb.datasets.download.data_dl", side_effect=archives) as download:
        folders = PoloHortiguela2025().data_path(1, "custom", True, verbose=False)
    # Named downloads: every Zenodo "/content" URL would otherwise share one cache file.
    assert folders == [str(p.with_suffix("")) for p in archives]
    assert [c.kwargs for c in download.call_args_list] == [
        {"fname": p.name} for p in archives
    ]
    assert all(c.args[2:] == ("custom", True, False) for c in download.call_args_list)
