"""Offline tests for the Ma2022 loader on a small synthetic BIDS tree."""

import json
from unittest.mock import Mock

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from moabb.datasets import Ma2022
from moabb.datasets.download import NemarDownloadError
from moabb.datasets.ma2022 import MA2022_CH_NAMES
from moabb.paradigms import MotorImagery


N_SESSIONS = 5


@pytest.fixture
def bids_root(tmp_path):
    """Five 12 s sessions for subject 3; F3 is flagged bad in session 2."""
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Synthetic Ma2022", "BIDSVersion": "1.7.0"})
    )
    (root / "task-motorimagery_eeg.json").write_text(
        json.dumps(
            {
                "TaskName": "motorimagery",
                "SamplingFrequency": 250,
                "PowerLineFrequency": 50,
                "EEGReference": "M1",
            }
        )
    )
    (root / "participants.tsv").write_text("participant_id\tsex\tage\nsub-003\tF\t23\n")
    signal = 20e-6 * np.sin(2 * np.pi * 10 * np.arange(3000) / 250)
    for session in range(1, N_SESSIONS + 1):
        folder = root / "sub-003" / f"ses-{session:02d}" / "eeg"
        folder.mkdir(parents=True)
        stem = f"sub-003_ses-{session:02d}_task-motorimagery"
        raw = mne.io.RawArray(
            np.tile(signal, (32, 1)),
            mne.create_info(MA2022_CH_NAMES, 250, "eeg"),
            verbose=False,
        )
        mne.export.export_raw(folder / f"{stem}_eeg.edf", raw, verbose=False)
        rows = "".join(
            f"{ch}\tEEG\tµV\t{'bad' if ch == 'F3' and session == 2 else 'good'}\n"
            for ch in MA2022_CH_NAMES
        )
        (folder / f"{stem}_channels.tsv").write_text("name\ttype\tunits\tstatus\n" + rows)
        (folder / f"{stem}_events.tsv").write_text(
            "onset\tduration\ttrial_type\tvalue\tsample\n"
            "0\t4\tright_hand\t2\t0\n"
            "4\t4\tleft_hand\t1\t1000\n"
            "8\t4\tright_hand\t2\t2000\n"
        )
    return root


@pytest.fixture
def dataset(bids_root, monkeypatch):
    ds = Ma2022(subjects=[3])
    monkeypatch.setattr(ds, "_download_nemar", lambda *a, **k: str(bids_root))
    return ds


def test_metadata():
    ds = Ma2022()
    assert ds.code == "Ma-edf2022"
    assert ds.nemar_id == "nm000288"
    assert ds.interval == [0, 3.996]
    assert ds.METADATA.file_format == "EDF"


def test_download_requests_raw_bids_for_the_subject(bids_root, monkeypatch):
    download = Mock(return_value=str(bids_root))
    monkeypatch.setattr("moabb.datasets.base.nemar_dl", download)
    ds = Ma2022(subjects=[3])
    paths = ds.data_path(3)
    assert download.call_args.args == ("nm000288", "Ma-edf2022")
    assert download.call_args.kwargs["subject"] == "003"
    assert download.call_args.kwargs["scope"] == "raw"
    assert [p.parent.parent.name for p in paths] == [
        f"ses-{i:02d}" for i in range(1, N_SESSIONS + 1)
    ]
    with pytest.raises(ValueError, match="Invalid subject"):
        ds.data_path(2)


def test_nemar_failure_is_raised(monkeypatch):
    ds = Ma2022(subjects=[3])
    monkeypatch.setattr(
        ds, "_download_nemar", Mock(side_effect=NemarDownloadError("unavailable"))
    )
    with pytest.raises(NemarDownloadError, match="unavailable"):
        ds.get_data([3], cache_config={"use": False})


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_session_count_is_checked(bids_root, dataset, change):
    edf = next(bids_root.glob("sub-003/ses-01/eeg/*.edf"))
    if change == "missing":
        edf.unlink()
    else:
        edf.with_name(edf.name.replace("_eeg", "_run-02_eeg")).write_bytes(
            edf.read_bytes()
        )
    with pytest.raises(FileNotFoundError, match="five EDF sessions"):
        dataset.data_path(3)


def test_get_data(dataset):
    sessions = dataset.get_data([3], cache_config={"use": False})[3]
    assert list(sessions) == [str(i) for i in range(N_SESSIONS)]
    for session, runs in sessions.items():
        raw = runs["0"]
        assert raw.ch_names == MA2022_CH_NAMES
        assert raw.info["sfreq"] == 250
        assert raw.info["bads"] == (["F3"] if session == "1" else [])
        assert all(np.isfinite(ch["loc"][:3]).all() for ch in raw.info["chs"])
        assert_allclose(np.abs(raw.get_data(picks=["C3"])).max(), 20e-6, rtol=0.003)
        events, _ = mne.events_from_annotations(raw, event_id=dataset.event_id)
        assert_array_equal(events[:, 0], [0, 1000, 2000])
        assert_array_equal(events[:, 2], [2, 1, 2])


def test_paradigm(dataset):
    channels = [ch for ch in MA2022_CH_NAMES if ch != "F3"]
    paradigm = MotorImagery(n_classes=2, channels=channels)
    X, y, metadata = paradigm.get_data(dataset, [3], cache_config={"use": False})
    assert X.shape == (3 * N_SESSIONS, len(channels), 1000)
    assert_array_equal(y, ["right_hand", "left_hand", "right_hand"] * N_SESSIONS)
    assert metadata["session"].nunique() == N_SESSIONS
