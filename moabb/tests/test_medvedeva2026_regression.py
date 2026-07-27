"""Regression test for Medvedeva2026 legacy montage runs."""

from unittest.mock import patch

import mne
import numpy as np
import pytest

from moabb.datasets.medvedeva2026 import Medvedeva2026, _EEG_CHANNELS


def _raw(channel_names):
    info = mne.create_info(channel_names, sfreq=500.0, ch_types="eeg")
    return mne.io.RawArray(np.zeros((len(channel_names), 10)), info, verbose=False)


def test_medvedeva2026_skips_only_legacy_incompatible_montage(monkeypatch):
    dataset = Medvedeva2026()
    legacy = ("F7", "F3", "F4", "F8", "C3", "C4", "A1", "A2")
    paths = ["L_01_session2_left1.fif", "L_01_session3_right1.fif"]
    monkeypatch.setattr(dataset, "data_path", lambda subject: paths)

    with (
        patch(
            "moabb.datasets.medvedeva2026.mne.io.read_raw_fif",
            side_effect=[_raw(legacy), _raw(_EEG_CHANNELS)],
        ),
        pytest.warns(UserWarning, match="incompatible EEG montage"),
    ):
        sessions = dataset._get_single_subject_data(1)

    assert list(sessions) == ["3"]
    loaded = sessions["3"]["0intactright1"]
    assert loaded.ch_names == list(_EEG_CHANNELS)
