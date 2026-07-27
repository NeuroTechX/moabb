"""Regression tests for the common InEarMI2026 EEG montage."""

import mne
import numpy as np
import pytest

from moabb.datasets.inear_mi2026 import InEarMI2026


def _raw(channel_names):
    channel_types = ["emg" if name.startswith("HAND") else "eeg" for name in channel_names]
    info = mne.create_info(channel_names, sfreq=500.0, ch_types=channel_types)
    return mne.io.RawArray(np.zeros((len(channel_names), 20)), info, verbose=False)


@pytest.mark.parametrize(
    "channel_names",
    [
        ["EARR", "EARL"],
        ["EARR", "EARL", "HANDR", "HANDL"],
        ["EARR", "EARL", "C4", "C3", "CZ", "HANDR", "HANDL"],
    ],
)
def test_inear_mi2026_uses_real_common_in_ear_pair(channel_names):
    """All observed 2/4/7-channel source variants yield one canonical raw."""
    raw = InEarMI2026._select_common_in_ear_eeg(_raw(channel_names), subject=1)
    assert raw.ch_names == ["EARR", "EARL"]
    assert raw.get_data().shape == (2, 20)


def test_inear_mi2026_rejects_missing_common_in_ear_channel():
    with pytest.raises(ValueError, match="missing required in-ear channel"):
        InEarMI2026._select_common_in_ear_eeg(_raw(["EARR", "C4"]), subject=1)
