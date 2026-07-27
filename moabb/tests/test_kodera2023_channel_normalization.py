"""Regression tests for Kodera2023's two source channel layouts."""

import mne
import numpy as np
import pytest

from moabb.datasets.kodera2023 import _COMMON_EEG_CHANNELS, Kodera2023


_FULL_EEG = (
    "Fp1",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Cz",
    "Fz",
    "Pz",
)


def _raw(ch_names):
    raw = mne.io.RawArray(
        np.zeros((len(ch_names), 100)),
        mne.create_info(ch_names, sfreq=500, ch_types="eeg"),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([0.1], [0], ["Stimulus/S  1"]))
    return raw


@pytest.mark.parametrize(
    "source_channels",
    [
        _FULL_EEG + ("17", "18"),
        # The 1000 Hz cohort's physical order is also the public canonical
        # order. Use a permutation to prove the loader actively reorders it.
        tuple(reversed(_COMMON_EEG_CHANNELS)),
    ],
)
def test_normalizes_every_observed_source_layout(monkeypatch, source_channels):
    monkeypatch.setattr(
        "moabb.datasets.kodera2023.mne.io.read_raw_brainvision",
        lambda *args, **kwargs: _raw(source_channels),
    )

    raw = Kodera2023()._read_run("1m03042023lh1.vhdr")

    assert raw.ch_names == list(_COMMON_EEG_CHANNELS)
    assert raw.get_channel_types() == ["eeg"] * len(_COMMON_EEG_CHANNELS)
    assert raw.annotations.description.tolist() == ["left_hand"]


def test_rejects_recording_without_a_required_shared_channel(monkeypatch):
    source_channels = tuple(ch for ch in _COMMON_EEG_CHANNELS if ch != "C4")
    monkeypatch.setattr(
        "moabb.datasets.kodera2023.mne.io.read_raw_brainvision",
        lambda *args, **kwargs: _raw(source_channels),
    )

    with pytest.raises(ValueError, match="missing required shared EEG channels.*C4"):
        Kodera2023()._read_run("1m03042023lh1.vhdr")
