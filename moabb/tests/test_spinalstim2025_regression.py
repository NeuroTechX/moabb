"""Regression tests for SpinalStim2025 cross-cohort channel order."""

import mne
import numpy as np

from moabb.datasets.spinalstim2025 import _AUX_CHANNELS, _EEG_CHANNELS, SpinalStim2025


def test_later_cohort_o2_oz_swap_is_normalized_before_cache_combination(
    tmp_path, monkeypatch
):
    source_channels = [*_EEG_CHANNELS[:-2], "O2", "OZ", *_AUX_CHANNELS]
    info = mne.create_info(
        source_channels,
        sfreq=512,
        ch_types=["eeg"] * len(_EEG_CHANNELS) + ["misc"] * len(_AUX_CHANNELS),
    )
    raw = mne.io.RawArray(np.zeros((len(source_channels), 16)), info, verbose=False)
    raw.set_annotations(mne.Annotations([0.0], [0.0], ["769"]))

    monkeypatch.setattr(
        "moabb.datasets.spinalstim2025.mne.io.read_raw_gdf",
        lambda *args, **kwargs: raw,
    )

    normalized = SpinalStim2025._read_run(tmp_path / "d3_run.gdf")

    assert normalized.ch_names == [*_EEG_CHANNELS, *_AUX_CHANNELS]
    assert normalized.annotations.description.tolist() == ["left_hand"]


def test_missing_spinalstim_eeg_channel_raises_useful_error(tmp_path, monkeypatch):
    channels = _EEG_CHANNELS[:-1]
    raw = mne.io.RawArray(
        np.zeros((len(channels), 16)),
        mne.create_info(channels, sfreq=512, ch_types="eeg"),
        verbose=False,
    )
    monkeypatch.setattr(
        "moabb.datasets.spinalstim2025.mne.io.read_raw_gdf",
        lambda *args, **kwargs: raw,
    )

    with np.testing.assert_raises_regex(ValueError, "missing required EEG channels.*O2"):
        SpinalStim2025._read_run(tmp_path / "incomplete.gdf")
