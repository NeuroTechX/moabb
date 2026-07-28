"""Regression coverage for Russo2024's inferred cross-cohort alignment."""

import numpy as np
import pytest

from moabb.datasets.russo2024 import (
    _ANATOMICAL_CHANNELS,
    _CHANNEL_MAPPING_STATUS,
    Russo2024,
    _load_block,
)


@pytest.mark.parametrize(
    "stored_names",
    [[f"ExG{index}" for index in range(1, 25)], list(_ANATOMICAL_CHANNELS)],
)
def test_russo2024_aligns_both_cohorts_to_anatomical_montage(monkeypatch, stored_names):
    """Legacy and newer blocks must expose one positional channel schema."""
    channel_names = np.empty((1, 24), dtype=object)
    for index in range(24):
        channel_names[0, index] = np.array([stored_names[index]])

    # Give every source column a distinct value so the test catches an
    # accidental data permutation in addition to checking the channel labels.
    eeg_data = np.tile(np.arange(24, dtype=float), (20, 1))

    monkeypatch.setattr(
        "moabb.datasets.russo2024.loadmat",
        lambda _: {
            "EEG_data": eeg_data,
            "Fs": np.array([[2048.0]]),
            "channel_names": channel_names,
            "prompt_times": np.array([[1, 0.0, 0.005, 0.010, 0.0]]),
            "prompt_start_time_marker": np.array([[0.0]]),
        },
    )

    raw = _load_block("synthetic.mat")

    assert raw.ch_names == list(_ANATOMICAL_CHANNELS) + ["STI 014"]
    assert raw.get_montage() is not None
    assert all(
        np.isfinite(raw.get_montage().get_positions()["ch_pos"][name]).all()
        for name in _ANATOMICAL_CHANNELS
    )
    np.testing.assert_allclose(
        raw.get_data(picks=list(_ANATOMICAL_CHANNELS))[:, 0], np.arange(24) * 1e-6
    )
    assert (
        Russo2024.METADATA.acquisition.hardware
        == "TMSi Porti 7 32-channel biosignal amplifier"
    )
    assert Russo2024.METADATA.acquisition.sensors == list(_ANATOMICAL_CHANNELS)
    assert Russo2024.channel_mapping_status == _CHANNEL_MAPPING_STATUS
