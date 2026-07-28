"""Regression coverage for Russo2024's data-borne channel provenance."""

import numpy as np
import pytest

from moabb.datasets.russo2024 import Russo2024, _load_block


@pytest.mark.parametrize(
    "expected_names",
    [
        [f"ExG{index}" for index in range(1, 25)],
        [
            "F1",
            "Fz",
            "F2",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P3",
            "P1",
            "Pz",
            "P2",
        ],
    ],
)
def test_russo2024_preserves_data_borne_channel_names_without_montage(
    monkeypatch, expected_names
):
    """Legacy and newer cohort labels must remain distinct and unmapped."""
    channel_names = np.empty((1, 24), dtype=object)
    for index in range(24):
        channel_names[0, index] = np.array([expected_names[index]])

    monkeypatch.setattr(
        "moabb.datasets.russo2024.loadmat",
        lambda _: {
            "EEG_data": np.zeros((20, 24)),
            "Fs": np.array([[2048.0]]),
            "channel_names": channel_names,
            "prompt_times": np.array([[1, 0.0, 0.005, 0.010, 0.0]]),
            "prompt_start_time_marker": np.array([[0.0]]),
        },
    )

    raw = _load_block("synthetic.mat")

    assert raw.ch_names == expected_names + ["STI 014"]
    assert raw.info["dig"] is None
    assert (
        Russo2024.METADATA.acquisition.hardware
        == "TMSi Porti 7 32-channel biosignal amplifier"
    )
