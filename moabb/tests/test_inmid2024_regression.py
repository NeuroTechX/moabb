"""Regression tests for malformed InMID2024 MAT segments."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from moabb.datasets.inmid2024 import InMID2024


def _joined(n_trials=1, malformed_at=None):
    """Small MATLAB-cell-like ``joined_data`` array for a loader test."""
    joined = np.empty((1, n_trials), dtype=object)
    for trial_idx in range(n_trials):
        shape = (6, 14) if trial_idx != malformed_at else (6, 6)
        joined[0, trial_idx] = np.zeros(shape)
    return {"joined_data": joined}


def _loadmat_with_one_bad_execution_trunk(path):
    is_execution = "Movement Dataset" in str(path)
    is_trunk = "Standing and Sitting" in str(path)
    return _joined(
        n_trials=2 if is_execution else 1,
        malformed_at=1 if is_execution and is_trunk else None,
    )


def test_inmid2024_skips_only_malformed_segment_and_preserves_classes():
    dataset = InMID2024()
    with (
        patch.object(dataset, "_extract_root", return_value=Path("/fake")),
        patch(
            "moabb.datasets.inmid2024.sio.loadmat",
            side_effect=_loadmat_with_one_bad_execution_trunk,
        ),
        pytest.warns(UserWarning, match="Skipping malformed InMID2024 segment"),
    ):
        sessions = dataset._get_single_subject_data(5)

    imagery = sessions["0imagery"]["0"]
    execution = sessions["1execution"]["0"]
    assert imagery.get_data().shape == (14, 18)
    assert execution.get_data().shape == (14, 30)
    assert set(imagery.annotations.description) == {
        "left_hand",
        "right_hand",
        "trunk",
    }
    assert set(execution.annotations.description) == {
        "left_hand",
        "right_hand",
        "trunk",
    }
    assert len(execution.annotations) == 5


def test_inmid2024_rejects_session_that_loses_a_class():
    dataset = InMID2024()

    def loadmat_without_usable_trunk(path):
        is_trunk = "Standing and Sitting" in str(path)
        return _joined(malformed_at=0 if is_trunk else None)

    with (
        patch.object(dataset, "_extract_root", return_value=Path("/fake")),
        patch(
            "moabb.datasets.inmid2024.sio.loadmat",
            side_effect=loadmat_without_usable_trunk,
        ),
        pytest.warns(UserWarning, match="Skipping malformed InMID2024 segment"),
        pytest.raises(ValueError, match="no usable segments for event code\\(s\\) \\[3\\]"),
    ):
        dataset._get_single_subject_data(5)
