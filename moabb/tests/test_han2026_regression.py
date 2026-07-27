"""Regression tests for Han2026 event annotations."""

import mne
import numpy as np

from moabb.datasets.han2026 import Han2026


def test_duplicate_pure_condition_annotation_is_removed():
    """An exact duplicate class marker would otherwise create duplicate windows."""
    raw = mne.io.RawArray(np.zeros((1, 1_000)), mne.create_info(["Cz"], 100))
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.0, 3.0, 3.0],
            duration=[0.001, 0.001, 0.001, 0.001],
            description=[
                "motor_observation",
                "motor_observation",
                "motor_imagery",
                "A88",
            ],
        )
    )

    Han2026._drop_duplicate_trial_annotations(raw)

    assert raw.annotations.description.tolist() == [
        "motor_observation",
        "motor_imagery",
        "A88",
    ]


def test_distinct_annotation_labels_at_one_onset_are_preserved():
    """Conflicting annotations are not silently discarded as duplicates."""
    raw = mne.io.RawArray(np.zeros((1, 1_000)), mne.create_info(["Cz"], 100))
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.0],
            duration=[0.001, 0.001],
            description=["motor_observation", "motor_imagery"],
        )
    )

    Han2026._drop_duplicate_trial_annotations(raw)

    assert raw.annotations.description.tolist() == ["motor_observation", "motor_imagery"]


def test_same_class_with_different_duration_is_preserved():
    """Only exact duplicates are removed, not duration disagreements."""
    raw = mne.io.RawArray(np.zeros((1, 1_000)), mne.create_info(["Cz"], 100))
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.0],
            duration=[0.001, 0.002],
            description=["motor_observation", "motor_observation"],
        )
    )

    Han2026._drop_duplicate_trial_annotations(raw)

    assert len(raw.annotations) == 2
