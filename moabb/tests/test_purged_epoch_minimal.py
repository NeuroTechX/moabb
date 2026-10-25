"""Opt-in raw-window overlap CV; does not certify independence after IIR."""

import numpy as np
import pytest

from moabb.evaluations.splitters import PurgedEpochKFold


def _groups(starts, width, recording_ids=None):
    n = len(starts)
    groups = np.empty((n, 3), dtype=object)
    groups[:, 0] = ["one"] * n if recording_ids is None else recording_ids
    groups[:, 1] = list(starts)
    groups[:, 2] = [width] * n
    return groups


def _has_overlap(train, test, groups):
    return any(
        groups[i, 0] == groups[j, 0]
        and groups[i, 1] < groups[j, 1] + groups[j, 2]
        and groups[j, 1] < groups[i, 1] + groups[i, 2]
        for i in train
        for j in test
    )


def test_complete_chronological_partition_with_exact_purge():
    groups = _groups(np.arange(30) * 100, 500)
    y = np.tile([0, 1], 15)
    folds = list(PurgedEpochKFold(5).split(np.zeros(30), y, groups))
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(30))
    assert len(folds) == 5
    for train, test in folds:
        assert not _has_overlap(train, test, groups)
        assert set(y[train]) == {0, 1}
        assert set(y[test]) == {0, 1}


def test_half_open_touching_boundaries_are_not_purged():
    actual = PurgedEpochKFold._overlap_mask(
        np.array([0, 10, 20]), np.array([10, 20, 30]), np.array([10]), np.array([20])
    )
    assert actual.tolist() == [False, True, False]


def test_independent_recordings_can_reuse_sample_coordinates():
    groups = _groups(list(np.arange(8) * 800) * 2, 500, ["A"] * 8 + ["B"] * 8)
    for train, test in PurgedEpochKFold(4).split(np.zeros(16), groups=groups):
        assert len(test) == 4
        assert len(train) == 12
        assert not _has_overlap(train, test, groups)


def test_tuple_recording_ids_are_atomic_with_reused_run_labels():
    # Subject/session/run is naturally a tuple, not an array dimension.
    labels = [(1, "session-0", "run-0")] * 8 + [(2, "session-0", "run-0")] * 8
    groups = _groups(list(np.arange(8) * 800) * 2, 500, labels)
    folds = list(PurgedEpochKFold(4).split(np.zeros(16), groups=groups))
    assert len(folds) == 4
    for train, test in folds:
        assert len(train) == 12
        assert len(test) == 4
        assert not _has_overlap(train, test, groups)


def test_unhashable_recording_id_fails_with_clear_error():
    groups = _groups(np.arange(8) * 200, 100)
    groups[0, 0] = ["subject", "run"]
    with pytest.raises(ValueError, match="hashable"):
        list(PurgedEpochKFold(2).split(np.zeros(8), groups=groups))


def test_chronology_is_not_assumed_from_dataframe_row_order():
    starts = np.random.default_rng(1).permutation(np.arange(20) * 500)
    groups = _groups(starts, 400)
    folds = list(PurgedEpochKFold(4).split(np.zeros(20), groups=groups))
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(20))
    for _, test in folds:
        positions = np.searchsorted(np.sort(starts), starts[test])
        assert sorted(positions) == list(range(min(positions), max(positions) + 1))


def test_only_exact_integer_coordinates_and_no_int64_overflow():
    groups = _groups(np.arange(8) * 200, 100)
    for invalid in (0.5, 1.0, True, "1"):
        malformed = groups.copy()
        malformed[0, 1] = invalid
        with pytest.raises(ValueError, match="integers"):
            list(PurgedEpochKFold(2).split(np.zeros(8), groups=malformed))
    shifted = groups.copy()
    shifted[:, 1] = [int(x) + 2**53 + 101 for x in groups[:, 1]]
    a = list(PurgedEpochKFold(2).split(np.zeros(8), groups=groups))
    b = list(PurgedEpochKFold(2).split(np.zeros(8), groups=shifted))
    for (train_a, test_a), (train_b, test_b) in zip(a, b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
    overflow = groups.copy()
    overflow[0, 1] = 2**63 - 1
    with pytest.raises(ValueError, match="stop exceeds"):
        list(PurgedEpochKFold(2).split(np.zeros(8), groups=overflow))


def test_missing_inputs_and_unsupported_class_partition_fail_closed():
    groups = _groups(np.arange(10) * 800, 500)
    with pytest.raises(ValueError, match="explicit epoch timing"):
        list(PurgedEpochKFold(2).split(np.zeros(10)))
    with pytest.raises(ValueError, match="shape|array"):
        list(PurgedEpochKFold(2).split(np.zeros(10), groups=np.ones((10, 2))))
    labels = np.array([0] * 8 + [1] * 2)
    with pytest.raises(ValueError, match="lacks a class"):
        list(PurgedEpochKFold(5).split(np.zeros(10), labels, groups))


def test_prefix_max_interval_search_equals_independent_dense_oracle():
    rng = np.random.default_rng(20261025)
    for _ in range(150):
        train_start = rng.integers(-20, 90, size=35)
        train_stop = train_start + rng.integers(1, 30, size=35)
        test_start = rng.integers(-20, 90, size=25)
        test_stop = test_start + rng.integers(1, 30, size=25)
        observed = PurgedEpochKFold._overlap_mask(
            train_start, train_stop, test_start, test_stop
        )
        expected = (
            (train_start[:, None] < test_stop[None, :])
            & (test_start[None, :] < train_stop[:, None])
        ).any(axis=1)
        np.testing.assert_array_equal(observed, expected)


def test_large_cohort_has_no_quadratic_overlap_matrix():
    n = 30000
    starts = np.arange(n, dtype=np.int64) * 4
    actual = PurgedEpochKFold._overlap_mask(starts + 1, starts + 3, starts, starts + 2)
    assert actual.shape == (n,)
    assert actual.all()


def test_docstring_preserves_filtering_limitation():
    doc = PurgedEpochKFold.__doc__
    assert "IIR" in doc
    assert "not" in doc.lower()
