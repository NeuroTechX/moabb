import numpy as np
import pytest
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.utils import check_random_state

from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import CrossSubjectSplitter, WithinSessionSplitter
from moabb.paradigms.motor_imagery import FakeImageryParadigm


dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=6, seed=12)
paradigm = FakeImageryParadigm()


# Split done for the Within Session evaluation
def eval_split_within_session(shuffle, random_state):
    rng = check_random_state(random_state) if shuffle else None

    X, y, metadata = paradigm.get_data(dataset=dataset)
    all_index = metadata.index.values
    subjects = metadata["subject"].unique()
    if shuffle:
        rng.shuffle(subjects)

    for i, subject in enumerate(subjects):
        subject_mask = metadata["subject"] == subject

        subject_indices = all_index[subject_mask]
        subject_metadata = metadata[subject_mask]
        sessions = subject_metadata["session"].unique()
        y_subject = y[subject_mask]

        if shuffle:
            rng.shuffle(sessions)

        for session in sessions:
            session_mask = subject_metadata["session"] == session
            indices = subject_indices[session_mask]
            metadata_ = subject_metadata[session_mask]
            y_ = y_subject[session_mask]

            cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=rng)

            for idx_train, idx_test in cv.split(metadata_, y_):
                yield indices[idx_train], indices[idx_test]


def eval_split_cross_subject(shuffle, random_state):
    rng = check_random_state(random_state) if shuffle else None

    X, y, metadata = paradigm.get_data(dataset=dataset)
    subjects = metadata["subject"].unique()

    if shuffle:
        rng.shuffle(subjects)
        splitter = GroupShuffleSplit(n_splits=len(subjects), random_state=rng)
    else:
        splitter = LeaveOneGroupOut()

    for train_subj_idx, test_subj_idx in splitter.split(
        X=np.zeros(len(subjects)), y=None, groups=subjects
    ):
        train_mask = metadata["subject"].isin(subjects[train_subj_idx])
        test_mask = metadata["subject"].isin(subjects[test_subj_idx])

        yield metadata.index[train_mask].values, metadata.index[test_mask].values


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_within_session_compatibility(shuffle, random_state):
    X, y, metadata = paradigm.get_data(dataset=dataset)

    split = WithinSessionSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_within_session(shuffle=shuffle, random_state=random_state),
        split.split(y, metadata),
    ):
        # Check if the output is the same as the input
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_is_shuffling():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    split = WithinSessionSplitter(n_folds=5, shuffle=False)
    split_shuffle = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=3)

    for (train, test), (train_shuffle, test_shuffle) in zip(
        split.split(y, metadata), split_shuffle.split(y, metadata)
    ):
        # Check if the output is the same as the input
        assert not np.array_equal(train, train_shuffle)
        assert not np.array_equal(test, test_shuffle)


def test_custom_inner_cv():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    # Use a custom inner cv
    split = WithinSessionSplitter(cv_class=TimeSeriesSplit, max_train_size=2)

    for train, test in split.split(y, metadata):
        assert len(train) == 2  # Due to TimeSeriesSplit constraints
        assert len(test) == 20


def test_custom_shuffle_group():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    n_splits = 5
    splitter = CrossSubjectSplitter(
        shuffle=True,
        random_state=42,
        cv_class=GroupShuffleSplit,
        n_splits=n_splits,
    )

    splits = list(splitter.split(y, metadata))

    assert len(splits) == n_splits, f"Expected {n_splits} splits, got {len(splits)}"

    for train, test in splits:
        train_subjects = metadata.iloc[train]["subject"].unique()
        test_subjects = metadata.iloc[test]["subject"].unique()

        # Assert no overlap between train and test subjects
        assert len(set(train_subjects) & set(test_subjects)) == 0

    # Check if shuffling produces different splits
    splitter_different_seed = CrossSubjectSplitter(
        random_state=24,
        shuffle=True,
        cv_class=GroupShuffleSplit,
        n_splits=n_splits,
    )
    splits_different_seed = list(splitter_different_seed.split(y, metadata))

    assert not all(
        np.array_equal(train, train_alt) and np.array_equal(test, test_alt)
        for (train, test), (train_alt, test_alt) in zip(splits, splits_different_seed)
    )


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_subject_compatibility(shuffle, random_state):
    X, y, metadata = paradigm.get_data(dataset=dataset)

    splitter = CrossSubjectSplitter(shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_cross_subject(shuffle=shuffle, random_state=random_state),
        splitter.split(y, metadata),
    ):
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_cross_subject_is_shuffling():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    splitter_no_shuffle = CrossSubjectSplitter(shuffle=False)
    splitter_shuffle = CrossSubjectSplitter(shuffle=True, random_state=123)

    for (train, test), (train_shuffle, test_shuffle) in zip(
        splitter_no_shuffle.split(y, metadata),
        splitter_shuffle.split(y, metadata),
    ):
        assert not np.array_equal(train, train_shuffle)
        assert not np.array_equal(test, test_shuffle)
