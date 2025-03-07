import numpy as np
import pytest
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.utils import check_random_state

from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import CrossSessionSplitter, WithinSessionSplitter
from moabb.paradigms.motor_imagery import FakeImageryParadigm


dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=3, seed=12)
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


def eval_split_cross_session(shuffle, random_state):
    rng = check_random_state(random_state) if shuffle else None

    X, y, metadata = paradigm.get_data(dataset=dataset)
    subjects = metadata["subject"].unique()

    if shuffle:
        rng.shuffle(subjects)

    for subject in subjects:
        subject_mask = metadata["subject"] == subject
        subject_metadata = metadata[subject_mask]
        subject_sessions = subject_metadata["session"].unique()

        if shuffle:
            rng.shuffle(subject_sessions)

        splitter = LeaveOneGroupOut()

        for train_ix, test_ix in splitter.split(
            X=subject_metadata, y=y[subject_mask], groups=subject_metadata["session"]
        ):
            yield subject_metadata.index[train_ix], subject_metadata.index[test_ix]


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


@pytest.mark.parametrize("splitter", [WithinSessionSplitter, CrossSessionSplitter])
def test_custom_inner_cv(splitter):
    X, y, metadata = paradigm.get_data(dataset=dataset)

    # Use a custom inner cv
    split = splitter(cv_class=TimeSeriesSplit, max_train_size=2)

    for train, test in split.split(y, metadata):
        # Check if the output is the same as the input
        assert len(train) <= 2  # Due to TimeSeriesSplit constraints
        assert len(test) >= 20


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session(shuffle, random_state):
    X, y, metadata = paradigm.get_data(dataset=dataset)

    split = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    for idx_train_splitter, idx_test_splitter in split.split(y, metadata):
        # Check if the output is the same as the input
        session_train = metadata.iloc[idx_train_splitter]["session"].unique()
        session_test = metadata.iloc[idx_test_splitter]["session"].unique()
        assert not np.intersect1d(session_train, session_test).size
        assert (
            np.union1d(session_train, session_test).size
            == metadata["session"].unique().size
        )


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session_compatibility(shuffle, random_state):
    _, y, metadata = paradigm.get_data(dataset=dataset)

    splitter = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_cross_session(shuffle=shuffle, random_state=random_state),
        splitter.split(y, metadata),
    ):
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_cross_session_is_shuffling():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    splitter_no_shuffle = CrossSessionSplitter(shuffle=False)
    splitter_shuffle = CrossSessionSplitter(shuffle=True, random_state=3)

    for idx, ((train, test), (train_shuffle, test_shuffle)) in enumerate(
        zip(
            splitter_no_shuffle.split(y, metadata),
            splitter_shuffle.split(y, metadata),
        )
    ):
        print(f"\n--- Fold {idx} ---")
        print("Train indices no shuffle:", train)
        print("Train indices shuffle   :", train_shuffle)
        print("Test indices no shuffle :", test)
        print("Test indices shuffle    :", test_shuffle)

        train_equal = np.array_equal(train, train_shuffle)
        test_equal = np.array_equal(test, test_shuffle)

        print("Train indices equal:", train_equal)
        print("Test indices equal :", test_equal)

        assert not train_equal or not test_equal, f"Shuffle had no effect on fold {idx}"
