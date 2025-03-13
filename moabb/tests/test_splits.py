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
from moabb.evaluations.splitters import CrossSessionSplitter, WithinSessionSplitter
from moabb.paradigms.motor_imagery import FakeImageryParadigm


@pytest.fixture
def data():
    dataset = FakeDataset(
        ["left_hand", "right_hand"], n_subjects=5, seed=12, n_sessions=5
    )
    paradigm = FakeImageryParadigm()
    return paradigm.get_data(dataset=dataset)


# Split done for the Within Session evaluation
def eval_split_within_session(shuffle, random_state, data):
    _, y, metadata = data
    rng = check_random_state(random_state) if shuffle else None

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


def eval_split_cross_session(shuffle, random_state, data):
    _, y, metadata = data

    rng = check_random_state(random_state) if shuffle else None

    subjects = metadata["subject"].unique()

    for subject in subjects:
        subject_mask = metadata["subject"] == subject
        subject_metadata = metadata[subject_mask]
        subject_sessions = subject_metadata["session"].unique()

        if shuffle:
            splitter = GroupShuffleSplit(n_splits=len(subject_sessions), random_state=rng)
        else:
            splitter = LeaveOneGroupOut()

        for train_ix, test_ix in splitter.split(
            X=subject_metadata, y=y[subject_mask], groups=subject_metadata["session"]
        ):
            yield subject_metadata.index[train_ix], subject_metadata.index[test_ix]


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_within_session_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_within_session(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        # Check if the output is the same as the input
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_is_shuffling(data):
    X, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=False)
    split_shuffle = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=3)

    for (train, test), (train_shuffle, test_shuffle) in zip(
        split.split(y, metadata), split_shuffle.split(y, metadata)
    ):
        # Check if the output is the same as the input
        assert not np.array_equal(train, train_shuffle)
        assert not np.array_equal(test, test_shuffle)


@pytest.mark.parametrize("splitter", [WithinSessionSplitter, CrossSessionSplitter])
def test_custom_inner_cv(splitter, data):
    X, y, metadata = data
    # Use a custom inner cv
    split = splitter(cv_class=TimeSeriesSplit, max_train_size=2)

    for train, test in split.split(y, metadata):
        # Check if the output is the same as the input
        assert len(train) <= 2  # Due to TimeSeriesSplit constraints
        assert len(test) >= 20


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session(shuffle, random_state, data):
    _, y, metadata = data

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
def test_cross_session_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    splitter = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_cross_session(shuffle=shuffle, random_state=random_state, data=data),
        splitter.split(y, metadata),
    ):
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_cross_session_is_shuffling_and_order(data):
    _, y, metadata = data

    splitter_no_shuffle = CrossSessionSplitter(shuffle=False)
    splitter_shuffle = CrossSessionSplitter(shuffle=True, random_state=3)

    splits_no_shuffle = list(splitter_no_shuffle.split(y, metadata))
    splits_shuffle = list(splitter_shuffle.split(y, metadata))

    train_diff = []
    test_diff = []

    # For tracking session order differences
    session_orders_no_shuffle = []
    session_orders_shuffle = []

    for i, ((train_ns, test_ns), (train_s, test_s)) in enumerate(
        zip(splits_no_shuffle, splits_shuffle)
    ):
        print(f"\nFold {i}:")

        # Get session ordering for non-shuffled and shuffled
        train_ns_sessions = metadata.iloc[train_ns]["session"].unique()
        test_ns_sessions = metadata.iloc[test_ns]["session"].unique()
        train_s_sessions = metadata.iloc[train_s]["session"].unique()
        test_s_sessions = metadata.iloc[test_s]["session"].unique()

        print(f"Train no shuffle sessions: {train_ns_sessions}")
        print(f"Test no shuffle sessions : {test_ns_sessions}")
        print(f"Train shuffled sessions  : {train_s_sessions}")
        print(f"Test shuffle sessions    : {test_s_sessions}")

        # Track if indices are the same
        train_diff.append(np.array_equal(train_ns, train_s))
        test_diff.append(np.array_equal(test_ns, test_s))

        # Track session orders
        session_orders_no_shuffle.append(
            (list(train_ns_sessions), list(test_ns_sessions))
        )
        session_orders_shuffle.append((list(train_s_sessions), list(test_s_sessions)))

    # Check if indices are different in at least some folds
    assert not all(train_diff), "All train indices are identical despite shuffle"
    assert not all(test_diff), "All test indices are identical despite shuffle"

    # Check if session ordering is different
    session_order_differences = [
        not (
            np.array_equal(no_shuffle[0], shuffle[0])
            and np.array_equal(no_shuffle[1], shuffle[1])
        )
        for no_shuffle, shuffle in zip(session_orders_no_shuffle, session_orders_shuffle)
    ]

    assert any(session_order_differences), (
        "Session ordering is identical in all folds despite shuffle. "
        "When shuffle=True, we expect some difference in the session ordering."
    )


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session_unique_sessions(shuffle, random_state, data):
    _, y, metadata = data

    split = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)
    splits = list(split.split(y, metadata))

    for i, (train, test) in enumerate(splits):
        train_sessions = metadata.iloc[train]["session"].unique()
        test_sessions = metadata.iloc[test]["session"].unique()
        assert not np.intersect1d(
            train_sessions, test_sessions
        ).size, f"Fold {i} train and test sessions overlap"
