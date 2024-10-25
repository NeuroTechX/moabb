import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state

from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import WithinSessionSplitter
from moabb.paradigms.motor_imagery import FakeImageryParadigm


dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=3, seed=12)
paradigm = FakeImageryParadigm()


# Split done for the Within Session evaluation
def eval_split_within_session(shuffle, random_state):
    random_state = check_random_state(random_state) if shuffle else None
    for subject in dataset.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        sessions = metadata.session
        for session in np.unique(sessions):
            ix = sessions == session
            cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=random_state)
            X_, metadata_, y_ = X[ix], y[ix], metadata[ix]
            for train, test in cv.split(y_, metadata_):
                yield X_[train], X_[test]


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("random_state", [0, 42])
def test_within_session(shuffle, random_state):
    X, y, metadata = paradigm.get_data(dataset=dataset)

    split = WithinSessionSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (X_train_t, X_test_t), (train, test) in zip(
        eval_split_within_session(shuffle=shuffle, random_state=random_state),
        split.split(y, metadata),
    ):
        X_train, X_test = X[train], X[test]

        # Check if the output is the same as the input
        assert np.array_equal(X_train, X_train_t)
        assert np.array_equal(X_test, X_test_t)


def test_is_shuffling():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    split = WithinSessionSplitter(n_folds=5, shuffle=False)
    split_shuffle = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=3)

    for (train, test), (train_shuffle, test_shuffle) in zip(
        split.split(y, metadata), split_shuffle.split(y, metadata)
    ):
        # Check if the output is the same as the input
        assert np.array_equal(train, train_shuffle) == False
        assert np.array_equal(test, test_shuffle) == False
