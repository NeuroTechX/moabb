import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import WithinSessionSplitter
from moabb.paradigms.motor_imagery import FakeImageryParadigm


dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=3, seed=12)
paradigm = FakeImageryParadigm()


# Split done for the Within Session evaluation
def eval_split_within_session():
    for subject in dataset.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        sessions = metadata.session
        for session in np.unique(sessions):
            ix = sessions == session
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            X_, y_ = X[ix], y[ix]
            for train, test in cv.split(X_, y_):
                yield X_[train], X_[test]


# Split done for the Cross Session evaluation
def eval_split_cross_session():
    for subject in dataset.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        groups = metadata.session.values
        cv = LeaveOneGroupOut()
        for train, test in cv.split(X, y, groups):
            yield X[train], X[test]


# Split done for the Cross Subject evaluation
def eval_split_cross_subject():
    X, y, metadata = paradigm.get_data(dataset=dataset)
    groups = metadata.subject.values
    cv = LeaveOneGroupOut()
    for train, test in cv.split(X, y, groups):
        yield X[train], X[test]

# TODO: test shuffle and random_state
def test_within_session():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    split = WithinSessionSplitter(n_folds=5)

    for ix, ((X_train_t, X_test_t), (train, test)) in enumerate(
        zip(eval_split_within_session(), split.split(X, y, metadata, random_state=42))
    ):
        X_train, X_test = X[train], X[test]

        # Check if the output is the same as the input
        assert np.array_equal(X_train, X_train_t)
        assert np.array_equal(X_test, X_test_t)

