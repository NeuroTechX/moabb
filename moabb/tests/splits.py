import numpy as np
from sklearn.model_selection import StratifiedKFold

from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import WithinSessionSplitter
from moabb.paradigms.motor_imagery import FakeImageryParadigm


dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=1, seed=12)
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
                yield train, test


# TODO: test shuffle and random_state
def test_within_session():
    X, y, metadata = paradigm.get_data(dataset=dataset)

    split = WithinSessionSplitter(n_folds=5, random_state=42, shuffle=True)

    for ix, ((idx_train_t, idx_test_t), (train, test)) in enumerate(
        zip(eval_split_within_session(), split.split(y, metadata))
    ):
        # Check if the output is the same as the input
        assert np.array_equal(train, idx_train_t)
        assert np.array_equal(test, idx_test_t)
