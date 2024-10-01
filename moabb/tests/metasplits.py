import os
import os.path as osp

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from moabb.evaluations.metasplitters import (OfflineSplit,PseudoOnlineSplit, SamplerSplit)
from moabb.evaluations.splitters import (CrossSessionSplitter, CrossSubjectSplitter, WithinSessionSplitter)
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.motor_imagery import FakeImageryParadigm

dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=3, seed=12)
paradigm = FakeImageryParadigm()

# Split done for the Within Session evaluation
def eval_sampler_split():
    for subject in dataset.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        sessions = metadata.session
        for session in np.unique(sessions):
            ix = sessions == session
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            X_, y_, meta_ = X[ix], y[ix], metadata.loc[ix]
            for train, test in cv.split(X_, y_):
                X_test, y_test, meta_test = X_[test], y_[test], meta_.loc[test]

                yield X_[train], X_[test]

# Split done for the Cross Session evaluation
def eval_split_cross_session():
    for subject in dataset.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        groups = metadata.session.values
        cv = LeaveOneGroupOut()
        for _, test in cv.split(X, y, groups):
            metadata_test = metadata.loc[test]
            runs = metadata_test.run.values
            for r in np.unique(runs):
                ix = runs == r
                yield X[test[ix]]

def pseudo_split_cross_session():
    for subject in dataset.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        groups = metadata.session.values
        cv = LeaveOneGroupOut()
        for _, test in cv.split(X, y, groups):
            metadata_test = metadata.loc[test]
            runs = metadata_test.run.values
            ix = runs == runs[0]
            yield X[test[ix]]


# Split done for the Cross Subject evaluation
def eval_split_cross_subject():
    X, y, metadata = paradigm.get_data(dataset=dataset)
    groups = metadata.subject.values
    cv = LeaveOneGroupOut()
    for _, test in cv.split(X, y, groups):
        metadata_test = metadata.loc[test]
        sessions = metadata_test.session.values
        for sess in np.unique(sessions):
            ix = sessions == sess
            yield X[test[ix]]

# Split done for the Cross Subject evaluation
def pseudo_split_cross_subject():
    X, y, metadata = paradigm.get_data(dataset=dataset)
    groups = metadata.subject.values
    cv = LeaveOneGroupOut()
    for _, test in cv.split(X, y, groups):
        metadata_test = metadata.loc[test]
        sessions = metadata_test.session.values
        runs = metadata_test.run.values
        for sess in np.unique(sessions):
            ix = sessions == sess
            X_sess, metadata_sess = X[test[ix]], metadata_test.loc[test[ix]].reset_index(drop=True)

            runs_in_session = metadata_sess.run.values
            # yield just calibration part
            yield X_sess[runs_in_session == runs_in_session[0]]


@pytest.mark.parametrize("split", [CrossSubjectSplitter, CrossSessionSplitter])
def test_offline(split):
    X, y, metadata = paradigm.get_data(dataset=dataset)

    run = True if isinstance(split, CrossSessionSplitter) else False

    if isinstance(split, CrossSessionSplitter):
        eval_split = eval_split_cross_session
    else:
        eval_split = eval_split_cross_subject

    split = split()
    metasplit = OfflineSplit(run=run)

    Tests = []
    for (_,test) in split.split(X, y, metadata):
        X_test, y_test, metadata_test = X[test], y[test], metadata.loc[test]
        for i, (test_index) in enumerate(metasplit.split(X_test, y_test, metadata_test)):
            Tests.append(X[test_index])

    for ix, X_test_t in enumerate(eval_split()):
            # Check if the output is the same as the input
            assert np.array_equal(Tests[ix], X_test_t)


@pytest.mark.parametrize("split", [CrossSubjectSplitter, CrossSessionSplitter])
def test_pseudoonline(split):
    X, y, metadata = paradigm.get_data(dataset=dataset)

    if isinstance(split, CrossSessionSplitter):
        eval_split = pseudo_split_cross_session
    else:
        eval_split = pseudo_split_cross_subject

    split = split()
    metasplit = PseudoOnlineSplit()

    Tests = []
    for (_,test) in split.split(X, y, metadata):
        X_test, y_test, metadata_test = X[test], y[test], metadata.loc[test]
        for i, (_,calib_index) in enumerate(metasplit.split(X_test, y_test, metadata_test)):
            Tests.append(X[calib_index])

    for ix, X_calib_t in enumerate(eval_split()):
            # Check if the output is the same as the input
            assert np.array_equal(Tests[ix], X_calib_t)

@pytest.mark.skip(reason="Still working on that")
def test_sampler():
    X, y, metadata = paradigm.get_data(dataset=dataset)
    data_size = dict(policy="per_class", value=np.array([5, 10, 30, 50]))

    split = SamplerSplit()

    for ix, ((X_train_t, X_test_t), (train, test)) in enumerate(
            zip(eval_split_cross_subject(), split.split(X, y, metadata))):
        X_train, X_test = X[train], X[test]

        # Check if the output is the same as the input
        assert np.array_equal(X_train, X_train_t)
        assert np.array_equal(X_test, X_test_t)

