import logging
from copy import deepcopy
from time import time
from typing import Optional, Union

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder

from moabb.evaluations.base import BaseEvaluation


log = logging.getLogger()

# TODO Base class for WithinSession/IncreasingData
# TODO implement per class sampling. Not yet clear -> how to sample test set?


class WithinSessionEvaluation(BaseEvaluation):
    """Within Session evaluation."""

    def __init__(
        self,
        n_perms: Union[int, np.ndarray] = 1,
        data_size: Optional[dict] = None,
        **kwargs,
    ):
        """
        :param n_perms: Number of permutations to perform. If an array
            is passed it has to be equal in size to the data_size array.
        :param data_size: Contains the policy to pick the datasizes to
            evaluate, as well as the actual values. The dict has the
            key 'policy' with either 'ratio' or 'per_class', and the key
            'value' with the actual values as an numpy array.
        """
        self.data_size = data_size
        self.n_perms = n_perms
        self.calculate_learning_curve = self.data_size is not None
        if self.calculate_learning_curve:
            if type(n_perms) is int:
                self.n_perms = np.full_like(self.data_size["value"], n_perms, dtype=int)
            if len(self.n_perms) != len(self.data_size["value"]):
                raise ValueError(
                    "Number of elements in n_perms must be equal "
                    "to number of elements in data_size['value']"
                )
            self.test_size = 0.2  # Roughly similar to 5-fold CV
            add_cols = ["data_size", "permutation"]
            super().__init__(additional_columns=add_cols, **kwargs)
        else:
            # Perform default within session evaluation
            super().__init__(**kwargs)

    def _evaluate(self, dataset, pipelines):
        for subject in dataset.subject_list:
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)
            if len(run_pipes) == 0:
                continue

            # get the data
            X, y, metadata = self.paradigm.get_data(dataset, [subject])

            # iterate over sessions
            for session in np.unique(metadata.session):
                ix = metadata.session == session

                for name, clf in run_pipes.items():

                    t_start = time()
                    cv = StratifiedKFold(5, shuffle=True, random_state=self.random_state)

                    le = LabelEncoder()
                    y = le.fit_transform(y[ix])
                    acc = cross_val_score(
                        clf,
                        X[ix],
                        y,
                        cv=cv,
                        scoring=self.paradigm.scoring,
                        n_jobs=self.n_jobs,
                        error_score=self.error_score,
                    )
                    score = acc.mean()
                    duration = time() - t_start
                    res = {
                        "time": duration / 5.0,  # 5 fold CV
                        "dataset": dataset,
                        "subject": subject,
                        "session": session,
                        "score": score,
                        "n_samples": len(y[ix]),  # not training sample
                        "n_channels": X.shape[1],
                        "pipeline": name,
                    }

                    yield res

    def get_data_size_steps(self, n_epochs):
        if self.data_size is None:
            return None
        if self.data_size["policy"] == "ratio":
            return np.ceil(self.data_size["value"] * n_epochs).astype(int)
        elif self.data_size["policy"] == "per_class":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown policy {self.data_size['policy']}")

    def score_explicit(self, clf, X_train, y_train, X_test, y_test):
        scorer = get_scorer(self.paradigm.scoring)
        t_start = time()
        try:
            model = clf.fit(X_train, y_train)
            score = _score(model, X_test, y_test, scorer)
        except ValueError as e:
            if self.error_score == "raise":
                raise e
            score = self.error_score
        duration = time() - t_start
        return score, duration

    def _evaluate_learning_curve(self, dataset, pipelines):
        for subject in dataset.subject_list:
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)
            if len(run_pipes) == 0:
                continue

            # get the data
            X_all, y_all, metadata_all = self.paradigm.get_data(dataset, [subject])
            le = LabelEncoder()
            y_all = le.fit_transform(y_all)
            # shuffle_data = True if self.n_perms > 1 else False
            for session in np.unique(metadata_all.session):
                sess_idx = metadata_all.session == session
                X_sess = X_all[sess_idx]
                y_sess = y_all[sess_idx]
                # metadata_sess = metadata_all[sess_idx]
                n_epochs = np.sum(sess_idx)
                sss = StratifiedShuffleSplit(
                    n_splits=self.n_perms[0], test_size=self.test_size
                )
                for perm_i, (train_idx, test_idx) in enumerate(sss.split(X_sess, y_sess)):
                    X_train_all = X_sess[train_idx]
                    y_train_all = y_sess[train_idx]
                    X_test = X_sess[test_idx]
                    y_test = y_sess[test_idx]
                    data_size_steps = self.get_data_size_steps(n_epochs)
                    for di, data_size in enumerate(data_size_steps):
                        if perm_i >= self.n_perms[di]:
                            continue
                        not_enough_data = False
                        log.info(
                            f"Permutation: {perm_i+1}," f" Training samples: {data_size}"
                        )

                        X_train = X_train_all[:data_size, :]
                        y_train = y_train_all[:data_size]
                        # metadata = metadata_perm[:data_size]

                        if len(np.unique(y_train)) < 2:
                            log.warning(
                                "For current data size, only one class" "would remain."
                            )
                            not_enough_data = True
                        for name, clf in run_pipes.items():
                            res = {
                                "dataset": dataset,
                                "subject": subject,
                                "session": session,
                                "n_samples": len(y_train),
                                "n_channels": X_train.shape[1],
                                "pipeline": name,
                                # Additional columns
                                "data_size": data_size,
                                "permutation": perm_i + 1,
                            }
                            if not_enough_data:
                                res["time"] = 0
                                res["score"] = np.nan
                            else:
                                res["score"], res["time"] = self.score_explicit(
                                    deepcopy(clf), X_train, y_train, X_test, y_test
                                )
                            yield res

    def evaluate(self, dataset, pipelines):
        if self.calculate_learning_curve:
            yield from self._evaluate_learning_curve(dataset, pipelines)
        else:
            yield from self._evaluate(dataset, pipelines)

    def is_valid(self, dataset):
        return True


class CrossSessionEvaluation(BaseEvaluation):
    """Cross session Context.

    Evaluate performance of the pipeline across sessions but for a single
    subject. Verifies that sufficient sessions are there for this to be
    reasonable

    """

    def evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        for subject in dataset.subject_list:
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)
            if len(run_pipes) == 0:
                continue

            # get the data
            X, y, metadata = self.paradigm.get_data(dataset, [subject])
            le = LabelEncoder()
            y = le.fit_transform(y)
            groups = metadata.session.values
            scorer = get_scorer(self.paradigm.scoring)

            for name, clf in run_pipes.items():

                # we want to store a results per session
                cv = LeaveOneGroupOut()
                for train, test in cv.split(X, y, groups):
                    t_start = time()
                    score = _fit_and_score(
                        clone(clf),
                        X,
                        y,
                        scorer,
                        train,
                        test,
                        verbose=False,
                        parameters=None,
                        fit_params=None,
                        error_score=self.error_score,
                    )[0]
                    duration = time() - t_start
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": groups[test][0],
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": X.shape[1],
                        "pipeline": name,
                    }
                    yield res

    def is_valid(self, dataset):
        return dataset.n_sessions > 1


class CrossSubjectEvaluation(BaseEvaluation):
    """Cross Subject evaluation Context.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    """

    def evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        # this is a bit akward, but we need to check if at least one pipe
        # have to be run before loading the data. If at least one pipeline
        # need to be run, we have to load all the data.
        # we might need a better granularity, if we query the DB
        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(self.results.not_yet_computed(pipelines, dataset, subject))
        if len(run_pipes) != 0:

            # get the data
            X, y, metadata = self.paradigm.get_data(dataset)

            # encode labels
            le = LabelEncoder()
            y = le.fit_transform(y)

            # extract metadata
            groups = metadata.subject.values
            sessions = metadata.session.values

            scorer = get_scorer(self.paradigm.scoring)

            # perform leave one subject out CV
            cv = LeaveOneGroupOut()
            for train, test in cv.split(X, y, groups):

                subject = groups[test[0]]
                # now we can check if this subject has results
                run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)

                # iterate over pipelines
                for name, clf in run_pipes.items():
                    t_start = time()
                    model = deepcopy(clf).fit(X[train], y[train])
                    duration = time() - t_start

                    # we eval on each session
                    for session in np.unique(sessions[test]):
                        ix = sessions[test] == session
                        score = _score(model, X[test[ix]], y[test[ix]], scorer)

                        res = {
                            "time": duration,
                            "dataset": dataset,
                            "subject": subject,
                            "session": session,
                            "score": score,
                            "n_samples": len(train),
                            "n_channels": X.shape[1],
                            "pipeline": name,
                        }

                        yield res

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1
