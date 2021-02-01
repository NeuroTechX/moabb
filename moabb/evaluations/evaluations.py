import logging
from time import time

import numpy as np
from copy import deepcopy
from sklearn.model_selection import (cross_val_score, LeaveOneGroupOut,
                                     StratifiedKFold, StratifiedShuffleSplit)
from sklearn.preprocessing import LabelEncoder

from moabb.evaluations.base import BaseEvaluation
from sklearn.base import clone
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.metrics import get_scorer

log = logging.getLogger()

# TODO Base class for WithinSession/IncreasingData

class WithinSessionEvaluationIncreasingData(BaseEvaluation):
    """Within session evaluation with increasing data

    returns Score computed within each recording session with x-% of data

    """
# n_perms = 1, datasize=1.0 -> should result in same results as WithinSession
    def __init__(self, n_perms=20, datasize=None, **kwargs):
        self.n_perms = n_perms
        self.k_folds = 5
        if datasize is None:
            # This is only training data ratio
            # 100% of the training data are the (4 training folds) of 5-fold
            self.datasize = dict(ratio=np.geomspace(0.05, 1, 20))
            # TODO indicate how many samples per class, e.g. 20 left 20 right
            # example:
            #self.datasize = dict('per_class', np.round(np.geomspace(10, 100)))
            # optionally / later
            # self.datasize = dict('absolute', np.geomspace(10, 100))
        super().__init__(**kwargs)

    def evaluate(self, dataset, pipelines):
        for subject in dataset.subject_list:
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject
            )
            if len(run_pipes) == 0:
                continue

            # get the data
            X_all, y_all, metadata_all = self.paradigm.get_data(
                dataset, [subject]
            )
            shuffle_data = True if self.n_perms > 1 else False

            for session in np.unique(metadata_all.session):
                sess_idx = metadata_all.session == session
                X_sess = X_all[sess_idx]
                y_sess = y_all[sess_idx]
                # metadata_sess = metadata_all[sess_idx]
                n_epochs = len(sess_idx)

                # FIXME START Something like this
                StratifiedShuffleSplit(n_splits=self.n_perms, train_size='5perc',
                                       test_size=None)
                # END

                # TODO Split Train / validate here into 5 folds
                for perm in range(self.n_perms):
                    if shuffle_data:
                        perm_idx = np.array(range(len(y_all)))
                        # TODO Check if there is a scikit-learn implementation
                        for c in np.unique(y_all):
                            c_idx = np.where(y_all == c)[0]
                            perm_idx[c_idx[:]] = np.random.permutation(c_idx)
                    else:
                        perm_idx = np.array(range(len(y_all)))
                    X_perm = X_sess[perm_idx]
                    y_perm = y_sess[perm_idx]
                    # metadata_perm = metadata_sess.iloc[perm_idx]

                    data_size_steps = np.ceil(
                        self.datasize * n_epochs
                    ).astype(np.int)
                    check_for_enough_epochs = True
                    for data_size in data_size_steps:
                        not_enough_data = False
                        log.info(f"Data size: {data_size}")
                        if len(X_all) < data_size:
                            break
                        X = X_perm[:data_size, :]
                        y = y_perm[:data_size]
                        # metadata = metadata_perm[:data_size]

                        if len(np.unique(y)) < 2:
                            log.warning(
                                "For current data size, only one class"
                                "would remain."
                            )
                            not_enough_data = True
                        if check_for_enough_epochs:
                            _, n_per_class = np.unique(y, return_counts=True)
                            if np.min(n_per_class) < self.k_folds:
                                log.warning(
                                    f"For current data subset size, the"
                                    f"smallest class has fewer entries"
                                    f"({np.min(n_per_class)}) than folds"
                                    f"({self.k_folds})."
                                )
                                not_enough_data = True
                        for name, clf in run_pipes.items():
                            # Store additionally: datasize, perm, fold
                            res = {
                                "dataset": dataset,
                                "subject": subject,
                                # This session name is needed as long as
                                # additional columns are not possible (PR #127)
                                "session": f"{session}_p{perm}_d({data_size}_f({{fold}}))",
                                "n_samples": len(y),  # not training sample
                                "n_channels": X.shape[1],
                                "pipeline": name,
                            }
                            if not_enough_data:
                                res["time"] = 0
                                res["score"] = np.nan
                            else:
                                t_start = time()
                                score = self.score(
                                    clf, X, y, self.paradigm.scoring
                                )
                                duration = time() - t_start
                                res["time"] = duration / self.k_folds
                                res["score"] = score

                            yield res

    def score(self, clf, X, y, scoring):
        # TODO change to _fit_and_score
        # cv = StratifiedKFold(
        #     self.k_folds, shuffle=True, random_state=self.random_state
        # )

        le = LabelEncoder()
        y = le.fit_transform(y)
        try:
            acc = cross_val_score(
                clf,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                error_score=self.error_score,
            ).mean()
        except ValueError as e:
            if self.error_score == "raise":
                raise e
            elif self.error_score is np.nan:
                acc = np.nan
        return acc

    def is_valid(self, dataset):
        return True


class WithinSessionEvaluation(BaseEvaluation):
    """Within session evaluation

    returns Score computed within each recording session

    """

    def evaluate(self, dataset, pipelines):
        for subject in dataset.subject_list:
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(pipelines,
                                                      dataset,
                                                      subject)
            if len(run_pipes) == 0:
                continue

            # get the data
            X, y, metadata = self.paradigm.get_data(dataset, [subject])

            # iterate over sessions
            for session in np.unique(metadata.session):
                ix = metadata.session == session

                for name, clf in run_pipes.items():

                    t_start = time()
                    score = self.score(clf, X[ix], y[ix],
                                       self.paradigm.scoring)
                    duration = time() - t_start
                    res = {'time': duration / 5.,  # 5 fold CV
                           'dataset': dataset,
                           'subject': subject,
                           'session': session,
                           'score': score,
                           'n_samples': len(y[ix]),  # not training sample
                           'n_channels': X.shape[1],
                           'pipeline': name}

                    yield res

    def score(self, clf, X, y, scoring):
        cv = StratifiedKFold(5, shuffle=True, random_state=self.random_state)

        le = LabelEncoder()
        y = le.fit_transform(y)
        acc = cross_val_score(clf, X, y, cv=cv, scoring=scoring,
                              n_jobs=self.n_jobs, error_score=self.error_score)
        return acc.mean()

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
            raise AssertionError('Dataset is not appropriate for evaluation')
        for subject in dataset.subject_list:
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(pipelines,
                                                      dataset,
                                                      subject)
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
                    score = _fit_and_score(clone(clf), X, y, scorer, train,
                                           test, verbose=False,
                                           parameters=None,
                                           fit_params=None,
                                           error_score=self.error_score)[0]
                    duration = time() - t_start
                    res = {'time': duration,
                           'dataset': dataset,
                           'subject': subject,
                           'session': groups[test][0],
                           'score': score,
                           'n_samples': len(train),
                           'n_channels': X.shape[1],
                           'pipeline': name}
                    yield res

    def is_valid(self, dataset):
        return (dataset.n_sessions > 1)


class CrossSubjectEvaluation(BaseEvaluation):
    """Cross Subject evaluation Context.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    """

    def evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError('Dataset is not appropriate for evaluation')
        # this is a bit akward, but we need to check if at least one pipe
        # have to be run before loading the data. If at least one pipeline
        # need to be run, we have to load all the data.
        # we might need a better granularity, if we query the DB
        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(self.results.not_yet_computed(pipelines,
                                                           dataset,
                                                           subject))
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
                run_pipes = self.results.not_yet_computed(pipelines, dataset,
                                                          subject)

                # iterate over pipelines
                for name, clf in run_pipes.items():
                    t_start = time()
                    model = deepcopy(clf).fit(X[train], y[train])
                    duration = time() - t_start

                    # we eval on each session
                    for session in np.unique(sessions[test]):
                        ix = sessions[test] == session
                        score = _score(model, X[test[ix]], y[test[ix]], scorer)

                        res = {'time': duration,
                               'dataset': dataset,
                               'subject': subject,
                               'session': session,
                               'score': score,
                               'n_samples': len(train),
                               'n_channels': X.shape[1],
                               'pipeline': name}

                        yield res

    def is_valid(self, dataset):
        return (len(dataset.subject_list) > 1)
