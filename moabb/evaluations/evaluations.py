import logging
from time import time

import numpy as np
from copy import deepcopy
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from moabb.evaluations.base import BaseEvaluation
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.metrics import get_scorer

log = logging.getLogger()


class WithinSessionEvaluation(BaseEvaluation):
    """Within session evaluation

    returns Score computed within each recording session

    """

    def evaluate(self, dataset, pipelines):
        """Prepare data for classification."""

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
                           'id': subject,
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
                              n_jobs=self.n_jobs)
        return acc.mean()

    def verify(self, dataset):
        pass


class CrossSessionEvaluation(BaseEvaluation):
    """Cross session Context.

    Evaluate performance of the pipeline across sessions but for a single
    subject. Verifies that sufficient sessions are there for this to be
    reasonable

    """

    def evaluate(self, dataset, pipelines):
        self.verify(dataset)
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
                    score = _fit_and_score(clf, X, y, scorer, train, test,
                                           verbose=False, parameters=None,
                                           fit_params=None)[0]
                    duration = time() - t_start
                    res = {'time': duration,
                           'dataset': dataset,
                           'id': subject,
                           'session': groups[test][0],
                           'score': score,
                           'n_samples': len(train),
                           'n_channels': X.shape[1],
                           'pipeline': name}
                    yield res

    def verify(self, dataset):
        assert dataset.n_sessions > 1


class CrossSubjectEvaluation(BaseEvaluation):
    """Cross Subject evaluation Context.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    """

    def evaluate(self, dataset, pipelines):
        self.verify(dataset)
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
                               'id': subject,
                               'session': session,
                               'score': score,
                               'n_samples': len(train),
                               'n_channels': X.shape[1],
                               'pipeline': name}

                        yield res

    def verify(self, dataset):
        assert len(dataset.subject_list) > 1
