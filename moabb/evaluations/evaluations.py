import logging
from time import time

import numpy as np
from copy import deepcopy
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from moabb.evaluations.base import BaseEvaluation
from sklearn.model_selection._validation import _fit_and_score
from sklearn.metrics import get_scorer

log = logging.getLogger()


class WithinSessionEvaluation(BaseEvaluation):
    """Within session evaluation, returns accuracy computed within each recording session

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
                    res = {'time': duration,
                           'dataset': dataset,
                           'id': subject,
                           'session': session,
                           'score': score,
                           'n_samples': len(y[ix]),
                           'n_channels': X.shape[1]}
                    self.push_result({name: res}, pipelines)

    def score(self, clf, X, y, scoring):
        cv = StratifiedKFold(5, shuffle=True, random_state=self.random_state)

        le = LabelEncoder()
        y = le.fit_transform(y)
        acc = cross_val_score(clf, X, y, cv=cv,
                              scoring=scoring, n_jobs=self.n_jobs)
        return acc.mean()

    def preprocess_data(self, dataset):
        '''
        Optional paramter if any sort of dataset-wide computation is needed
        per subject
        '''
        pass


class CrossSessionEvaluation(BaseEvaluation):
    """Cross session Context.

    Evaluate performance of the pipeline across sessions but for a single subject.
    Verifies that sufficient sessions are there for this to be reasonable

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
            le = LabelEncoder()
            y = le.fit_transform(y)
            groups = metadata.session.values

            for name, clf in run_pipes.items():

                # we want to store a results per session
                cv = LeaveOneGroupOut()
                for train, test in cv.split(X, y, groups):
                    t_start = time()
                    scorer = get_scorer(self.paradigm.scoring)
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
                           'n_channels': X.shape[1]}
                    self.push_result({name: res}, pipelines)

    def score(self, clf, X, y, groups, scoring):
        pass


class CrossSubjectEvaluation(BaseEvaluation):
    """Cross Subject evaluation Context.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    Parameters
    ----------
    random_state
    n_jobs


    """

    def evaluate(self, dataset, pipelines):
        # check if we already have result for this subject/pipeline
        # we might need a better granularity, if we query the DB
        run_pipes = []
        for subject in dataset.subject_list:
            run_pipes += self.results.not_yet_computed(pipelines,
                                                       dataset,
                                                       subject)
        if len(run_pipes) == 0:

            # get the data
            X, y, metadata = self.paradigm.get_data(dataset)
            le = LabelEncoder()
            y = le.fit_transform(y)
            groups = metadata.subject.values

            for name, clf in run_pipes.items():

                # we want to store a results per session
                cv = LeaveOneGroupOut()
                for train, test in cv.split(X, y, groups):
                    t_start = time()
                    scorer = get_scorer(self.paradigm.scoring)
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
                           'n_channels': X.shape[1]}
                    self.push_result({name: res}, pipelines)

    def score(self, clf, X, y, groups, scoring):
        pass
