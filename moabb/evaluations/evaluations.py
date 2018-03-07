from time import time
import numpy as np
import logging

from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold
from sklearn.preprocessing import LabelEncoder

from mne.epochs import concatenate_epochs, equalize_epoch_counts

from moabb.evaluations.base import BaseEvaluation

log = logging.getLogger()

class TrainTestEvaluation(BaseEvaluation):
    '''
    Base class for evaluations that split data into undifferentiated train/test
    (not compatible with multi-task/transfer/multi-dataset schemes)
    '''

    def extract_data_from_cont(self, ep_list, event_id):
        event_epochs = {key: [] for key in event_id.keys()}
        for epoch in ep_list:
            for key in event_id.keys():
                if key in epoch.event_id.keys():
                    event_epochs[key].append(epoch[key])
        all_events = []
        for key in event_id.keys():
            if len(event_epochs[key]) > 0:
                all_events.append(concatenate_epochs(event_epochs[key]))
        # equalize for accuracy
        if len(all_events) > 1:
            equalize_epoch_counts(all_events)
        ep = concatenate_epochs(all_events)
        # previously multipled data by 1e6
        X, y = (ep.get_data(), ep.events[:, -1])
        return X, y


class CrossSubjectEvaluation(TrainTestEvaluation):
    """Cross Subject evaluation Context.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    Parameters
    ----------
    random_state
    n_jobs


    """

    def evaluate(self, dataset, subject, pipelines):
        # requires that subject be an int
        s = subject-1
        self.ind_cache[s] = self.ind_cache[s]*0
        allX = np.concatenate(self.X_cache)
        ally = np.concatenate(self.y_cache)
        groups = np.concatenate(self.ind_cache)
        # re-generate s group label
        self.ind_cache[s] = np.ones(self.ind_cache[s].shape)
        out = {}
        for name, clf in pipelines.items():
            t_start = time()
            score = self.score(clf, allX, ally, groups, self.paradigm.scoring)
            duration = time() - t_start
            out[name] = {'time': duration,
                         'dataset': dataset,
                         'id': subject,
                         'score': score,
                         'n_samples': groups.sum(),
                         'n_channels': allX.shape[1]}
        return out

    def preprocess_data(self, dataset):
        assert len(dataset.subject_list) > 1, "Dataset {} has only one subject".format(
            dataset.code)
        self.X_cache = []
        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))
        self.y_cache = []
        self.ind_cache = []
        for s in dataset.subject_list:
            sub = dataset.get_data([s], stack_sessions=True)[0]
            # get all epochs for individual files in given subject
            epochs = self.paradigm._epochs(sub, event_id, dataset.interval)
            # equalize events from different classes
            X, y = self.extract_data_from_cont(epochs, event_id)
            self.X_cache.append(X)
            self.y_cache.append(y)
            self.ind_cache.append(np.ones(y.shape))

    def score(self, clf, X, y, groups, scoring):
        le = LabelEncoder()
        y = le.fit_transform(y)
        acc = cross_val_score(clf, X, y, cv=[(np.nonzero(groups == 1)[0],
                                              np.nonzero(groups == 0)[0])],
                              scoring=scoring, n_jobs=self.n_jobs)
        return acc.mean()


class WithinSessionEvaluation(TrainTestEvaluation):
    """Within session evaluation, returns accuracy computed within each recording session

    """

    def evaluate(self, dataset, subject, pipelines):
        """Prepare data for classification."""
        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))

        sub = dataset.get_data([subject], stack_sessions=False)[0]
        out = {k: [] for k in pipelines.keys()}
        for ind, session in enumerate(sub):
            skip = False
            # sess_id = '{:03d}_{:d}'.format(subject, ind)

            # get all epochs for individual files in given session
            epochs = self.paradigm._epochs(session, event_id, dataset.interval)
            X, y = self.extract_data_from_cont(epochs, event_id)
            if len(np.unique(y)) > 1:
                counts = np.unique(y,return_counts=True)[1]
                log.debug('score imbalance: {}'.format(counts))
                for name, clf in pipelines.items():
                    t_start = time()
                    score = self.score(clf, X, y, self.paradigm.scoring)
                    duration = time() - t_start
                    out[name].append({'time': duration,
                                      'dataset': dataset,
                                      'id': subject,
                                      'score': score,
                                      'n_samples': len(y),
                                      'n_channels': X.shape[1]})
        return out

    def score(self, clf, X, y, scoring):
        cv = KFold(5, shuffle=True, random_state=self.random_state)

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


class CrossSessionEvaluation(TrainTestEvaluation):
    """Cross session Context.

    Evaluate performance of the pipeline across sessions but for a single subject.
    Verifies that sufficient sessions are there for this to be reasonable

    """

    def evaluate(self, dataset, subject, pipelines):
        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))

        sub = dataset.get_data([subject], stack_sessions=False)[0]
        listX, listy = ([], [])
        for ind, session in enumerate(sub):
            # get list epochs for individual files in given session
            epochs = self.paradigm._epochs(session, event_id, dataset.interval)
            # equalize events from different classes
            X, y = self.extract_data_from_cont(epochs, event_id)
            listX.append(X)
            listy.append(y)
        groups = []
        for ind, y in enumerate(listy):
            groups.append(np.ones((len(y),)) * ind)
        allX = np.concatenate(listX, axis=0)
        ally = np.concatenate(listy, axis=0)
        groupvec = np.concatenate(groups, axis=0)
        out = {}
        for name, clf in pipelines.items():
            t_start = time()
            score = self.score(clf, allX, ally, groupvec, self.paradigm.scoring)
            duration = time() - t_start
            out[name] = {'time': duration,
                         'dataset': dataset,
                         'id': subject,
                         'score': score,
                         'n_samples': len(y),
                         'n_channels':allX.shape[1]}
        return out

    def preprocess_data(self, dataset):
        assert dataset.n_sessions > 1, "Proposed dataset {} has only one session".format(
            dataset.code)

    def score(self, clf, X, y, groups, scoring):
        le = LabelEncoder()
        y = le.fit_transform(y)
        acc = cross_val_score(clf, X, y, groups=groups, cv=LeaveOneGroupOut(),
                              scoring=scoring, n_jobs=self.n_jobs)
        return acc.mean()
