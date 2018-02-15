from time import time
import numpy as np

from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

from mne.epochs import concatenate_epochs, equalize_epoch_counts

from .base import BaseEvaluation

class TrainTestEvaluation(BaseEvaluation):

    def extract_data_from_cont(self, ep_list, event_id):
        skip=False
        event_epochs = dict(zip(event_id.keys(), [[]] * len(event_id)))
        for epoch in ep_list:
            for key in event_id.keys():
                if key in epoch.event_id.keys():
                    event_epochs[key].append(epoch[key])
        all_events = []
        for key in event_id.keys():
            if len(event_epochs[key]) > 0 :
                all_events.append(concatenate_epochs(event_epochs[key]))
        # equalize for accuracy
        if len(all_events) > 1:
            equalize_epoch_counts(all_events)
        ep = concatenate_epochs(all_events)
        # previously multipled data by 1e6
        X, y = (ep.get_data(), ep.events[:, -1])
        return X, y
    
    def score(self, clf, X, y, scoring):
        cv = KFold(5, shuffle=True, random_state=self.random_state)

        y_uniq = np.unique(y)
        if len(y_uniq) == 2:
            for ind, l in enumerate(y_uniq):
                y[y==l] = ind
        acc = cross_val_score(clf, X, y, cv=cv,
                              scoring=scoring, n_jobs=self.n_jobs)
        return acc.mean()

class CrossSubjectEvaluation(TrainTestEvaluation):
    """Cross Subject evaluation Context.

    Evaluate performance of the pipeline trained on all subjects but one,
    concatenating sessions.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.

    See Also
    --------
    BaseContext
    """
    def evaluate(self, dataset, subject, clf, paradigm):
        """Prepare data for classification."""
        # requires that subject be an int
        s = subject-1
        self.ind_cache[s] = self.ind_cache[s]*0
        allX = np.concatenate(self.X_cache)
        ally = np.concatenate(self.y_cache)
        groups = np.concatenate(self.ind_cache)
        # re-generate s group label
        self.ind_cache[s] = np.ones(self.ind_cache[s].shape)
        t_start = time()
        score = self.score(clf, allX, ally, groups, paradigm.scoring)
        duration = time() - t_start
        return {'time': duration, 'dataset': dataset.code, 'id': subject, 'score': score, 'n_samples': len(ally)}

    def preprocess_data(self, d, paradigm):
        assert len(d.subject_list) > 1, "Dataset {} has only one subject".format(d.code)
        self.X_cache = []
        event_id = d.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))
        self.y_cache = []
        self.ind_cache = []
        for s in d.subject_list:
            sub = d.get_data([s], stack_sessions=True)[0]
            # get all epochs for individual files in given subject
            epochs = paradigm._epochs(sub, event_id, d.interval)
            # equalize events from different classes
            X, y = self.extract_data_from_cont(epochs, event_id)
            self.X_cache.append(X)
            self.y_cache.append(y)
            self.ind_cache.append(np.ones(y.shape))
        
    def score(self, clf, X, y, groups, scoring):
        y_uniq = np.unique(y)
        if len(y_uniq) == 2:
            for ind, l in enumerate(y_uniq):
                y[y==l] = ind
        acc = cross_val_score(clf, X, y, cv=[(np.nonzero(groups==1)[0], np.nonzero(groups==0)[0])],
                              scoring=scoring, n_jobs=self.n_jobs)
        return acc.mean()

class WithinSessionEvaluation(TrainTestEvaluation):
    """Within session evaluation Context.

    """
    def evaluate(self, dataset, subject, clf, paradigm):
        """Prepare data for classification."""
        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))

        sub = dataset.get_data([subject], stack_sessions=False)[0]
        results = []
        for ind, session in enumerate(sub):
            skip=  False
            sess_id = '{:03d}_{:d}'.format(subject, ind)

            # get all epochs for individual files in given session
            epochs = paradigm._epochs(session, event_id, dataset.interval)
            X, y = self.extract_data_from_cont(epochs, event_id)
            if len(np.unique(y))>1:
                t_start = time()
                score = self.score(clf, X, y, paradigm.scoring)
                duration = time() - t_start
                results.append({'time': duration, 'dataset': dataset.code,
                                'id': sess_id, 'score': score, 'n_samples': len(y)})
        return results


class CrossSessionEvaluation(TrainTestEvaluation):
    """Cross session Context.

    Evaluate performance of the pipeline across sessions,

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.

    See Also
    --------
    BaseContext
    """

    def evaluate(self, dataset, subject, clf, paradigm):
        """Prepare data for classification."""
        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))

        sub = dataset.get_data([subject], stack_sessions=False)[0]
        results = []
        listX, listy = ([], [])
        for ind, session in enumerate(sub):
            # get list epochs for individual files in given session
            epochs = paradigm._epochs(session, event_id, dataset.interval)
            # equalize events from different classes
            X, y  = self.extract_data_from_cont(epochs, event_id)
            listX.append(X)
            listy.append(y)
        groups = []
        for ind, y in enumerate(listy):
            groups.append(np.ones((len(y),)) * ind)
        allX = np.concatenate(listX, axis=0)
        ally = np.concatenate(listy, axis=0)
        groupvec = np.concatenate(groups, axis=0)
        t_start = time()
        score = self.score(clf, allX, ally, groupvec, paradigm.scoring)
        duration = time() - t_start
        return {'time': duration,
                'dataset': dataset.code,
                'id': subject,
                'score': score,
                'n_samples': len(y)}

    def preprocess_data(self, d, paradigm):
        assert d.n_sessions > 1, "Proposed dataset {} has only one session".format(
            d.code)

    def score(self, clf, X, y, groups, scoring):
        y_uniq = np.unique(y)
        if len(y_uniq) == 2:
            for ind, l in enumerate(y_uniq):
                y[y==l] = ind
        acc = cross_val_score(clf, X, y, groups=groups,cv=LeaveOneGroupOut(),
                              scoring=scoring, n_jobs=self.n_jobs)
        return acc.mean()
