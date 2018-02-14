from time import time
import numpy as np

from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

from mne.epochs import concatenate_epochs, equalize_epoch_counts

from .base import BaseEvaluation

class WithinSubjectEvaluation(BaseEvaluation):
    """Within Subject evaluation Context.

    Evaluate performance of the pipeline on each subject independently,
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
        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))

        sub = dataset.get_data([subject], stack_sessions=True)[0]
        # get all epochs for individual files in given subject
        epochs = paradigm._epochs(sub, event_id, dataset.interval)
        # equalize events from different classes
        event_epochs = dict(zip(event_id.keys(), [[]] * len(event_id)))
        for epoch in epochs:
            for key in event_id.keys():
                if key in epoch.event_id.keys():
                    event_epochs[key].append(epoch[key])
        for key in event_id.keys():
            event_epochs[key] = concatenate_epochs(event_epochs[key])

        # equalize for accuracy
        equalize_epoch_counts(list(event_epochs.values()))
        ep = concatenate_epochs(list(event_epochs.values()))
        # previously multipled data by 1e6
        X, y = (ep.get_data(), ep.events[:, -1])
        t_start = time()
        score = self.score(clf, X, y, paradigm.scoring)
        duration = time() - t_start
        return {'time': duration, 'dataset': dataset.code, 'id': subject, 'score': score, 'n_samples': len(y)}

    def score(self, clf, X, y, scoring):
        cv = KFold(5, shuffle=True, random_state=self.random_state)

        y_uniq = np.unique(y)
        if len(y_uniq) == 2:
            for ind, l in enumerate(y_uniq):
                y[y==l] = ind
        acc = cross_val_score(clf, X, y, cv=cv,
                              scoring=scoring, n_jobs=self.n_jobs)
        return acc.mean()


class WithinSessionEvaluation(WithinSubjectEvaluation):
    """Within Subject evaluation Context.

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
            # equalize events from different classes
            event_epochs = dict(zip(event_id.keys(), [[]] * len(event_id)))
            for epoch in epochs:
                for key in event_id.keys():
                    if key in epoch.event_id.keys():
                        event_epochs[key].append(epoch[key])
            for key in event_id.keys():
                if len(event_epochs[key]) == 0 :
                    skip = True
                    continue
                event_epochs[key] = concatenate_epochs(event_epochs[key])

            if not skip:
                # equalize for accuracy
                equalize_epoch_counts(list(event_epochs.values()))
                ep = concatenate_epochs(list(event_epochs.values()))
                # previously multipled data by 1e6
                X, y = (ep.get_data(), ep.events[:, -1])
                t_start = time()
                score = self.score(clf, X, y, paradigm.scoring)
                duration = time() - t_start
                results.append({'time': duration, 'dataset': dataset.code,
                                'id': sess_id, 'score': score, 'n_samples': len(y)})
        return results


class CrossSessionEvaluation(BaseEvaluation):
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
            event_epochs = dict(zip(event_id.keys(), [[]] * len(event_id)))
            for epoch in epochs:
                for key in event_id.keys():
                    if key in epoch.event_id.keys():
                        event_epochs[key].append(epoch[key])
            for key in event_id.keys():
                event_epochs[key] = concatenate_epochs(event_epochs[key])

            # equalize for accuracy
            equalize_epoch_counts(list(event_epochs.values()))
            ep = concatenate_epochs(list(event_epochs.values()))
            # previously multipled data by 1e6
            X, y = (ep.get_data(), ep.events[:, -1])
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

    def preprocess_data(self, d):
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
