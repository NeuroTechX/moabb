"""Motor Imagery contexts"""

import numpy as np
from .base import WithinSubjectContext, BaseContext
from mne import Epochs, find_events
from mne.epochs import concatenate_epochs, equalize_epoch_counts
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold

class BaseMotorImagery(BaseContext):
    """Base Motor imagery context


    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """

    def __init__(self, pipelines, datasets=None, fmin=7., fmax=35.):
        self.fmin = fmin
        self.fmax = fmax
        super().__init__(pipelines, datasets)


class MotorImageryMultiClasses(BaseMotorImagery, WithinSubjectContext):
    """Motor Imagery for multi class classification

    Multiclass motor imagery context. Evaluation is done in Randomized KFold or
    LeaveOneGroupOut (depending on the group variable, can be run or session)
    with accuracy as a metric. Epochs count are equalized.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    for now only works with equalized class counts (but this should be made more flexible...)

    See Also
    --------
    MotorImageryTwoClasses
    """

    def prepare_data(self, dataset, subjects, stack_sessions=False):
        """Prepare data for classification."""
        if len(dataset.selected_events) < 3:
            raise(ValueError("Dataset %s only contains two classes" %
                             dataset.code))

        event_id = dataset.selected_events
        if not event_id:
            raise(ValueError("Dataset had no selected events"))

        subjects = dataset.get_data(subjects, stack_sessions)
        subject_processed = []
        for sub in subjects:
            full_epochs = []
            # get all epochs for individual files in given subject
            epochs = self._epochs(sub, event_id, dataset.interval, self.fmin, self.fmax)
            # equalize events from different classes
            event_epochs = dict(zip(event_id.keys(), [[]]*len(event_id)))
            for epoch in epochs:
                for key in event_id.keys():
                    if key in epoch.event_id.keys():
                        event_epochs[key].append(epoch[key])
            for key in event_id.keys():
                event_epochs[key] = concatenate_epochs(event_epochs[key])

            # equalize for accuracy
            equalize_epoch_counts(list(event_epochs.values()))
            ep = concatenate_epochs(list( event_epochs.values() ))
            subject_processed.append((ep.get_data()*1e6, ep.events[:,-1]))
        return subject_processed

    def score(self, clf, X, y, groups=None, n_jobs=1):
        """get the score"""
        cv = KFold(5, shuffle=True, random_state=45)

        auc = cross_val_score(clf, X, y, cv=cv, groups=groups,
                              scoring='accuracy', n_jobs=n_jobs)
        return auc.mean()


class MotorImageryTwoClasses(BaseMotorImagery, WithinSubjectContext):
    """Motor Imagery for binary classification

    Binary motor imagery context. Evaluation is done in Randomized KFold or
    LeaveOneGroupOut (depending on the group variable, can be run or session)
    with AUC as a metric.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    fmin : float | None, (default 7.)
        Low cut-off frequency in Hz. If None the data are only low-passed.
    fmax : float | None, (default 35)
        High cut-off frequency in Hz. If None the data are only high-passed.

    See Also
    --------
    MotorImageryTwoClasses
    """

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""

        if len(dataset.event_id) > 2:
            # multiclass, pick two first classes
            raise(ValueError("Dataset %s contain more than two classes" %
                             dataset.name))

        event_id = dataset.event_id
        epochs = self._epochs(dataset, subjects, event_id)
        groups = []
        for ii, ep in enumerate(epochs):
            groups.extend([ii] * len(ep))
        epochs = concatenate_epochs(epochs)
        X = epochs.get_data()*1e6
        y = epochs.events[:, -1]
        y = np.asarray(y == np.max(y), dtype=np.int32)

        groups = np.asarray(groups)
        return X, y, groups

    def score(self, clf, X, y, groups, n_jobs=1):
        """get the score"""
        if len(np.unique(groups)) > 1:
            # if group as different values, use group
            cv = LeaveOneGroupOut()
        else:
            # else use kfold
            cv = KFold(5, shuffle=True, random_state=45)

        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring='roc_auc', n_jobs=n_jobs)
        return auc.mean()
