"""Motor Imagery contexts"""

import numpy as np
from .base import WithinSubjectContext
from mne import Epochs, find_events
from mne.epochs import concatenate_epochs, equalize_epoch_counts
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, KFold


class MotorImageryMultiClasses(WithinSubjectContext):

    def __init__(self, fmin=7., fmax=35):
        self.fmin = fmin
        self.fmax = fmax

    def _epochs(self, dataset, subjects, event_id):
        """epoch data"""
        raws = dataset.get_data(subjects=subjects)
        raws = raws[0]
        ep = []
        # we process each run independently
        for raw in raws:

            # find events
            events = find_events(raw, shortest_event=0, verbose=False)

            # pick some channels
            raw.pick_types(meg=True, eeg=True, stim=False,
                           eog=False, exclude='bads')

            # filter data
            raw.filter(self.fmin, self.fmax, method='iir')

            # epoch data
            epochs = Epochs(raw, events, event_id, dataset.tmin, dataset.tmax,
                            proj=False, baseline=None, preload=True,
                            verbose=False)
            ep.append(epochs)
        return ep

    def prepare_data(self, dataset, subjects):
        """Prepare data for classification."""
        if len(dataset.event_id) < 3:
            # multiclass, pick two first classes
            raise(ValueError("Dataset %s only contains two classes" %
                             dataset.name))

        event_id = dataset.event_id
        epochs = self._epochs(dataset, subjects, event_id)
        groups = []
        full_epochs = []

        for ii, epoch in enumerate(epochs):
            epochs_list = [epoch[k] for k in event_id]
            # equalize for accuracy
            equalize_epoch_counts(epochs_list)
            ep = concatenate_epochs(epochs_list)
            groups.extend([ii] * len(ep))
            full_epochs.append(ep)

        epochs = concatenate_epochs(full_epochs)
        X = epochs.get_data()*1e6
        y = epochs.events[:, -1]
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
                              scoring='accuracy', n_jobs=n_jobs)
        return auc.mean()


class MotorImageryTwoClasses(MotorImageryMultiClasses):

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
