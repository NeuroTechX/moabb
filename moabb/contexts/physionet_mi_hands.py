"""Physionet left hand vs right hand."""

from ..datasets.physionet_mi import PhysionetMI
from .base import BaseContext
from mne import Epochs, pick_types, find_events
from mne.epochs import equalize_epoch_counts, concatenate_epochs
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut


class PhysionetMIHands(BaseContext):

    def __init__(self, tmin=1, tmax=3):
        self.tmin = tmin
        self.tmax = tmax
        self.dataset = PhysionetMI(feets=False, imagined=True)

    def _epochs(self, subjects, event_id):
        """epoch data"""
        raws = self.dataset.get_data(subjects=subjects)
        raws = raws[0]
        ep = []
        # we process each run independently
        for raw in raws:
            raw.filter(7., 35., method='iir')

            events = find_events(raw, shortest_event=0, stim_channel='STI 014',
                                 verbose=False)

            picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                               eog=False, exclude='bads')
            epochs = Epochs(raw, events, event_id, self.tmin, self.tmax,
                            proj=False, picks=picks, baseline=None,
                            preload=True, verbose=False)
            ep.append(epochs)
        return ep

    def prepare_data(self, subjects):
        """Prepare data for classification."""
        event_id = dict(left_hand=2, right_hand=3)
        epochs = self._epochs(subjects, event_id)
        groups = []
        for ii, ep in enumerate(epochs):
            groups.extend([ii] * len(ep))
        epochs = concatenate_epochs(epochs)
        X = epochs.get_data()*1e6
        y = epochs.events[:, -1] - 2
        return X, y, groups

    def score(self, clf, X, y, groups, n_jobs=1):
        """get the score"""
        cv = LeaveOneGroupOut()
        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring='roc_auc', n_jobs=n_jobs)
        return auc.mean()


class PhysionetMIHandsMultiClass(PhysionetMIHands):

    def prepare_data(self, subjects):
        """Prepare data for classification."""
        event_id = dict(rest=1, left_hand=2, right_hand=3)
        epochs = self._epochs(subjects, event_id)
        # since we are using accuracy, we have to equalize the number of
        # events
        groups = []
        full_epochs = []
        for ii, epoch in enumerate(epochs):
            epochs_list = [epoch[k] for k in event_id]
            equalize_epoch_counts(epochs_list)
            ep = concatenate_epochs(epochs_list)
            groups.extend([ii] * len(ep))
            full_epochs.append(ep)
        epochs = concatenate_epochs(full_epochs)
        X = epochs.get_data()*1e6
        y = epochs.events[:, -1]
        return X, y, groups

    def score(self, clf, X, y, groups, n_jobs=1):
        """get the score"""
        cv = cv = LeaveOneGroupOut()
        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring='accuracy', n_jobs=n_jobs)
        return auc.mean()
