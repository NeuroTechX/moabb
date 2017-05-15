"""Physionet left hand vs right hand."""

from ..datasets.bnci import BNCI2014001
from .base import BaseContext
from mne import Epochs, pick_types, find_events
from mne.epochs import concatenate_epochs
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut


class BNCI2014001MIMultiClass(BaseContext):

    def __init__(self, tmin=3.5, tmax=5.5):
        self.tmin = tmin
        self.tmax = tmax
        self.dataset = BNCI2014001()

    def _epochs(self, subjects, event_id):
        """epoch data"""
        raws = self.dataset.get_data(subjects=subjects)
        raws = raws[0]
        ep = []
        # we process each run independently
        for raw in raws:
            raw.filter(7., 35., method='iir')

            events = find_events(raw, shortest_event=0, verbose=False)

            picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                               eog=False, exclude='bads')
            epochs = Epochs(raw, events, event_id, self.tmin, self.tmax,
                            proj=False, picks=picks, baseline=None,
                            preload=True, verbose=False)
            ep.append(epochs)
        return ep

    def prepare_data(self, subjects):
        """Prepare data for classification."""
        event_id = dict(feet=3, left_hand=1, right_hand=2, tongue=4)
        epochs = self._epochs(subjects, event_id)
        groups = []
        for ii, ep in enumerate(epochs):
            groups.extend([ii] * len(ep))
        epochs = concatenate_epochs(epochs)
        X = epochs.get_data()*1e6
        y = epochs.events[:, -1]
        return X, y, groups

    def score(self, clf, X, y, groups, n_jobs=1):
        """get the score"""
        cv = cv = LeaveOneGroupOut()
        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring='accuracy', n_jobs=n_jobs)
        return auc.mean()


class BNCI2014001MIHands(BNCI2014001MIMultiClass):

    def prepare_data(self, subjects):
        """Prepare data for classification."""
        event_id = dict(left_hand=1, right_hand=2)
        epochs = self._epochs(subjects, event_id)
        groups = []
        for ii, ep in enumerate(epochs):
            groups.extend([ii] * len(ep))
        epochs = concatenate_epochs(epochs)
        X = epochs.get_data()*1e6
        y = epochs.events[:, -1] - 1
        return X, y, groups

    def score(self, clf, X, y, groups, n_jobs=1):
        """get the score"""
        cv = cv = LeaveOneGroupOut()
        auc = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring='roc_auc', n_jobs=n_jobs)
        return auc.mean()
