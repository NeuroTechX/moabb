"""BCNI left hand vs right hand."""

from ..datasets.bnci import BNCI2014004
from .base import BaseContext
from .bnci_2014_002 import BNCI2014002MI
from mne import Epochs, pick_types, find_events
from mne.epochs import concatenate_epochs
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut


class BNCI2014004MI(BNCI2014002MI):

    def __init__(self, tmin=4.5, tmax=6.5):
        self.tmin = tmin
        self.tmax = tmax
        self.dataset = BNCI2014004()
