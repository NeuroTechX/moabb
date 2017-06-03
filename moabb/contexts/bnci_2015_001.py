"""BCNI left hand vs right hand."""

from ..datasets.bnci import BNCI2015001
from .bnci_2014_002 import BNCI2014002MI


class BNCI2015001MI(BNCI2014002MI):

    def __init__(self, tmin=4., tmax=7.5):
        self.tmin = tmin
        self.tmax = tmax
        self.dataset = BNCI2015001()
