"""BCNI left hand vs right hand."""

from ..datasets.bbci_eeg_fnirs import BBCIEEGfNIRS
from .bnci_2014_002 import BNCI2014002MI


class BBCIEEGfNIRSMI(BNCI2014002MI):

    def __init__(self, tmin=3.5, tmax=10.):
        self.tmin = tmin
        self.tmax = tmax
        self.dataset = BBCIEEGfNIRS(motor=True)
