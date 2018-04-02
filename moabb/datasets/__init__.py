"""
A dataset handle and abstract low level access to the data. the dataset will
takes data stored locally, in the format in which they have been downloaded,
and will convert them into a MNE raw object. There are options to pool all the
different recording sessions per subject or to evaluate them separately.
"""
from .gigadb import GigaDbMI
from .alex_mi import AlexMI
from .physionet_mi import PhysionetMI
from .bnci import BNCI2014001, BNCI2014002, BNCI2014004, BNCI2015001, BNCI2015004
from .openvibe_mi import OpenvibeMI
from .bbci_eeg_fnirs import BBCIEEGfNIRS
from .upper_limb import UpperLimb
