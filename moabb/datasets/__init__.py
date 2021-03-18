"""
A dataset handle and abstract low level access to the data. the dataset will
takes data stored locally, in the format in which they have been downloaded,
and will convert them into a MNE raw object. There are options to pool all the
different recording sessions per subject or to evaluate them separately.
"""
# flake8: noqa
from .alex_mi import AlexMI
from .bbci_eeg_fnirs import Shin2017A, Shin2017B
from .bnci import (
    BNCI2014001,
    BNCI2014002,
    BNCI2014004,
    BNCI2014008,
    BNCI2014009,
    BNCI2015001,
    BNCI2015003,
    BNCI2015004,
)
from .braininvaders import bi2013a
from .epfl import EPFLP300
from .gigadb import Cho2017
from .Lee2019 import Lee2019_MI
from .mpi_mi import MunichMI
from .neiry import DemonsP300
from .physionet_mi import PhysionetMI
from .schirrmeister2017 import Schirrmeister2017
from .ssvep_exo import SSVEPExo
from .ssvep_mamem import MAMEM1, MAMEM2, MAMEM3
from .ssvep_nakanishi import Nakanishi2015
from .ssvep_wang import Wang2016
from .upper_limb import Ofner2017
from .Weibo2014 import Weibo2014
from .Zhou2016 import Zhou2016
