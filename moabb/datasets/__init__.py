"""A dataset handle and abstract low level access to the data. the dataset will
takes data stored locally, in the format in which they have been downloaded,
and will convert them into a MNE raw object. There are options to pool all the
different recording sessions per subject or to evaluate them separately.

See https://github.com/NeuroTechX/moabb/wiki/Datasets-Support for detail
on datasets (electrodes, number of trials, sessions, etc.)
"""

from . import compound_dataset

# flake8: noqa
from .alex_mi import AlexMI
from .alphawaves import Rodrigues2017
from .bbci_eeg_fnirs import Shin2017A, Shin2017B

# Depreciated datasets (will be removed in the future):
from .bnci import BNCI2014001  # noqa: F401
from .bnci import BNCI2014002  # noqa: F401
from .bnci import BNCI2014004  # noqa: F401
from .bnci import BNCI2014008  # noqa: F401
from .bnci import BNCI2014009  # noqa: F401
from .bnci import BNCI2015001  # noqa: F401
from .bnci import BNCI2015003  # noqa: F401
from .bnci import BNCI2015004  # noqa: F401
from .bnci import (
    BNCI2014_001,
    BNCI2014_002,
    BNCI2014_004,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_001,
    BNCI2015_003,
    BNCI2015_004,
)
from .braininvaders import VirtualReality  # noqa: F401
from .braininvaders import bi2012  # noqa: F401
from .braininvaders import bi2013a  # noqa: F401
from .braininvaders import bi2014a  # noqa: F401
from .braininvaders import bi2014b  # noqa: F401
from .braininvaders import bi2015a  # noqa: F401
from .braininvaders import bi2015b  # noqa: F401
from .braininvaders import (
    BI2012,
    BI2013a,
    BI2014a,
    BI2014b,
    BI2015a,
    BI2015b,
    Cattan2019_VR,
)
from .castillos2023 import (
    CastillosBurstVEP40,
    CastillosBurstVEP100,
    CastillosCVEP40,
    CastillosCVEP100,
)
from .epfl import EPFLP300
from .fake import FakeDataset, FakeVirtualRealityDataset
from .gigadb import Cho2017
from .hinss2021 import Hinss2021
from .huebner_llp import Huebner2017, Huebner2018
from .Lee2019 import Lee2019_ERP, Lee2019_MI, Lee2019_SSVEP
from .liu2024 import Liu2024
from .mpi_mi import MunichMI  # noqa: F401
from .mpi_mi import GrosseWentrup2009
from .neiry import DemonsP300
from .phmd_ml import HeadMountedDisplay  # noqa: F401
from .phmd_ml import Cattan2019_PHMD
from .physionet_mi import PhysionetMI
from .schirrmeister2017 import Schirrmeister2017
from .sosulski2019 import Sosulski2019
from .ssvep_exo import SSVEPExo  # noqa: F401
from .ssvep_exo import Kalunga2016
from .ssvep_mamem import MAMEM1, MAMEM2, MAMEM3
from .ssvep_nakanishi import Nakanishi2015
from .ssvep_wang import Wang2016
from .stieger2021 import Stieger2021
from .thielen2015 import Thielen2015
from .thielen2021 import Thielen2021
from .upper_limb import Ofner2017
from .utils import _init_dataset_list
from .Weibo2014 import Weibo2014
from .Zhou2016 import Zhou2016


# Call this last in order to make sure the dataset list is populated with
# the datasets imported in this file.
_init_dataset_list()
