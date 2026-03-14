"""A dataset handle and abstract low level access to the data. the dataset will
takes data stored locally, in the format in which they have been downloaded,
and will convert them into a MNE raw object. There are options to pool all the
different recording sessions per subject or to evaluate them separately.

See http://moabb.neurotechx.com/docs/dataset_summary.html for detail
on datasets (electrodes, number of trials, sessions, etc.)
"""

from . import compound_dataset

# flake8: noqa
from .alex_mi import AlexMI
from .alphawaves import Rodrigues2017
from .bbci_eeg_fnirs import Shin2017A, Shin2017B
from .beetl import Beetl2021_A, Beetl2021_B

# BNCI datasets (from bnci subpackage)
from .bnci import (
    BNCI2003_004,
    BNCI2014_001,
    BNCI2014_002,
    BNCI2014_004,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_001,
    BNCI2015_003,
    BNCI2015_004,
    BNCI2015_006,
    BNCI2015_007,
    BNCI2015_008,
    BNCI2015_009,
    BNCI2015_010,
    BNCI2015_012,
    BNCI2015_013,
    BNCI2016_002,
    BNCI2019_001,
    BNCI2020_001,
    BNCI2020_002,
    BNCI2022_001,
    BNCI2024_001,
    BNCI2025_001,
    BNCI2025_002,
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
from .brandl2020 import Brandl2020
from .castillos2023 import (
    CastillosBurstVEP40,
    CastillosBurstVEP100,
    CastillosCVEP40,
    CastillosCVEP100,
)
from .chailloux2020 import Chailloux2020
from .chang2025 import Chang2025
from .dreyer2023 import Dreyer2023, Dreyer2023A, Dreyer2023B, Dreyer2023C
from .epfl import EPFLP300
from .erpcore2021 import (
    ErpCore2021_ERN,
    ErpCore2021_LRP,
    ErpCore2021_MMN,
    ErpCore2021_N2pc,
    ErpCore2021_N170,
    ErpCore2021_N400,
    ErpCore2021_P3,
)
from .fake import FakeDataset, FakeVirtualRealityDataset
from .forenzo2023 import Forenzo2023
from .gao2026 import Gao2026
from .gigadb import Cho2017
from .guttmann_flury2025 import (
    GuttmannFlury2025_MI,
    GuttmannFlury2025_P300,
    GuttmannFlury2025_SSVEP,
)
from .hefmi_ich2025 import HefmiIch2025
from .hinss2021 import Hinss2021
from .huebner_llp import Huebner2017, Huebner2018
from .jeong2020 import Jeong2020
from .kaneshiro2015 import Kaneshiro2015
from .kaya2018 import Kaya2018
from .kojima2024a import Kojima2024A
from .kojima2024b import Kojima2024B
from .kumar2024 import Kumar2024
from .Lee2019 import Lee2019_ERP, Lee2019_MI, Lee2019_SSVEP
from .lee2021_mobile import Lee2021Mobile_ERP, Lee2021Mobile_SSVEP
from .lee2024 import Lee2024_AC, Lee2024_BS, Lee2024_DL, Lee2024_EL, Lee2024_TV
from .liu2024 import Liu2024
from .liu2025 import Liu2025
from .ma2020 import Ma2020
from .mainsah2025 import (
    Mainsah2025_A,
    Mainsah2025_B,
    Mainsah2025_C,
    Mainsah2025_D,
    Mainsah2025_E,
    Mainsah2025_F,
    Mainsah2025_G,
    Mainsah2025_H,
    Mainsah2025_I,
    Mainsah2025_J,
    Mainsah2025_K,
    Mainsah2025_L,
    Mainsah2025_M,
    Mainsah2025_N,
    Mainsah2025_O,
    Mainsah2025_P,
    Mainsah2025_Q,
    Mainsah2025_R,
    Mainsah2025_S1,
    Mainsah2025_S2,
)
from .martinezcagigal2023_checker_cvep import MartinezCagigal2023Checker
from .martinezcagigal2023_pary_cvep import MartinezCagigal2023Pary
from .mpi_mi import MunichMI  # noqa: F401
from .mpi_mi import GrosseWentrup2009
from .phmd_ml import HeadMountedDisplay  # noqa: F401
from .phmd_ml import Cattan2019_PHMD
from .physionet_mi import PhysionetMI
from .romani_bf2025_erp import RomaniBF2025ERP
from .rozado2015 import Rozado2015
from .schirrmeister2017 import Schirrmeister2017
from .simoes2020 import Simoes2020
from .sosulski2019 import Sosulski2019
from .speier2017 import Speier2017
from .ssvep_chen2017 import Chen2017SingleFlicker
from .ssvep_dong2023 import Dong2023
from .ssvep_exo import SSVEPExo  # noqa: F401
from .ssvep_exo import Kalunga2016
from .ssvep_han2024 import Han2024Fatigue
from .ssvep_kim2025 import Kim2025BetaRange
from .ssvep_liu2020 import Liu2020BETA
from .ssvep_liu2022 import Liu2022EldBETA
from .ssvep_mamem import MAMEM1, MAMEM2, MAMEM3
from .ssvep_nakanishi import Nakanishi2015
from .ssvep_wang import Wang2016
from .ssvep_wang2021 import Wang2021Combined
from .stieger2021 import Stieger2021
from .tavakolan2017 import Tavakolan2017
from .thielen2015 import Thielen2015
from .thielen2021 import Thielen2021
from .triana_guzman2024 import TrianaGuzman2024
from .upper_limb import Ofner2017
from .utils import _init_dataset, dataset_dict
from .wairagkar2018 import Wairagkar2018
from .Weibo2014 import Weibo2014
from .wu2020 import Wu2020
from .yang2025 import Yang2025
from .yi2025 import Yi2025
from .zhang2017 import Zhang2017
from .zhang2025 import Zhang2025
from .zheng2020 import Zheng2020
from .Zhou2016 import Zhou2016
from .zhou2020 import Zhou2020
from .zuo2025 import Zuo2025


# Call this last in order to make sure the dataset list, dict are populated with
# the datasets imported in this file.
_init_dataset()

# Defer canonicalization to lazy catalog build time to avoid instantiating
# all dataset classes on every import of moabb.datasets.


_REMOVED_DATASETS = {
    "DemonsP300": "DemonsP300 has been removed due to unresolved data issues.",
    "Forenzo2024": (
        "Forenzo2024 has been removed because it is a continuous pursuit "
        "regression task, not a discrete-trial motor imagery paradigm. "
        "The .mat files contain no left/right trial labels."
    ),
}


def __getattr__(name):
    if name in _REMOVED_DATASETS:
        raise AttributeError(_REMOVED_DATASETS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
