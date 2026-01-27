"""Legacy BNCI datasets (2003-2019).

This module is kept for backwards compatibility. The legacy datasets are now
organized by year in separate modules.
"""

from .bnci_2003 import BNCI2003_004
from .bnci_2014 import (
    BNCI2014_001,
    BNCI2014_002,
    BNCI2014_004,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2014001,
    BNCI2014002,
    BNCI2014004,
    BNCI2014008,
    BNCI2014009,
)
from .bnci_2015 import (
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
    BNCI2015001,
    BNCI2015003,
    BNCI2015004,
)
from .bnci_2019 import BNCI2019_001
from .legacy_base import MNEBNCI, load_data


__all__ = [
    "MNEBNCI",
    "load_data",
    "BNCI2003_004",
    "BNCI2014_001",
    "BNCI2014_002",
    "BNCI2014_004",
    "BNCI2014_008",
    "BNCI2014_009",
    "BNCI2015_001",
    "BNCI2015_003",
    "BNCI2015_004",
    "BNCI2015_006",
    "BNCI2015_007",
    "BNCI2015_008",
    "BNCI2015_009",
    "BNCI2015_010",
    "BNCI2015_012",
    "BNCI2015_013",
    "BNCI2019_001",
    "BNCI2014001",
    "BNCI2014002",
    "BNCI2014004",
    "BNCI2014008",
    "BNCI2014009",
    "BNCI2015001",
    "BNCI2015003",
    "BNCI2015004",
]
