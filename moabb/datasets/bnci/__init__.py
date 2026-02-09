"""BNCI Horizon 2020 datasets.

This subpackage contains all BNCI datasets organized by year.
"""

from .base import MNEBNCI, load_data

# Legacy datasets (2003-2019), split by year
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

# Newer datasets (2016-2025)
from .bnci_2016_002 import BNCI2016_002
from .bnci_2019 import BNCI2019_001
from .bnci_2020 import BNCI2020_001, BNCI2020_002
from .bnci_2022_001 import BNCI2022_001
from .bnci_2024_001 import BNCI2024_001
from .bnci_2025 import BNCI2025_001, BNCI2025_002


__all__ = [
    # Base classes and utilities
    "MNEBNCI",
    "load_data",
    # Legacy datasets
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
    # Newer datasets
    "BNCI2016_002",
    "BNCI2020_001",
    "BNCI2020_002",
    "BNCI2022_001",
    "BNCI2024_001",
    "BNCI2025_001",
    "BNCI2025_002",
    # Deprecated aliases
    "BNCI2014001",
    "BNCI2014002",
    "BNCI2014004",
    "BNCI2014008",
    "BNCI2014009",
    "BNCI2015001",
    "BNCI2015003",
    "BNCI2015004",
]
