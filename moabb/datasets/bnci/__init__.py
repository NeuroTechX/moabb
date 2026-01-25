"""BNCI Horizon 2020 datasets.

This subpackage contains all BNCI datasets organized by year.
"""

# Newer datasets (2016-2025)
from .bnci_2016_002 import BNCI2016_002
from .bnci_2020_001 import BNCI2020_001
from .bnci_2020_002 import BNCI2020_002
from .bnci_2022_001 import BNCI2022_001
from .bnci_2024_001 import BNCI2024_001
from .bnci_2025_001 import BNCI2025_001
from .bnci_2025_002 import BNCI2025_002

# Legacy datasets (2003-2019) from the original BNCI file
from .legacy import (  # Base classes and utilities; Dataset classes; Deprecated aliases
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
    BNCI2019_001,
    BNCI2014001,
    BNCI2014002,
    BNCI2014004,
    BNCI2014008,
    BNCI2014009,
    BNCI2015001,
    BNCI2015003,
    BNCI2015004,
    MNEBNCI,
    load_data,
)


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
