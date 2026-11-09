"""BNCI Horizon 2020 datasets.

This subpackage contains all BNCI datasets organized by year.
"""

import importlib


_LAZY_IMPORTS = {
    # Base classes and utilities
    "MNEBNCI": ".base",
    "load_data": ".base",
    # Legacy datasets (2003-2019), split by year
    "BNCI2003_004": ".bnci_2003",
    "BNCI2014_001": ".bnci_2014",
    "BNCI2014_002": ".bnci_2014",
    "BNCI2014_004": ".bnci_2014",
    "BNCI2014_008": ".bnci_2014",
    "BNCI2014_009": ".bnci_2014",
    "BNCI2014001": ".bnci_2014",
    "BNCI2014002": ".bnci_2014",
    "BNCI2014004": ".bnci_2014",
    "BNCI2014008": ".bnci_2014",
    "BNCI2014009": ".bnci_2014",
    "BNCI2015_001": ".bnci_2015",
    "BNCI2015_003": ".bnci_2015",
    "BNCI2015_004": ".bnci_2015",
    "BNCI2015_006": ".bnci_2015",
    "BNCI2015_007": ".bnci_2015",
    "BNCI2015_008": ".bnci_2015",
    "BNCI2015_009": ".bnci_2015",
    "BNCI2015_010": ".bnci_2015",
    "BNCI2015_012": ".bnci_2015",
    "BNCI2015_013": ".bnci_2015",
    "BNCI2015001": ".bnci_2015",
    "BNCI2015003": ".bnci_2015",
    "BNCI2015004": ".bnci_2015",
    # Newer datasets (2016-2025)
    "BNCI2016_002": ".bnci_2016_002",
    "BNCI2019_001": ".bnci_2019",
    "BNCI2020_001": ".bnci_2020",
    "BNCI2020_002": ".bnci_2020",
    "BNCI2022_001": ".bnci_2022_001",
    "BNCI2024_001": ".bnci_2024_001",
    "BNCI2025_001": ".bnci_2025",
    "BNCI2025_002": ".bnci_2025",
}

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


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    try:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
