"""SSVEP dataset adapters for MOABB.

This package provides additional SSVEP dataset implementations that subclass
moabb's BaseDataset/BaseBIDSDataset. Each dataset module is designed to be
portable to moabb/datasets/ with minimal import changes.
"""

from .chen2017_single_flicker import Chen2017SingleFlicker
from .dong2023_ssvep import Dong2023
from .han2024_fatigue import Han2024Fatigue
from .kim2025_beta_range import Kim2025BetaRange
from .lee2021_mobile import Lee2021Mobile, Lee2021Mobile_ERP, Lee2021Mobile_SSVEP
from .liu2020_beta import Liu2020BETA
from .liu2022_eldbeta import Liu2022EldBETA
from .wang2021_combined import Wang2021Combined


__all__ = [
    "Liu2020BETA",
    "Liu2022EldBETA",
    "Kim2025BetaRange",
    "Dong2023",
    "Lee2021Mobile",
    "Lee2021Mobile_SSVEP",
    "Lee2021Mobile_ERP",
    "Chen2017SingleFlicker",
    "Wang2021Combined",
    "Han2024Fatigue",
]
