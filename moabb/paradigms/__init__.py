"""A paradigm defines how the raw data will be converted to trials ready to be
processed by a decoding algorithm.

This is a function of the paradigm used, i.e. in motor imagery one can
have two-class, multi-class, or continuous paradigms; similarly,
different preprocessing is necessary for ERP vs ERD paradigms.
"""

from .cvep import CVEP, BaseCVEP, FakeCVEPParadigm, FilterBankCVEP
from .fixed_interval_windows import (
    BaseFixedIntervalWindowsProcessing,
    FilterBankFixedIntervalWindowsProcessing,
    FixedIntervalWindowsProcessing,
)
from .motor_imagery import (
    BaseMotorImagery,
    FakeImageryParadigm,
    FilterBankLeftRightImagery,
    FilterBankMotorImagery,
    LeftRightImagery,
    MotorImagery,
)
from .p300 import BaseP300, FakeP300Paradigm, P300
from .resting_state import RestingStateToP300Adapter
from .ssvep import BaseSSVEP, FakeSSVEPParadigm, FilterBankSSVEP, SSVEP
