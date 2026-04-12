"""A paradigm defines how the raw data will be converted to trials ready to be
processed by a decoding algorithm.

This is a function of the paradigm used, i.e. in motor imagery one can
have two-class, multi-class, or continuous paradigms; similarly,
different preprocessing is necessary for ERP vs ERD paradigms.
"""

from .cvep import CVEP as CVEP
from .cvep import BaseCVEP as BaseCVEP
from .cvep import FakeCVEPParadigm as FakeCVEPParadigm
from .cvep import FilterBankCVEP as FilterBankCVEP
from .fixed_interval_windows import (
    BaseFixedIntervalWindowsProcessing as BaseFixedIntervalWindowsProcessing,
)
from .fixed_interval_windows import (
    FilterBankFixedIntervalWindowsProcessing as FilterBankFixedIntervalWindowsProcessing,
)
from .fixed_interval_windows import (
    FixedIntervalWindowsProcessing as FixedIntervalWindowsProcessing,
)
from .motor_imagery import BaseMotorImagery as BaseMotorImagery
from .motor_imagery import FakeImageryParadigm as FakeImageryParadigm
from .motor_imagery import FilterBankLeftRightImagery as FilterBankLeftRightImagery
from .motor_imagery import FilterBankMotorImagery as FilterBankMotorImagery
from .motor_imagery import Imagery as Imagery
from .motor_imagery import LeftRightImagery as LeftRightImagery
from .motor_imagery import MotorImagery as MotorImagery
from .motor_imagery import SpeechImagery as SpeechImagery
from .p300 import P300 as P300
from .p300 import BaseP300 as BaseP300
from .p300 import FakeP300Paradigm as FakeP300Paradigm
from .resting_state import RestingStateToP300Adapter as RestingStateToP300Adapter
from .ssvep import SSVEP as SSVEP
from .ssvep import BaseSSVEP as BaseSSVEP
from .ssvep import FakeSSVEPParadigm as FakeSSVEPParadigm
from .ssvep import FilterBankSSVEP as FilterBankSSVEP
