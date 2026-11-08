"""A paradigm defines how the raw data will be converted to trials ready to be
processed by a decoding algorithm.

This is a function of the paradigm used, i.e. in motor imagery one can
have two-class, multi-class, or continuous paradigms; similarly,
different preprocessing is necessary for ERP vs ERD paradigms.
"""

import importlib


_LAZY_IMPORTS = {
    "CognitiveTask": ".cognitive",
    "FakeCognitiveParadigm": ".cognitive",
    "CVEP": ".cvep",
    "BaseCVEP": ".cvep",
    "FakeCVEPParadigm": ".cvep",
    "FilterBankCVEP": ".cvep",
    "BaseFixedIntervalWindowsProcessing": ".fixed_interval_windows",
    "FilterBankFixedIntervalWindowsProcessing": ".fixed_interval_windows",
    "FixedIntervalWindowsProcessing": ".fixed_interval_windows",
    "BaseMotorImagery": ".motor_imagery",
    "FakeImageryParadigm": ".motor_imagery",
    "FilterBankLeftRightImagery": ".motor_imagery",
    "FilterBankMotorImagery": ".motor_imagery",
    "Imagery": ".motor_imagery",
    "LeftRightImagery": ".motor_imagery",
    "MotorImagery": ".motor_imagery",
    "SpeechImagery": ".motor_imagery",
    "P300": ".p300",
    "BaseP300": ".p300",
    "FakeP300Paradigm": ".p300",
    "RestingStateToP300Adapter": ".resting_state",
    "SSVEP": ".ssvep",
    "BaseSSVEP": ".ssvep",
    "FakeSSVEPParadigm": ".ssvep",
    "FilterBankSSVEP": ".ssvep",
}

__all__ = list(_LAZY_IMPORTS)


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

