"""Pipeline defines all steps required by an algorithm to obtain predictions.

Pipelines are typically a chain of sklearn compatible transformers and
end with a sklearn compatible estimator.
"""

# flake8: noqa
import importlib
from warnings import warn


_LAZY_IMPORTS = {
    "SSVEP_CCA": ".classification",
    "SSVEP_SSCOR": ".classification",
    "SSVEP_TDCA": ".classification",
    "SSVEP_TRCA": ".classification",
    "SSVEP_TRCA_R": ".classification",
    "SSVEP_eCCA": ".classification",
    "SSVEP_itCCA": ".classification",
    "SSVEP_MsetCCA": ".classification",
    "FM": ".features",
    "AugmentedDataset": ".features",
    "ExtendedSSVEPSignal": ".features",
    "LogVariance": ".features",
    "FilterBank": ".utils",
    "create_pipeline_from_config": ".utils",
    "get_benchmark_pipelines": ".utils",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val

    deep_learning_classes = {
        "KerasDeepConvNet",
        "KerasEEGITNet",
        "KerasEEGNet_8_2",
        "KerasEEGNeX",
        "KerasEEGTCNet",
        "KerasShallowConvNet",
    }
    utils_deep_model_classes = {"EEGNet", "TCN_block"}

    if name in deep_learning_classes or name in utils_deep_model_classes:
        raise AttributeError(
            f"Module deep learning using tensorflow is not "
            f"longer part of moabb package. Please use "
            f"braindecode package instead."
            f"See https://braindecode.org/ for more information."
        )

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

