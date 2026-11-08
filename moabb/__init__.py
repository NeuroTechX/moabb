# flake8: noqa
import importlib

from .benchmark import benchmark


__version__ = "1.8.0"

_UTILS_EXPORTS = (
    "get_download_provider",
    "make_process_pipelines",
    "set_download_dir",
    "set_download_provider",
    "set_log_level",
    "setup_seed",
)

_SUBPACKAGES = (
    "analysis",
    "datasets",
    "evaluations",
    "paradigms",
    "pipelines",
    "utils",
)

__all__ = [
    "__version__",
    "benchmark",
    *_UTILS_EXPORTS,
]


def __getattr__(name):
    if name in _UTILS_EXPORTS:
        utils = importlib.import_module("moabb.utils")
        val = getattr(utils, name)
        globals()[name] = val
        return val
    if name in _SUBPACKAGES:
        mod = importlib.import_module(f"moabb.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_UTILS_EXPORTS) | set(_SUBPACKAGES))

