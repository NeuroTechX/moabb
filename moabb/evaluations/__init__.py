"""An evaluation defines how we go from trials per subject and session to a
generalization statistic (AUC score, f-score, accuracy, etc) -- it can be
either within-recording-session accuracy, across-session within-subject
accuracy, across-subject accuracy, or other transfer learning settings."""

# flake8: noqa
import importlib


_LAZY_IMPORTS = {
    "CrossSessionEvaluation": ".evaluations",
    "CrossSubjectEvaluation": ".evaluations",
    "WithinSessionEvaluation": ".evaluations",
    "WithinSubjectEvaluation": ".evaluations",
    "CrossSubjectMode": ".protocols",
    "CrossDatasetSplitter": ".splitters",
    "CrossSessionSplitter": ".splitters",
    "CrossSubjectSplitter": ".splitters",
    "LearningCurveSplitter": ".splitters",
    "WithinSessionSplitter": ".splitters",
    "WithinSubjectSplitter": ".splitters",
    "_create_save_path": ".utils",
    "_save_model_cv": ".utils",
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

