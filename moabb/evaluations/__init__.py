"""An evaluation defines how we go from trials per subject and session to a
generalization statistic (AUC score, f-score, accuracy, etc) -- it can be
either within-recording-session accuracy, across-session within-subject
accuracy, across-subject accuracy, or other transfer learning settings."""

# flake8: noqa
from .evaluations import (
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
    WithinSessionEvaluation,
)
from .splitters import CrossSessionSplitter, CrossSubjectSplitter, WithinSessionSplitter
from .utils import _create_save_path, _save_model_cv
