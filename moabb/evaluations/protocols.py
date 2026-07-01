"""Cross-subject transfer protocol presets.

This module defines named presets for cross-subject evaluation in which one
subject is held out as the target subject and the remaining subjects are used
as training/source subjects.

The presets describe what information from the held-out target subject is
allowed to be used by an estimator, and how the remaining target data are
scored. They make the evaluation protocol explicit, so that different
cross-subject and transfer-learning methods can be compared under controlled
and reproducible conditions.

A preset controls two related aspects of the evaluation:

1. target calibration/adaptation access:
   whether the estimator receives no target data, an unlabeled target slice, or
   a labeled target slice during fitting;

2. prediction access at scoring time:
   whether the estimator receives the target test data as a full block or one
   trial at a time.

Blockwise prediction means that the estimator receives the whole target test
block at scoring time, as in the standard MOABB CrossSubjectEvaluation path.

Trialwise prediction means that the estimator is scored one target trial at a
time. This prevents methods from using statistics of the full target test block
during prediction.
"""

from enum import Enum


class CrossSubjectMode(str, Enum):
    # Train only on training/source subjects; no target calibration data is used.
    # The held-out target test data is predicted blockwise.
    TRAIN = "train"

    # Train only on training/source subjects; no target calibration data is used.
    # The held-out target test data is predicted one trial at a time.
    TRAIN_TRIALWISE = "train_trialwise"

    # Train on source subjects and use 20% of the held-out subject as
    # unlabeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    TRAIN_AND_TARGET_UNLABELED_20P = "train_and_target_unlabeled_20p"

    # Train on source subjects and use 50% of the held-out subject as
    # unlabeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    TRAIN_AND_TARGET_UNLABELED_50P = "train_and_target_unlabeled_50p"

    # Train on source subjects and use all held-out target data as
    # unlabeled target adaptation data.
    # Evaluate transductively on the target test block, predicted blockwise.
    TRAIN_AND_TARGET_UNLABELED_FULL = "train_and_target_unlabeled_full"

    # Train on source subjects and use 20% of the held-out subject as
    # labeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    TRAIN_AND_TARGET_LABELED_20P = "train_and_target_labeled_20p"

    # Train on source subjects and use 50% of the held-out subject as
    # labeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    TRAIN_AND_TARGET_LABELED_50P = "train_and_target_labeled_50p"


_CROSS_SUBJECT_MODE_MAP = {
    CrossSubjectMode.TRAIN: dict(
        calibration_size=0.0,
        calibration_labeled=False,
    ),
    CrossSubjectMode.TRAIN_TRIALWISE: dict(
        calibration_size=0.0,
        calibration_labeled=False,
    ),
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_20P: dict(
        calibration_size=0.2,
        calibration_labeled=False,
    ),
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_50P: dict(
        calibration_size=0.5,
        calibration_labeled=False,
    ),
    CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_FULL: dict(
        calibration_size=1.0,
        calibration_labeled=False,
    ),
    CrossSubjectMode.TRAIN_AND_TARGET_LABELED_20P: dict(
        calibration_size=0.2,
        calibration_labeled=True,
    ),
    CrossSubjectMode.TRAIN_AND_TARGET_LABELED_50P: dict(
        calibration_size=0.5,
        calibration_labeled=True,
    ),
}


def validate_transfer_protocol(calibration_size, calibration_labeled):
    if isinstance(calibration_size, bool) or not isinstance(
        calibration_size, (int, float)
    ):
        raise TypeError(
            f"calibration_size must be a number. Got {type(calibration_size).__name__}."
        )

    calibration_size = float(calibration_size)

    if not 0.0 <= calibration_size <= 1.0:
        raise ValueError(f"calibration_size must be in [0, 1]. Got {calibration_size!r}.")

    if not isinstance(calibration_labeled, bool):
        raise TypeError(
            "calibration_labeled must be a bool. "
            f"Got {type(calibration_labeled).__name__}."
        )

    if calibration_labeled and calibration_size > 0.5:
        raise ValueError(
            "calibration_labeled=True is only allowed with calibration_size <= 0.5."
        )


def resolve_cross_subject_mode(cross_subject_mode):
    params = dict(_CROSS_SUBJECT_MODE_MAP[CrossSubjectMode(cross_subject_mode)])
    validate_transfer_protocol(params["calibration_size"], params["calibration_labeled"])
    return params


def is_trialwise_mode(cross_subject_mode):
    return CrossSubjectMode(cross_subject_mode) == CrossSubjectMode.TRAIN_TRIALWISE