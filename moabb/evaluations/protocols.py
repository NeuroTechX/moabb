"""Cross-subject transfer protocol presets.

This module defines named presets for held-out-subject (HOS) evaluation.

Blockwise prediction means that the estimator receives the whole target test
block at scoring time, as in the standard MOABB CrossSubjectEvaluation path.

Trialwise prediction means that the estimator is scored one target trial at a
time. This prevents methods from using statistics of the full target test block
during prediction.
"""
from enum import Enum

class CsMode(str, Enum):
    # Default source-only cross-subject evaluation.
    # Train only on source subjects; no target calibration data is used.
    # The held-out target test data is predicted blockwise.
    HOS_SOURCE_ONLY = "hos_source_only"

    # Strict source-only one-shot evaluation.
    # Train only on source subjects; no target calibration data is used.
    # The held-out target test data is predicted one trial at a time.
    HOS_SOURCE_ONLY_TRIALWISE = "hos_source_only_trialwise"

    # Use 20% of the held-out subject as unlabeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    HOS_UNLABELED_20P = "hos_unlabeled_20p"

    # Use 50% of the held-out subject as unlabeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    HOS_UNLABELED_50P = "hos_unlabeled_50p"

    # Use all held-out target data as unlabeled adaptation data.
    # Evaluate transductively on the target test block, predicted blockwise.
    HOS_UNLABELED_100P = "hos_unlabeled_100p"

    # Use 20% of the held-out subject as labeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    HOS_LABELED_20P = "hos_labeled_20p"

    # Use 50% of the held-out subject as labeled target calibration/adaptation data.
    # Evaluate on the remaining target test data, predicted blockwise.
    HOS_LABELED_50P = "hos_labeled_50p"

_CS_MODE_MAP = {
    CsMode.HOS_SOURCE_ONLY: dict(
        calibration_size=0.0,
        calibration_labeled=False,
    ),
    CsMode.HOS_SOURCE_ONLY_TRIALWISE: dict(
        calibration_size=0.0,
        calibration_labeled=False,
    ),
    CsMode.HOS_UNLABELED_20P: dict(
        calibration_size=0.2,
        calibration_labeled=False,
    ),
    CsMode.HOS_UNLABELED_50P: dict(
        calibration_size=0.5,
        calibration_labeled=False,
    ),
    CsMode.HOS_UNLABELED_100P: dict(
        calibration_size=1.0,
        calibration_labeled=False,
    ),
    CsMode.HOS_LABELED_20P: dict(
        calibration_size=0.2,
        calibration_labeled=True,
    ),
    CsMode.HOS_LABELED_50P: dict(
        calibration_size=0.5,
        calibration_labeled=True,
    ),
}


def validate_transfer_protocol(calibration_size, calibration_labeled):
    if not isinstance(calibration_size, (int, float)):
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


def resolve_cs_mode(cs_mode):
    params = dict(_CS_MODE_MAP[CsMode(cs_mode)])
    validate_transfer_protocol(params["calibration_size"], params["calibration_labeled"])
    return params


def is_one_shot_mode(cs_mode):
    return CsMode(cs_mode) == CsMode.HOS_SOURCE_ONLY_TRIALWISE
 