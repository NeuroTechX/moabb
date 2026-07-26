"""Named target-access protocols for cross-subject evaluation."""

from enum import Enum


class CrossSubjectMode(str, Enum):
    """Target data made available in a cross-subject benchmark.

    A mode fixes the target calibration fraction, whether calibration labels
    are routed, and whether prediction is blockwise or trialwise.

    Pass a member to the ``cs_mode`` parameter of
    :class:`moabb.evaluations.CrossSubjectEvaluation`.
    """

    def __new__(cls, value, calibration_size, calibration_labeled, trialwise=False):
        member = str.__new__(cls, value)
        member._value_ = value
        member.calibration_size = calibration_size
        member.calibration_labeled = calibration_labeled
        member.trialwise = trialwise
        return member

    # Train only on source subjects and predict the target block normally.
    TRAIN = ("train", 0.0, False)

    # Train only on source subjects and predict one target trial at a time.
    TRAIN_TRIALWISE = ("train_trialwise", 0.0, False, True)

    # Use an unlabeled target slice for adaptation.
    TRAIN_AND_TARGET_UNLABELED_20P = ("train_and_target_unlabeled_20p", 0.2, False)
    TRAIN_AND_TARGET_UNLABELED_50P = ("train_and_target_unlabeled_50p", 0.5, False)

    # Transductive: adapt on the same unlabeled target block that is scored.
    TRAIN_AND_TARGET_UNLABELED_FULL = ("train_and_target_unlabeled_full", 1.0, False)

    # Use a labeled target slice for calibration.
    TRAIN_AND_TARGET_LABELED_20P = ("train_and_target_labeled_20p", 0.2, True)
    TRAIN_AND_TARGET_LABELED_50P = ("train_and_target_labeled_50p", 0.5, True)


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
