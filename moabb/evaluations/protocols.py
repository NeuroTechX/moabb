from enum import Enum


class PredictMode(str, Enum):
    BLOCKWISE = "blockwise"
    TRIALWISE = "trialwise"


class CsMode(str, Enum):
    HOS_SOURCE_ONLY_BLOCKWISE = "hos_source_only_blockwise"
    HOS_SOURCE_ONLY_TRIALWISE = "hos_source_only_trialwise"
    HOS_UNLABELED_20P = "hos_unlabeled_20p"
    HOS_UNLABELED_50P = "hos_unlabeled_50p"
    HOS_UNLABELED_100P = "hos_unlabeled_100p"
    HOS_LABELED_20P = "hos_labeled_20p"
    HOS_LABELED_50P = "hos_labeled_50p"


_CS_MODE_MAP = {
    CsMode.HOS_SOURCE_ONLY_BLOCKWISE: dict(
        calibration_size=0.0,
        calibration_labeled=False,
        predict_mode=PredictMode.BLOCKWISE.value,
    ),
    CsMode.HOS_SOURCE_ONLY_TRIALWISE: dict(
        calibration_size=0.0,
        calibration_labeled=False,
        predict_mode=PredictMode.TRIALWISE.value,
    ),
    CsMode.HOS_UNLABELED_20P: dict(
        calibration_size=0.2,
        calibration_labeled=False,
        predict_mode=PredictMode.BLOCKWISE.value,
    ),
    CsMode.HOS_UNLABELED_50P: dict(
        calibration_size=0.5,
        calibration_labeled=False,
        predict_mode=PredictMode.BLOCKWISE.value,
    ),
    CsMode.HOS_UNLABELED_100P: dict(
        calibration_size=1.0,
        calibration_labeled=False,
        predict_mode=PredictMode.BLOCKWISE.value,
    ),
    CsMode.HOS_LABELED_20P: dict(
        calibration_size=0.2,
        calibration_labeled=True,
        predict_mode=PredictMode.BLOCKWISE.value,
    ),
    CsMode.HOS_LABELED_50P: dict(
        calibration_size=0.5,
        calibration_labeled=True,
        predict_mode=PredictMode.BLOCKWISE.value,
    ),
}


def validate_transfer_protocol(calibration_size, calibration_labeled):
    if not 0.0 <= calibration_size <= 1.0:
        raise ValueError(
            f"calibration_size must be in [0, 1]. Got {calibration_size!r}."
        )

    if not isinstance(calibration_labeled, bool):
        raise TypeError(
            "calibration_labeled must be a bool. "
            f"Got {type(calibration_labeled).__name__}."
        )

    if calibration_labeled and calibration_size > 0.5:
        raise ValueError(
            "calibration_labeled=True is only allowed with "
            "calibration_size <= 0.5."
        )


def resolve_cs_mode(cs_mode):
    params = dict(_CS_MODE_MAP[CsMode(cs_mode)])
    validate_transfer_protocol(
        params["calibration_size"],
        params["calibration_labeled"],
    )
    return params