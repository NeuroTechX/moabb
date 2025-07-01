"""Pipeline defines all steps required by an algorithm to obtain predictions.

Pipelines are typically a chain of sklearn compatible transformers and
end with a sklearn compatible estimator.
"""

from warnings import warn

# flake8: noqa
from .classification import SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA
from .features import (
    FM,
    AugmentedDataset,
    ExtendedSSVEPSignal,
    LogVariance,
)
from .utils import FilterBank, create_pipeline_from_config


def __getattr__(name):

    deep_learning_classes = {
        "KerasDeepConvNet",
        "KerasEEGITNet",
        "KerasEEGNet_8_2",
        "KerasEEGNeX",
        "KerasEEGTCNet",
        "KerasShallowConvNet",
    }
    utils_deep_model_classes = {
        "EEGNet",
        "TCN_block",
    }

    if name in deep_learning_classes or name in utils_deep_model_classes:
        raise AttributeError(
            f"Module deep learning using tensorflow is not "
            f"longer part of moabb package. Please use "
            f"braindecode package instead."
            f"See https://braindecode.org/ for more information."
        )
