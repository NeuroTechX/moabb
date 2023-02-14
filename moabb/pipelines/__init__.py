"""
Pipeline defines all steps required by an algorithm to obtain predictions.
Pipelines are typically a chain of sklearn compatible transformers and end
with a sklearn compatible estimator.
"""
# flake8: noqa
from .classification import SSVEP_CCA, SSVEP_TRCA
from .deep_learning import (
    KerasDeepConvNet,
    KerasEEGITNet,
    KerasEEGNet_8_2,
    KerasEEGNeX,
    KerasEEGTCNet,
    KerasShallowConvNet,
)
from .features import FM, ExtendedSSVEPSignal, LogVariance
from .utils import FilterBank, create_pipeline_from_config
from .utils_deep_model import EEGNet, TCN_block
