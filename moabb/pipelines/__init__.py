"""
Pipeline defines all steps required by an algorithm to obtain predictions.
Pipelines are typically a chain of sklearn compatible transformers and end
with a sklearn compatible estimator.
"""

# flake8: noqa
try:
    from .braindecode import CreateBraindecodeDataset
except ModuleNotFoundError as err:
    print("Braindecode not install, you could not use those pipelines")

from .classification import SSVEP_CCA, SSVEP_TRCA
from .features import FM, ExtendedSSVEPSignal, LogVariance
from .utils import FilterBank, create_pipeline_from_config


try:
    from .deep_learning import (
        KerasDeepConvNet,
        KerasEEGITNet,
        KerasEEGNet_8_2,
        KerasEEGNeX,
        KerasEEGTCNet,
        KerasShallowConvNet,
    )
    from .utils_deep_model import EEGNet, TCN_block
except ModuleNotFoundError as err:
    print("Tensorflow not install, you could not use those pipelines")
