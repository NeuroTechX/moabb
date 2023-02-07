"""
Pipeline defines all steps required by an algorithm to obtain predictions.
Pipelines are typically a chain of sklearn compatible transformers and end
with an sklearn compatible estimator.
"""
from .braindecode import BraindecodeClassifierModel, CreateBraindecodeDataset

# flake8: noqa
from .classification import SSVEP_CCA, SSVEP_TRCA
from .features import FM, ExtendedSSVEPSignal, LogVariance
from .utils import FilterBank, create_pipeline_from_config
