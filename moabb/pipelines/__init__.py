"""
Pipeline defines all steps required by an algorithm to obtain predictions.
Pipelines are typically a chain of sklearn compatible transformers and end
with an sklearn compatible estimator.
"""
# flake8: noqa
from .classification import SSVEP_CCA
from .features import LogVariance, FM, ExtendedSSVEPSignal
from .utils import create_pipeline_from_config, FilterBank
