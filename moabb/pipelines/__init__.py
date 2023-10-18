"""Pipeline defines all steps required by an algorithm to obtain predictions.

Pipelines are typically a chain of sklearn compatible transformers and
end with a sklearn compatible estimator.
"""

# flake8: noqa
from mne.utils import warn

from .classification import SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA
from .features import FM, AugmentedDataset, ExtendedSSVEPSignal, LogVariance
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
    warn(
        "Tensorflow is not installed. "
        "You won't be able to use these MOABB pipelines if you attempt to do so.",
        category=ModuleNotFoundError,
        module="moabb.pipelines",
    )

try:
    from .utils_pytorch import (
        BraindecodeDatasetLoader,
        InputShapeSetterEEG,
        get_shape_from_baseconcat,
    )
except ModuleNotFoundError as err:
    print(
        "To use the get_shape_from_baseconcar, InputShapeSetterEEG, BraindecodeDatasetLoader"
        "you need to install `braindecode`."
        "`pip install braindecode` or Please refer to `https://braindecode.org`."
    )
