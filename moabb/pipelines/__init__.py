"""Pipeline defines all steps required by an algorithm to obtain predictions.

Pipelines are typically a chain of sklearn compatible transformers and
end with a sklearn compatible estimator.
"""

# flake8: noqa
from mne.utils import warn

from .classification import SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA
from .features import FM, AugmentedDataset, ExtendedSSVEPSignal, LogVariance
from .utils import FilterBank, create_pipeline_from_config


def __getattr__(name):
    # ideas from https://stackoverflow.com/a/57110249/1469195
    import importlib
    from warnings import warn

    if name in [
        "KerasDeepConvNet",
        "KerasEEGITNet",
        "KerasEEGNet_8_2",
        "KerasEEGNeX",
        "KerasEEGTCNet",
        "KerasShallowConvNet",
    ]:
        warn(
            f"{name} is incorrectly import, please use from "
            f"moabb.pipeline.deep_learning import {name} or"
            f"instead of moabb.pipeline import {name}",
            category=DeprecationWarning,
        )
        if check_if_tensorflow_install():
            dl_module = importlib.import_module(".deep_learning", __package__)
            dl_class = dl_module.__dict__[name]
            return dl_class

    if name in [
        "InputShapeSetterEEG",
        "BraindecodeDatasetLoader",
        "get_shape_from_baseconcat",
    ]:
        print("AAAAAAA")
        warn(
            f"{name} is incorrectly import, please use from "
            f"moabb.pipeline.utils_pytorch import {name} or"
            f"instead of moabb.pipeline import {name}",
            category=DeprecationWarning,
        )
        if check_if_braindecode_install():
            sub_module = importlib.import_module(".utils_pytorch", __package__)
            class_obj = sub_module.__dict__[name]
            return class_obj

    raise AttributeError("No possible import named " + name)


def check_if_tensorflow_install():
    try:
        import keras
        import scikeras
        import tensorflow

        return True
    except ModuleNotFoundError as err:
        warn(
            f"{err}\n. You have issues importing tensorflow or keras.\n"
            "You won't be able to use these deep learning pipelines if you "
            "attempt to do so. Please run `pip install moabb[deeplearning]`, if you"
            "want to use these pipelines.",
            category=ModuleNotFoundError,
            module="moabb.pipelines",
        )
        return False


def check_if_braindecode_install():
    try:
        import braindecode

        return True
    except ModuleNotFoundError as err:
        warn(
            "Braindecode is not installed. "
            "You won't be able to use these braindecode function if you "
            "attempt to do. Please install `braindecode`",
            category=ModuleNotFoundError,
            module="moabb.pipelines",
        )
        return False
