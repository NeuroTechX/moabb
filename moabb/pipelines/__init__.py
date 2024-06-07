"""Pipeline defines all steps required by an algorithm to obtain predictions.

Pipelines are typically a chain of sklearn compatible transformers and
end with a sklearn compatible estimator.
"""

from warnings import warn

# flake8: noqa
from .classification import SSVEP_CCA, SSVEP_TRCA, SSVEP_MsetCCA
from .features import FM, AugmentedDataset, ExtendedSSVEPSignal, LogVariance
from .utils import FilterBank, create_pipeline_from_config


def __getattr__(name):
    import importlib

    if name in [
        "KerasDeepConvNet",
        "KerasEEGITNet",
        "KerasEEGNet_8_2",
        "KerasEEGNeX",
        "KerasEEGTCNet",
        "KerasShallowConvNet",
    ]:
        if _check_if_tensorflow_install():
            warning_msg = _warning_msg(name, "deep_learning")
            warn(f"{warning_msg}")

            dl_module = importlib.import_module(".deep_learning", __package__)
            dl_class = dl_module.__dict__[name]
            return dl_class

    if name in [
        "EEGNet",
        "TCN_block",
    ]:
        if _check_if_tensorflow_install():
            warning_msg = _warning_msg(name, "utils_deep_model")
            warn(f"{warning_msg}")

            dl_module = importlib.import_module(".utils_deep_model", __package__)
            dl_class = dl_module.__dict__[name]
            return dl_class

    if name in [
        "InputShapeSetterEEG",
        "BraindecodeDatasetLoader",
        "get_shape_from_baseconcat",
    ]:
        if _check_if_braindecode_install():
            warn(_warning_msg(name, "utils_pytorch"))
            sub_module = importlib.import_module(".utils_pytorch", __package__)
            class_obj = sub_module.__dict__[name]
            return class_obj

    raise AttributeError("No possible import named " + name)


def _warning_msg(name, submodule):
    msg = (
        f"{name} is incorrectly imported. \nPlease use:\033[1m "
        f"from moabb.pipeline.{submodule} import {name}\033[0m.\n"
        f"Instead of: \033[1mfrom moabb.pipeline import {name}\033[0m."
    )
    return msg


def _check_if_tensorflow_install():
    try:
        import keras
        import scikeras
        import tensorflow

        return True
    except ModuleNotFoundError:
        warn(
            "\nThere was a problem importing tensorflow or keras, "
            "which are required for the deep learning pipelines. \n"
            "The Keras MOABB deep learning pipelines cannot be used.\n "
            "To resolve this issue, please install the necessary dependencies "
            "by running the following command in your terminal: \n"
            "\033[94m"  # This is the ANSI escape code for green
            "pip install moabb[deeplearning]"
            "\033[0m",  # This resets the color back to normal
        )
        return False


def _check_if_braindecode_install():
    try:
        import braindecode

        return True
    except ModuleNotFoundError:
        warn(
            "Braindecode is not installed. "
            "You won't be able to use these braindecode function if you "
            "attempt to do. \n"
            "\033[94m"  # This is the ANSI escape code for green
            "pip install braindecode"
            "\033[0m",  # This resets the color back to normal
        )
        return False
