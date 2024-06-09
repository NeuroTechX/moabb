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
    utils_pytorch_classes = {
        "InputShapeSetterEEG",
        "BraindecodeDatasetLoader",
        "get_shape_from_baseconcat",
    }

    if name in deep_learning_classes and _check_if_tensorflow_installed():
        return _import_class(name, ".deep_learning")
    elif name in utils_deep_model_classes and _check_if_tensorflow_installed():
        return _import_class(name, ".utils_deep_model")
    elif name in utils_pytorch_classes and _check_if_braindecode_installed():
        return _import_class(name, ".utils_pytorch")

    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


def _import_class(name, module_name):
    import importlib

    warning_msg = _warning_msg(name, module_name)
    warn(warning_msg)

    module = importlib.import_module(module_name, __package__)
    return getattr(module, name)


def _warning_msg(name, submodule):
    return (
        f"{name} is incorrectly imported. \nPlease use:\033[1m "
        f"from moabb.pipeline{submodule} import {name}\033[0m.\n"
        f"Instead of: \033[1mfrom moabb.pipeline import {name}\033[0m."
    )


def _check_if_tensorflow_installed():
    try:
        import scikeras

        return True
    except ModuleNotFoundError:
        warn(
            "\nThere was a problem importing tensorflow or keras, "
            "which are required for the deep learning pipelines. \n"
            "The Keras MOABB deep learning pipelines cannot be used.\n "
            "To resolve this issue, please install the necessary dependencies "
            "by running the following command in your terminal: \n"
            "\033[94m"  # This is the ANSI escape code for blue
            "pip install moabb[deeplearning]"
            "\033[0m",  # This resets the color back to normal
        )
        return False


def _check_if_braindecode_installed():
    try:
        import braindecode

        return True
    except ModuleNotFoundError:
        warn(
            "Braindecode is not installed. "
            "You won't be able to use these braindecode functions if you "
            "attempt to do so. \n"
            "\033[94m"  # This is the ANSI escape code for blue
            "pip install braindecode"
            "\033[0m",  # This resets the color back to normal
        )
        return False
