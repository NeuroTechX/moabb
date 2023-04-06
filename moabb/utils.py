import logging
import os
import os.path as osp
import random

import numpy as np
from mne import get_config, set_config
from mne import set_log_level as sll


def _set_random_seed(seed: int) -> None:
    """
    Set the seed for Python's built-in random module and numpy.
    Parameters
    ----------
    seed: int
        The random seed to use.
    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)


def _set_tensorflow_seed(seed: int) -> None:
    """
    Set the seed for TensorFlow.
    Parameters
    ----------
    seed: int
        The random seed to use.
    Returns
    -------
    None
    """
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)  # tf cpu fix seed
        os.environ[
            "TF_DETERMINISTIC_OPS"
        ] = "1"  # tf gpu fix seed, please `pip install tensorflow-determinism` first
        tf.keras.utils.set_random_seed(seed)

    except ImportError:
        print(
            "We try to set the tensorflow seeds, but it seems that tensorflow is not installed. "
            "Please refer to `https://www.tensorflow.org/` to install if you need to use "
            "this deep learning module."
        )
        return False


def _set_torch_seed(seed: int) -> None:
    """
    Set the seed for PyTorch.
    Parameters
    ----------
    seed: int
        The random seed to use.
    Returns
    -------
    None
    """
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        print(
            "We try to set the torch seeds, but it seems that torch is not installed. "
            "Please refer to `https://pytorch.org/` to install if you need to use "
            "this deep learning module."
        )
        return False


def setup_seed(seed: int) -> None:
    """
    Set the seed for random, numpy, TensorFlow and PyTorch.
    Parameters
    ----------
    seed: int
        The random seed to use.
    Returns
    -------
    None
    """
    _set_random_seed(seed)
    # check if the return is bool
    tensorflow_return = _set_tensorflow_seed(seed)
    torch_return = _set_torch_seed(seed)

    if tensorflow_return is False or torch_return is False:
        return False
    else:
        return None


def set_log_level(level="INFO"):
    """Set log level

    Set the general log level.
    Use one of the levels supported by python logging, i.e.:
    DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    VALID_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level = level.upper()
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level {level}. Choose one of {VALID_LEVELS}.")
    sll(False)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
    )


def set_download_dir(path):
    """Set the download directory if required to change from default mne path

    Parameters
    ----------
    path : None | str
    The new storage location, if it does not exist, a warning is raised and the
    path is created
    If None, and MNE_DATA config does not exist, a warning is raised and the
    storage location is set to the MNE default directory

    """
    if path is None:
        if get_config("MNE_DATA") is None:
            print(
                "MNE_DATA is not already configured. It will be set to "
                "default location in the home directory - "
                + osp.join(osp.expanduser("~"), "mne_data")
                + "All datasets will be downloaded to this location, if anything is "
                "already downloaded, please move manually to this location"
            )

            set_config("MNE_DATA", osp.join(osp.expanduser("~"), "mne_data"))
    else:
        # Check if the path exists, if not, create it
        if not osp.isdir(path):
            print("The path given does not exist, creating it..")
            os.makedirs(path)
        set_config("MNE_DATA", path)
