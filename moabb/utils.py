"""Util functions for moabb."""

import inspect
import logging
import os
import os.path as osp
import random
import re
import sys
from typing import TYPE_CHECKING

import numpy as np
from mne import get_config, set_config
from mne import set_log_level as sll


if TYPE_CHECKING:
    from moabb.datasets.base import BaseDataset
    from moabb.paradigms.base import BaseProcessing

log = logging.getLogger(__name__)


def _set_random_seed(seed: int) -> None:
    """Set the seed for Python's built-in random module and numpy.

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
    """Set the seed for TensorFlow.

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
        os.environ["TF_DETERMINISTIC_OPS"] = (
            "1"  # tf gpu fix seed, please `pip install tensorflow-determinism` first
        )
        tf.keras.utils.set_random_seed(seed)

    except ImportError:
        print(
            "We try to set the tensorflow seeds, but it seems that tensorflow is not installed. "
            "Please refer to `https://www.tensorflow.org/` to install if you need to use "
            "this deep learning module."
        )
        return False


def _set_torch_seed(seed: int) -> None:
    """Set the seed for PyTorch.

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
    """Set the seed for random, numpy, TensorFlow and PyTorch.

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
    """Set log level.

    Set the general log level. Use one of the levels supported by python
    logging, i.e.: DEBUG, INFO, WARNING, ERROR, CRITICAL
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
    """Set the download directory if required to change from default mne path.

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


def make_process_pipelines(
    processing: "BaseProcessing",
    dataset: "BaseDataset",
    return_epochs: bool = False,
    return_raws: bool = False,
    postprocess_pipeline=None,
):
    """Shortcut for the method :func:`moabb.paradigms.base.BaseProcessing.make_process_pipelines`"""
    return processing.make_process_pipelines(
        dataset, return_epochs, return_raws, postprocess_pipeline
    )


aliases_list = []  # list of tuples containing (old name, new name, expire version)


def update_docstring_list(doc, section, msg):
    header = f"{section}[ ]*\n[ ]*[\-]+[ ]*\n"
    if section not in doc:
        doc = doc + f"\n\n    {section}\n    {'-' * len(section)}\n"
    if re.search(f"[ ]*{header}", doc) is None:
        raise ValueError(
            f"Incorrect formatting of section {section!r} in docstring {doc!r}"
        )
    doc = re.sub(f"([ ]*)({header})", f"\g<1>\g<2>\n\g<1>{msg}\n", doc)
    return doc


def depreciated_alias(name, expire_version):
    """Decorator that creates an alias for the decorated function or class,
    marks that alias as depreciated, and adds the alias to ``aliases_list``.
    Not working on methods."""

    def factory(func):
        warn_msg = (
            f"{name} has been renamed to {func.__name__}. "
            f"{name} will be removed in version {expire_version}."
        )
        note_msg = (
            f".. note:: ``{func.__name__}`` was previously named ``{name}``. "
            f"``{name}`` will be removed in  version {expire_version}."
        )

        namespace = sys._getframe(1).f_globals  # Caller's globals.
        if inspect.isclass(func):

            def __init__(self, *args, **kwargs):
                log.warning(warn_msg)
                func.__init__(self, *args, **kwargs)

            namespace[name] = type(name, (func,), dict(func.__dict__, __init__=__init__))
        elif inspect.isfunction(func):

            def depreciated_func(*args, **kwargs):
                log.warning(warn_msg)
                return func(*args, **kwargs)

            depreciated_func.__name__ = name
            namespace[name] = depreciated_func
        else:
            raise ValueError("Can only decorate functions and classes")
        func.__doc__ = update_docstring_list(func.__doc__ or "", "Notes", note_msg)
        aliases_list.append((name, func.__name__, expire_version))
        return func

    return factory
