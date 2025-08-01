"""Util functions for moabb."""

import contextlib
import inspect
import logging
import os
import os.path as osp
import random
import re
import sys
from typing import TYPE_CHECKING

import filelock
import h5py
import numpy as np
from mne import get_config, set_config
from mne import set_log_level as sll
from mne.utils import warn


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


def setup_seed(seed: int) -> None:
    """Set the seed for random, numpy.

    Parameters
    ----------
    seed: int
        The random seed to use.
    Returns
    -------
    None
    """
    _set_random_seed(seed)

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
            log.info(
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
            log.info("The path given does not exist, creating it..")
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


@contextlib.contextmanager
def _open_lock_hdf5(path, *args, **kwargs):
    """
    Context manager that opens a file with an optional file lock.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock').

    Otherwise, a null context is used. The path is then opened in the
    specified mode.

    Parameters
    ----------
    path : str
        The path to the file to be opened.
    *args, **kwargs : optional
        Additional arguments and keyword arguments to be passed to the
        `open` function.

    """
    lock_context = contextlib.nullcontext()  # default to no lock

    if filelock:
        lock_path = f"{path}.lock"
        try:
            lock_context = filelock.FileLock(lock_path, timeout=5)
            lock_context.acquire()
        except TimeoutError:
            warn(
                "Could not acquire lock file after 5 seconds, consider deleting it "
                f"if you know the corresponding file is usable:\n{lock_path}"
            )
            lock_context = contextlib.nullcontext()

    with lock_context, h5py.File(path, *args, **kwargs) as fid:
        yield fid
