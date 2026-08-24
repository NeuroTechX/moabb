"""Util functions for moabb."""

import abc
import contextlib
import functools
import inspect
import logging
import os
import os.path as osp
import random
import re
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import filelock
import h5py
import numpy as np
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from mne import get_config, set_config
from mne import set_log_level as sll
from mne.utils import warn


if TYPE_CHECKING:
    from moabb.datasets.base import BaseDataset
    from moabb.paradigms.base import BaseProcessing

log = logging.getLogger(__name__)


def _handle_deprecated_kwargs(kwargs, renames, class_name):
    """Handle deprecated PascalCase kwargs, returning resolved values.

    Parameters
    ----------
    kwargs : dict
        The **kwargs from the constructor.
    renames : dict
        Mapping of old PascalCase names to new snake_case names.
    class_name : str
        The class name for the warning message.

    Returns
    -------
    resolved : dict
        Mapping of new snake_case names to values from deprecated kwargs.
    """
    resolved = {}
    for old_name, new_name in renames.items():
        if old_name in kwargs:
            warnings.warn(
                f"Parameter '{old_name}' is deprecated and will be removed in "
                f"version 2.0. Use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            resolved[new_name] = kwargs.pop(old_name)
    if kwargs:
        raise TypeError(
            f"{class_name}.__init__() got unexpected keyword arguments: "
            f"{list(kwargs.keys())}"
        )
    return resolved


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


def verbose(function):
    """Verbose decorator to allow setting the level of verbosity.

    This decorator checks for a 'verbose' argument in the function signature
    or 'self.verbose' if available, and sets the logging level of the 'moabb'
    logger accordingly for the duration of the function.

    Parameters
    ----------
    function : function
        Function to be decorated.

    Returns
    -------
    dec : function
        The decorated function.
    """
    arg_names = inspect.getfullargspec(function).args

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        verbose_val = None

        # Check kwargs
        if "verbose" in kwargs:
            verbose_val = kwargs["verbose"]

        # Check positional args
        elif "verbose" in arg_names:
            sig = inspect.signature(function)
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                if "verbose" in bound.arguments:
                    verbose_val = bound.arguments["verbose"]
            except TypeError as exc:
                log.debug(
                    "Failed to bind 'verbose' argument for %s: %s", function.__name__, exc
                )

        # Check self.verbose
        if verbose_val is None and len(args) > 0:
            if hasattr(args[0], "verbose"):
                verbose_val = getattr(args[0], "verbose", None)

        logger = logging.getLogger("moabb")
        old_level = logger.level
        level = None

        if verbose_val is True:
            level = logging.INFO
        elif verbose_val is False:
            level = logging.WARNING
        elif isinstance(verbose_val, (int, str)):
            level = verbose_val

        if level is not None:
            try:
                logger.setLevel(level)
            except (TypeError, ValueError) as exc:
                logger.warning("Failed to set log level %r: %s", level, exc)

        try:
            return function(*args, **kwargs)
        finally:
            if level is not None:
                logger.setLevel(old_level)

    return wrapper


def _paths_match(first, second):
    """Return whether two path-like values refer to the same location."""
    if first is None or second is None:
        return first is second
    try:
        return Path(first).expanduser().resolve(strict=False) == Path(
            second
        ).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError):
        return first == second


def _clear_legacy_dataset_paths(old_path, new_path):
    """Remove stale dataset keys created by older MOABB releases.

    Older versions copied ``MNE_DATA`` into a
    ``MNE_DATASETS_<SIGN>_PATH`` key on first access. Those copies prevented a
    later change to the shared download directory from taking effect. Remove
    only persisted keys that still mirror the previous shared path; explicit
    per-dataset paths and environment overrides are preserved.
    """
    if old_path is None or new_path is None or _paths_match(old_path, new_path):
        return
    for key, value in (get_config(use_env=False) or {}).items():
        if (
            key.startswith("MNE_DATASETS_")
            and key.endswith("_PATH")
            and _paths_match(value, old_path)
        ):
            remove_environment_value = _paths_match(os.environ.get(key), old_path)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r'Setting non-standard config type: "MNE_DATASETS_.*_PATH"',
                    category=RuntimeWarning,
                )
                set_config(key, None, set_env=remove_environment_value)


def set_download_dir(path):
    """Set the download directory if required to change from default mne path.

    Parameters
    ----------
    path : None | str
        The new storage location. If it does not exist, a warning is raised
        and the path is created. If None and MNE_DATA config does not exist,
        a warning is raised and the storage location is set to the MNE
        default directory.
    """
    old_path = get_config("MNE_DATA")
    if path is None:
        if old_path is None:
            log.info(
                "MNE_DATA is not already configured. It will be set to "
                "default location in the home directory - "
                + osp.join(osp.expanduser("~"), "mne_data")
                + "All datasets will be downloaded to this location, if anything is "
                "already downloaded, please move manually to this location"
            )

            # No migration here: this branch only runs when MNE_DATA was
            # unset, so there is no previous value for a per-dataset key to
            # still be mirroring.
            set_config("MNE_DATA", osp.join(osp.expanduser("~"), "mne_data"))
    else:
        # Check if the path exists, if not, create it
        if not osp.isdir(path):
            log.info("The path given does not exist, creating it..")
            os.makedirs(path, exist_ok=True)
        set_config("MNE_DATA", path)
        _clear_legacy_dataset_paths(old_path, path)


def _set_moabb_config(key, value, **kwargs):
    """Write a MOABB-owned config key without MNE's unknown-key warning.

    MNE warns on every write to a key outside its own vocabulary, and offers no
    flag to suppress it, so the filter is the only lever. Written once here
    rather than at each call site.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=rf'Setting non-standard config type: "{re.escape(key)}"',
            category=RuntimeWarning,
        )
        set_config(key, value, **kwargs)


#: Where MOABB may fetch dataset files from.
#:
#: ``"auto"``     try NEMAR first for datasets that declare a ``nemar_id``,
#:                fall back to the dataset's own upstream downloader.
#: ``"nemar"``    NEMAR only -- do not fall back to the upstream host.
#: ``"upstream"`` never use NEMAR; always use the dataset's own downloader.
DOWNLOAD_PROVIDERS = ("auto", "nemar", "upstream")


def set_download_provider(provider):
    """Choose where MOABB fetches dataset files from.

    Many datasets are mirrored on NEMAR, which also republishes the original
    pre-BIDS distribution under ``sourcedata/``. Pinning the provider to
    ``"nemar"`` makes downloads reproducible and immune to upstream hosts that
    are slow, rate-limited, behind a bot gate, or retired -- at the cost of
    failing outright, rather than silently falling back, when NEMAR cannot
    serve a dataset.

    The provider governs where :meth:`~moabb.datasets.base.BaseDataset.download`
    fetches from; loading then serves files from the fetched NEMAR store before
    the URL-derived layout (pinning ``"upstream"`` opts loading out as well).

    Parameters
    ----------
    provider : str | None
        One of ``"auto"`` (default), ``"nemar"``, or ``"upstream"``. Passing
        ``None`` restores the default.

    Raises
    ------
    ValueError
        If ``provider`` is not a recognised value.

    Notes
    -----
    The choice is stored in the MNE config as ``MOABB_DOWNLOAD_PROVIDER`` and
    can also be set for a single run via the environment variable of the same
    name, which takes precedence.
    """
    if provider is not None:
        normalized = str(provider).lower()
        if normalized not in DOWNLOAD_PROVIDERS:
            raise ValueError(
                f"Unknown download provider {provider!r}; "
                f"expected one of {', '.join(DOWNLOAD_PROVIDERS)}."
            )
    else:
        normalized = None
    _set_moabb_config("MOABB_DOWNLOAD_PROVIDER", normalized)


def get_download_provider():
    """Return the active download provider.

    The environment variable ``MOABB_DOWNLOAD_PROVIDER`` wins over the stored
    MNE config value, so a single run can override a persisted preference.
    An unrecognised value falls back to ``"auto"`` with a warning rather than
    raising, so a stale config cannot break every download.

    Returns
    -------
    str
        One of :data:`DOWNLOAD_PROVIDERS`.
    """
    # get_config consults os.environ first, so an env var set for one run wins
    # over the stored preference without a second lookup here.
    provider = get_config("MOABB_DOWNLOAD_PROVIDER")
    if not provider:
        return "auto"
    normalized = str(provider).lower()
    if normalized not in DOWNLOAD_PROVIDERS:
        log.warning(
            "Ignoring unknown MOABB_DOWNLOAD_PROVIDER %r; using 'auto'. "
            "Expected one of %s.",
            provider,
            ", ".join(DOWNLOAD_PROVIDERS),
        )
        return "auto"
    return normalized


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
    header = rf"{section}[ ]*\n[ ]*[\-]+[ ]*\n"
    if re.search(rf"[ ]*{header}", doc) is None:
        doc = doc + f"\n\n    {section}\n    {'-' * len(section)}\n"
    doc = re.sub(rf"([ ]*)({header})", rf"\g<1>\g<2>\n\g<1>{msg}\n", doc)
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
            ".. note::\n\n"
            f"        ``{func.__name__}`` was previously named ``{name}``. "
            f"``{name}`` will be removed in version {expire_version}.\n"
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
    Context manager that opens an HDF5 file with a file lock.

    Acquires a lock on a lock file based on the given path (by appending
    '.lock') before opening the HDF5 file. The lock timeout and fallback
    behavior can be controlled via environment variables:

    - ``MOABB_HDF5_LOCK_TIMEOUT``: seconds to wait for the lock (default 30).
    - ``MOABB_ALLOW_UNLOCKED_HDF5``: if ``"1"``/``"true"``/``"yes"``, fall
      back to unlocked access on timeout instead of raising.

    Parameters
    ----------
    path : str
        The path to the HDF5 file to be opened.
    *args, **kwargs : optional
        Additional arguments and keyword arguments to be passed to
        ``h5py.File``.

    """
    lock_timeout = float(os.environ.get("MOABB_HDF5_LOCK_TIMEOUT", "30"))
    allow_unlocked = os.environ.get("MOABB_ALLOW_UNLOCKED_HDF5", "").lower() in (
        "1",
        "true",
        "yes",
    )
    lock_path = f"{path}.lock"
    lock = filelock.FileLock(lock_path, timeout=lock_timeout)

    try:
        with lock, h5py.File(path, *args, **kwargs) as fid:
            yield fid
    except TimeoutError as err:
        msg = f"Could not acquire lock file after {lock_timeout:g} seconds:\n{lock_path}"
        if not allow_unlocked:
            raise TimeoutError(msg) from err
        warn(
            msg + "\nProceeding without lock because "
            "MOABB_ALLOW_UNLOCKED_HDF5 is enabled."
        )
        with h5py.File(path, *args, **kwargs) as fid:
            yield fid


class MoabbMetaClass(abc.ABCMeta, NumpyDocstringInheritanceInitMeta):
    pass
