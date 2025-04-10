import contextlib
import json
import os
import os.path as op

from mne.utils._logging import logger, warn
from mne.utils.check import _soft_import, _validate_type
from mne.utils.config import _known_config_types, _known_config_wildcards, get_config_path


@contextlib.contextmanager
def _open_lock(path, *args, **kwargs):
    """
    Context manager that opens a file with an optional file lock.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock').  Otherwise, a null context is used.
    The file is then opened in the specified mode.
    """
    filelock = _soft_import("filelock", raise_error=False)
    if filelock is not None:
        lock_path = f"{path}.lock"
        try:
            # Here we set an optional timeout (e.g., 5 sec) so that processes
            # do not hang indefinitely. Adjust as needed.
            lock = filelock.FileLock(lock_path, timeout=5)
        except Exception as e:
            warn(f"Failed to create a FileLock object for {lock_path}: {e}")
            lock = None
    else:
        # Warn that locking is disabled which might lead to parallel write issues.
        warn(
            "File locking is disabled because the filelock package is not installed. "
            "This might lead to data corruption when multiple processes write simultaneously."
        )
        lock = None

    # Use the lock if available; otherwise, use a null context
    lock_context = lock if lock is not None else contextlib.nullcontext()

    # It is important to acquire the lock *before* opening the file to
    # avoid race conditions.
    with lock_context:
        with open(path, *args, **kwargs) as fid:
            yield fid


def get_config(key=None, default=None, raise_error=False, home_dir=None, use_env=True):
    """Read MNE-Python preferences from environment or config file.

    Parameters
    ----------
    key : None | str
        The preference key to look for. The os environment is searched first,
        then the mne-python config file is parsed.
        If None, all the config parameters present in environment variables or
        the path are returned. If key is an empty string, a list of all valid
        keys (but not values) is returned.
    default : str | None
        Value to return if the key is not found.
    raise_error : bool
        If True, raise an error if the key is not found (instead of returning
        default).
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    use_env : bool
        If True, consider env vars, if available.
        If False, only use MNE-Python configuration file values.

        .. versionadded:: 0.18

    Returns
    -------
    value : dict | str | None
        The preference key value.

    See Also
    --------
    set_config
    """
    _validate_type(key, (str, type(None)), "key", "string or None")

    if key == "":
        # These are str->str (immutable) so we should just copy the dict
        # itself, no need for deepcopy
        return _known_config_types.copy()

    # first, check to see if key is in env
    if use_env and key is not None and key in os.environ:
        return os.environ[key]

    # second, look for it in mne-python config file
    config_path = get_config_path(home_dir=home_dir)
    if not op.isfile(config_path):
        config = {}
    else:
        config = _load_config(config_path)

    if key is None:
        # update config with environment variables
        if use_env:
            env_keys = set(config).union(_known_config_types).intersection(os.environ)
            config.update({key: os.environ[key] for key in env_keys})
        return config
    elif raise_error is True and key not in config:
        loc_env = "the environment or in the " if use_env else ""
        meth_env = (
            (f'either os.environ["{key}"] = VALUE for a temporary solution, or ')
            if use_env
            else ""
        )
        extra_env = (
            " You can also set the environment variable before running python."
            if use_env
            else ""
        )
        meth_file = (
            f'mne.utils.set_config("{key}", VALUE, set_env=True) for a permanent one'
        )
        raise KeyError(
            f'Key "{key}" not found in {loc_env}'
            f"the mne-python config file ({config_path}). "
            f"Try {meth_env}{meth_file}.{extra_env}"
        )
    else:
        return config.get(key, default)


def set_config(key, value, home_dir=None, set_env=True):
    """Set a MNE-Python preference key in the config file and environment.

    Parameters
    ----------
    key : str
        The preference key to set.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    set_env : bool
        If True (default), update :data:`os.environ` in addition to
        updating the MNE-Python config file.

    See Also
    --------
    get_config
    """
    _validate_type(key, "str", "key")
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    _validate_type(value, (str, "path-like", type(None)), "value")
    if value is not None:
        value = str(value)

    if key not in _known_config_types and not any(
        key.startswith(k) for k in _known_config_wildcards
    ):
        warn(f'Setting non-standard config type: "{key}"')

    # Read all previous values
    config_path = get_config_path(home_dir=home_dir)
    if op.isfile(config_path):
        config = _load_config(config_path, raise_error=True)
    else:
        config = dict()
        logger.info(
            f"Attempting to create new mne-python configuration file:\n{config_path}"
        )
    if value is None:
        config.pop(key, None)
        if set_env and key in os.environ:
            del os.environ[key]
    else:
        config[key] = value
        if set_env:
            os.environ[key] = value
        if key == "MNE_BROWSER_BACKEND":
            from ..viz._figure import set_browser_backend

            set_browser_backend(value)

    # Write all values. This may fail if the default directory is not
    # writeable.
    directory = op.dirname(config_path)
    if not op.isdir(directory):
        os.mkdir(directory)
    with open(config_path, "w") as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


def _load_config(config_path, raise_error=False):
    """Safely load a config file."""
    with open(config_path) as fid:
        try:
            config = json.load(fid)
        except ValueError:
            # No JSON object could be decoded --> corrupt file?
            msg = (
                f"The MNE-Python config file ({config_path}) is not a valid JSON "
                "file and might be corrupted"
            )
            if raise_error:
                raise RuntimeError(msg)
            warn(msg)
            config = dict()
    return config
