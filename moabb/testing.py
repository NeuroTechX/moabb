import json
import os
import os.path as op

from mne.utils._logging import logger, warn
from mne.utils.check import _validate_type
from mne.utils.config import _known_config_types, _known_config_wildcards


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
        config = _load_config_no_lock(config_path)

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


def get_config_path(home_dir=None):
    # Dummy implementation: In practice, this finds ~/.mne/mne-python.json, etc.
    if home_dir is None:
        home_dir = os.path.expanduser("~")
    config_dir = op.join(home_dir, ".mne")
    if not op.isdir(config_dir):
        os.mkdir(config_dir)
    return op.join(config_dir, "mne-python.json")


def _load_config_no_lock(config_path, raise_error=False):
    """Load config from file without acquiring the lock (assumes lock already held)."""
    if not op.isfile(config_path):
        return {}
    with open(config_path, "r") as fid:
        try:
            config = json.load(fid)
        except ValueError:
            msg = (
                f"The MNE-Python config file ({config_path}) is not a valid JSON "
                "file and might be corrupted"
            )
            if raise_error:
                raise RuntimeError(msg)
            warn(msg)
            config = {}
    return config


def set_config(key, value, home_dir=None, set_env=True):
    """Set a MNE-Python preference key in the config file and environment.

    Parameters
    ----------
    key : str
        The preference key to set.
    value : str | None
        The value to assign to the preference key. If None, the key is deleted.
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    set_env : bool
        If True (default), update os.environ in addition to updating the config file.

    See Also
    --------
    get_config
    """
    _validate_type(key, "str", "key")
    # We only allow string (or path-like/None) values, so enforce that:
    _validate_type(value, (str, "path-like", type(None)), "value")
    if value is not None:
        value = str(value)
    if key not in _known_config_types and not any(
        key.startswith(k) for k in _known_config_wildcards
    ):
        warn(f'Setting non-standard config type: "{key}"')

    config_path = get_config_path(home_dir=home_dir)
    # Use one lock for the whole read-update-write cycle:
    from filelock import FileLock

    lock_path = config_path + ".lock"
    lock = FileLock(lock_path)
    with lock:
        # Read the current config (without acquiring the lock again)
        if op.isfile(config_path):
            config = _load_config_no_lock(config_path, raise_error=True)
        else:
            config = dict()
            logger.info(
                f"Attempting to create new mne-python configuration file:\n{config_path}"
            )
        # Update the config: if value is None, delete the key; otherwise, set it.
        if value is None:
            config.pop(key, None)
            if set_env and key in os.environ:
                del os.environ[key]
        else:
            config[key] = value
            if set_env:
                os.environ[key] = value
            if key == "MNE_BROWSER_BACKEND":
                from mne.viz._figure import set_browser_backend

                set_browser_backend(value)
        # Ensure directory exists.
        directory = op.dirname(config_path)
        if not op.isdir(directory):
            os.mkdir(directory)
        # Write the updated config.
        with open(config_path, "w") as fid:
            json.dump(config, fid, sort_keys=True, indent=0)
