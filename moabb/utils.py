import logging
import os
import os.path as osp

from mne import get_config, set_config
from mne import set_log_level as sll


def set_log_level(level="INFO"):
    """Set lot level.

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
        if not osp.isfile(path):
            print("The path given does not exist, creating it..")
            os.makedirs(path)
        set_config("MNE_DATA", path)
