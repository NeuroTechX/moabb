import mne
import logging
import coloredlogs


def set_log_level(verbose='info'):
    """Set lot level.

    Set the general log level. level can be 'info', 'debug' or 'warning'
    """
    mne.set_log_level(False)

    level = {'debug': logging.DEBUG,
             'info': logging.INFO,
             'warning': logging.WARNING}

    coloredlogs.install(level=level.get(verbose, logging.INFO))
