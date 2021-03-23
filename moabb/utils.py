import mne


def set_log_level(verbose="info"):
    """Set lot level.

    Set the general log level. level can be 'info', 'debug' or 'warning'
    """
    mne.set_log_level(False)
