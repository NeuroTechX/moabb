import logging

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
