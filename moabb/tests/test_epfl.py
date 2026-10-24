"""EPFLP300 data_path: archive extraction and its log record."""

import logging
import zipfile

from moabb.datasets import EPFLP300
from moabb.datasets import download as dl


class _StrictHandler(logging.Handler):
    """Format every record, re-raising errors stock handlers only print."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def handleError(self, record):  # noqa: N802 (logging API name)
        raise


def test_data_path_unzip_log_record_formats(tmp_path, monkeypatch):
    """The 'unzip' record passed an argument without a placeholder.

    ``logger.info("unzip", path_zip)`` raised ``TypeError: not all arguments
    converted during string formatting`` in any handler that formats INFO
    records strictly (stock handlers print it to stderr instead).
    """
    zip_path = tmp_path / "subject1.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("subject1/session1/eeg_200605191428_epochs.mat", "x")
    monkeypatch.setattr(dl, "data_dl", lambda *args, **kwargs: str(zip_path))

    logger = logging.getLogger("moabb.datasets.epfl")
    handler = _StrictHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        paths = EPFLP300().data_path(1, path=str(tmp_path))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert handler.messages == [f"Unzipping {zip_path}"]
    assert paths == [
        str(tmp_path / "subject1" / "session1" / "eeg_200605191428_epochs.mat")
    ]
