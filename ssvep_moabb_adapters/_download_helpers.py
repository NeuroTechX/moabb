"""Download helpers for SSVEP dataset adapters.

Provides utilities for downloading from Tsinghua BCI server, extracting
Zenodo zip archives, and Google Drive access.
"""

import logging
import zipfile
from pathlib import Path

from moabb.datasets import download as dl


log = logging.getLogger(__name__)


def download_zenodo_zip(
    url, sign, subject_label, path=None, force_update=False, verbose=None, extract_to=None
):
    """Download a zip file from Zenodo and extract it.

    Parameters
    ----------
    url : str
        URL of the zip file.
    sign : str
        Dataset signifier for MNE config (e.g. ``"LIU2024FATIGUE"``).
    subject_label : str
        Label used to identify the subject (e.g. ``"S1"``).
    path : str or None
        Override storage location.
    force_update : bool
        Force re-download.
    verbose : bool or None
        Verbosity.
    extract_to : str or None
        Directory to extract into. If None, extracts next to the zip file.

    Returns
    -------
    extract_dir : Path
        Path to the directory containing extracted files.
    """
    zip_path = dl.data_dl(url, sign, path, force_update, verbose)
    zip_path = Path(zip_path)

    if extract_to is None:
        extract_dir = zip_path.parent / subject_label
    else:
        extract_dir = Path(extract_to)

    if not extract_dir.exists() or force_update:
        log.info("Extracting %s to %s", zip_path.name, extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    return extract_dir


def download_from_google_drive(
    file_id, dest_path, sign, path=None, force_update=False, verbose=None
):
    """Download a file from Google Drive using pooch.

    Uses the direct download URL format:
    ``https://drive.google.com/uc?id={file_id}&export=download``

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    dest_path : str
        Filename for local storage.
    sign : str
        Dataset signifier for MNE config.
    path : str or None
        Override storage location.
    force_update : bool
        Force re-download.
    verbose : bool or None
        Verbosity.

    Returns
    -------
    local_path : str
        Local path to the downloaded file.
    """
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    return dl.data_dl(url, sign, path, force_update, verbose)


def get_figshare_url(article_id, file_id=None):
    """Build a Figshare download URL.

    Parameters
    ----------
    article_id : int
        Figshare article ID.
    file_id : int or None
        Specific file ID. If None, returns the article files API endpoint.

    Returns
    -------
    url : str
        Figshare URL.
    """
    base = f"https://api.figshare.com/v2/articles/{article_id}/files"
    if file_id is not None:
        return f"https://ndownloader.figshare.com/files/{file_id}"
    return base


def find_mat_files(directory, pattern="*.mat"):
    """Find all .mat files in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search.
    pattern : str
        Glob pattern. Default ``"*.mat"``.

    Returns
    -------
    files : list of Path
        Sorted list of matching files.
    """
    directory = Path(directory)
    return sorted(directory.rglob(pattern))
