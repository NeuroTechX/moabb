# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Sylvain Chevallier <sylvain.chevallier@uvsq.fr>
#         Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD Style.

import contextlib
import contextvars
import functools
import glob as glob_module
import json
import logging
import os
import os.path as osp
import shutil
import warnings
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import nemar
import pandas as pd
import requests
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.utils import _url_to_local_path, verbose, warn
from nemar.errors import NemarError, SelectionError
from pooch import file_hash, retrieve
from pooch.downloaders import choose_downloader
from requests.exceptions import HTTPError


logger = logging.getLogger(__name__)

#: Local directory mirroring the upstream files of the dataset whose loader is
#: currently executing (its NEMAR sourcedata store), or None. Set by
#: ``BaseDataset`` around ``_get_single_subject_data`` so the URL downloaders
#: below can serve prefetched files without threading a parameter through
#: every dataset's ``data_path``.
_ACTIVE_STORE = contextvars.ContextVar("moabb_active_sourcedata_store", default=None)


@contextlib.contextmanager
def active_sourcedata_store(store):
    """Declare the local store that mirrors upstream files for these downloads."""
    token = _ACTIVE_STORE.set(store)
    try:
        yield
    finally:
        _ACTIVE_STORE.reset(token)


def nemar_store(dataset_code, nemar_id, path=None):
    """The on-disk root of a NEMAR deposit -- shared by its writer and readers."""
    return Path(get_dataset_path(dataset_code, path)) / "NEMAR" / nemar_id


def _store_lookup(url, fname=None):
    """Serve ``url``'s file from the active local store, if present.

    The store keeps the upstream relative layout, so the file is probed by the
    trailing segments of the URL path, longest tail first; an explicit
    ``fname`` is probed as-is. A miss returns None and the caller downloads.
    """
    store = _ACTIVE_STORE.get()
    if store is None:
        return None
    if fname is not None:
        tails = [PurePosixPath(fname).parts]
    else:
        parts = PurePosixPath(urlparse(url).path).parts
        # ponytail: three trailing segments cover every current dataset layout;
        # deepen if a deposit ever nests further.
        tails = [parts[i:] for i in range(max(len(parts) - 3, 0), len(parts))]
    for tail in tails:
        if not tail:
            continue
        candidate = Path(store).joinpath(*tail)
        if candidate.is_file():
            logger.info(
                "Filling %s from the local NEMAR store instead of downloading %s.",
                candidate.name,
                url,
            )
            return str(candidate)
    return None


class NemarDownloadError(RuntimeError):
    """Raised when a NEMAR download cannot be completed."""


class DatasetDownloadError(RuntimeError):
    """Raised when a download completed but did not deliver the dataset's data.

    Distinct from a transport error: the request succeeded. Repositories
    increasingly answer a missing file with HTTP 200 and an HTML page, which a
    status-code check accepts and caches under the requested name. Callers that
    sweep many datasets need to tell "this upstream is unavailable, skip it"
    apart from "moabb has a bug", and a bare RuntimeError is indistinguishable
    from the ones raised for a missing unrar binary or an absent XDF stream.
    """


def get_user_agent():
    """Return a user agent string for outbound requests."""
    try:
        from importlib import metadata

        version = metadata.version("moabb")
        return f"moabb/{version} (https://github.com/NeuroTechX/moabb)"
    except Exception:
        return "moabb (https://github.com/NeuroTechX/moabb)"


def _set_user_agent(downloader):
    headers = downloader.kwargs.setdefault("headers", {})
    headers.setdefault("User-Agent", get_user_agent())


def _sanitize_path(path: Path) -> Path:
    path = Path(path)
    table = {ord(c): "-" for c in ':*?"<>|'}

    if path.anchor:
        return Path(path.anchor, *(part.translate(table) for part in path.parts[1:]))

    return Path(*(part.translate(table) for part in path.parts))


def _normalize_destination(url: str, root: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc == "zenodo.org":
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 4 and parts[0] in {"record", "records"} and parts[2] == "files":
            record_id = parts[1]
            fname = parts[-1]
            return root / "zenodo" / record_id / fname
        if len(parts) >= 5 and parts[0] == "api" and parts[1] == "records":
            record_id = parts[2]
            fname = parts[-1]
            return root / "zenodo" / record_id / fname
    return Path(_url_to_local_path(url, root))


def get_dataset_path(sign, path):
    """Returns the dataset path allowing for changes in MNE_DATA config.

    Parameters
    ----------
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.

    Returns
    -------
        path : None | str
        Location of where to look for the data storing location
    """
    sign = sign.upper()
    key = "MNE_DATASETS_{:s}_PATH".format(sign)
    if get_config(key) is None:
        if get_config("MNE_DATA") is None:
            path_def = Path.home() / "mne_data"
            logger.info(
                "MNE_DATA is not already configured. It will be set to "
                "default location in the home directory - "
                + str(path_def)
                + "\nAll datasets will be downloaded to this location, if anything is "
                "already downloaded, please move manually to this location"
            )
            if not path_def.is_dir():
                path_def.mkdir(parents=True)
            set_config("MNE_DATA", str(Path.home() / "mne_data"))
    return _get_path(path, key, sign)


@verbose
def data_path(url, sign, path=None, force_update=False, update_path=True, verbose=None):
    """Get path to local copy of given dataset URL. **Deprecated**

    This is a low-level function useful for getting a local copy of a
    remote dataset. It is deprecated in favor of data_dl.

    Parameters
    ----------
    url : str
        Path to remote location of data
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None, **Deprecated**
        Unused, kept for compatibility purpose.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """  # noqa: E501
    path = get_dataset_path(sign, path)
    key_dest = "MNE-{:s}-data".format(sign.lower())
    destination = _url_to_local_path(url, osp.join(path, key_dest))
    if not force_update and not osp.isfile(destination):
        mirror = _store_lookup(url)
        if mirror is not None:
            os.makedirs(osp.dirname(destination), exist_ok=True)
            try:
                os.link(mirror, destination)
            except OSError:
                shutil.copy2(mirror, destination)
    # Fetch the file
    if not osp.isfile(destination) or force_update:
        if osp.isfile(destination):
            os.remove(destination)
        if not osp.isdir(osp.dirname(destination)):
            os.makedirs(osp.dirname(destination))
        retrieve(url, None, path=destination)
    return destination


@verbose
def data_dl(url, sign, path=None, force_update=False, verbose=None, fname=None):
    """Download file from url to specified path.

    This function should replace data_path as the MNE will not support the download
    of dataset anymore. This version is using Pooch.

    Parameters
    ----------
    url : str
        Path to remote location of data
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).
    fname : None | str
        Name to store the file under, relative to the dataset directory. By
        default the name is derived from ``url``, which assumes the URL ends in
        the filename. Pass this for APIs whose download URLs do not: DSpace
        serves every bitstream from ``.../bitstreams/<uuid>/content``, so
        URL-derived naming would store each one as ``content``.

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """
    path = Path(get_dataset_path(sign, path))
    key_dest = "MNE-{:s}-data".format(sign.lower())
    root = path / key_dest
    if fname is not None:
        destination = _sanitize_path(root / fname)
    else:
        destination = _sanitize_path(_normalize_destination(url, root))
        legacy_destination = _sanitize_path(Path(_url_to_local_path(url, root)))
        if legacy_destination.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                legacy_destination.replace(destination)
            except OSError:
                destination = legacy_destination

    # Fill a missing destination from the governed NEMAR store instead of the
    # network. Hardlinked so datasets that move their files cannot gut the
    # store; the normal flow below then validates and returns the destination.
    if not force_update and not destination.is_file():
        mirror = _store_lookup(url, fname=fname)
        if mirror is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(mirror, destination)
            except OSError:
                shutil.copy2(mirror, destination)

    downloader = choose_downloader(url, progressbar=True)
    if type(downloader).__name__ in ["HTTPDownloader", "DOIDownloader"]:
        downloader.kwargs.setdefault("verify", False)
    _set_user_agent(downloader)

    # Fetch the file
    if not destination.is_file() or force_update:
        if destination.is_file():
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        known_hash = None
    else:
        known_hash = file_hash(str(destination))
    dlpath = retrieve(
        url,
        known_hash,
        fname=destination.name,
        path=str(destination.parent),
        progressbar=True,
        downloader=downloader,
    )
    return dlpath


@verbose
def nemar_dl(
    nemar_id,
    dataset_code,
    path=None,
    force_update=False,
    subject=None,
    verbose=None,
    **bids_filters,
):
    """Download a NEMAR dataset and return the local BIDS root.

    Parameters
    ----------
    nemar_id : str
        NEMAR dataset identifier.
    dataset_code : str
        MOABB dataset code used to choose the local dataset directory.
    path : None | str
        Base path where MOABB stores datasets.
    force_update : bool
        Remove an existing NEMAR download before downloading again.
    subject : str | list of str | None
        BIDS subject label(s) to pass to :func:`nemar.download`.
    verbose : bool, str, int, or None
        If not None, override default verbose level.
    **bids_filters
        Additional BIDS filters forwarded to :func:`nemar.download`.

    Returns
    -------
    str
        Local BIDS root for the downloaded NEMAR dataset.

    Raises
    ------
    NemarDownloadError
        If nemar-py fails to download the selected files.
    """
    root = Path(get_dataset_path(dataset_code, path))
    target_dir = root / f"MNE-{dataset_code.lower()}-data" / nemar_id

    # ``force_update`` is handled by ``trust_existing=False`` below, which makes
    # nemar-py re-fetch the requested subject. Do not delete ``target_dir``
    # here: it is the shared BIDS root, and ``download()`` calls this once per
    # subject, so removing it would wipe subjects fetched in earlier iterations.
    try:
        nemar.download(
            dataset=nemar_id,
            target_dir=target_dir,
            subject=subject,
            trust_existing=not force_update,
            **bids_filters,
        )
    except (NemarError, OSError, ConnectionError, TimeoutError, ValueError) as exc:
        raise NemarDownloadError(f"Could not download NEMAR dataset {nemar_id}.") from exc
    return str(target_dir)


#: Name of the manifest every enriched deposit ships inside ``sourcedata/``.
SOURCEDATA_PROVENANCE = "sourcedata/sourcedata_provenance.json"


def _sourcedata_files_for_subject(target_dir, nemar_id, subject, force_update):
    """Return the ``sourcedata/`` paths belonging to one subject, or ``None``.

    Reads the deposit's ``sourcedata_provenance.json``, which records every
    file's upstream name, size, SHA-256 and -- for deposits enriched after the
    subject field was added -- which subject it came from.

    This is deliberately data-driven rather than pattern-driven. ``sourcedata/``
    preserves whatever layout the authors published, and across the catalogue
    that is genuinely arbitrary: subjects appear as ``sub-A`` (letters),
    ``subject_01``, ``sub1``, ``S10_Session_1``, ``session1/s1/sess01_subj01``
    (subject split across two components), ``subject_01_PC`` / ``_VR`` (two
    files per subject) -- and in at least one deposit the path carries no
    subject at all, just an upstream file id. No glob template spans that, so
    asking the deposit is the only approach that generalises.

    Returns
    -------
    list of str | None
        Include patterns for that subject, or ``None`` when the provenance
        predates the subject field, in which case the caller should fall back
        to fetching the whole tree.
    """
    provenance = target_dir / SOURCEDATA_PROVENANCE
    if force_update or not provenance.is_file():
        try:
            nemar.download(
                dataset=nemar_id,
                target_dir=target_dir,
                scope="sourcedata",
                include=SOURCEDATA_PROVENANCE,
                trust_existing=not force_update,
            )
        except SelectionError as exc:
            # The deposit lists no sourcedata_provenance.json at all -- either
            # it publishes no sourcedata/ or it predates provenance enrichment.
            raise NemarDownloadError(
                f"NEMAR dataset {nemar_id} publishes no sourcedata manifest -- "
                "the original distribution is not mirrored there."
            ) from exc
        except (NemarError, OSError, ConnectionError, TimeoutError, ValueError) as exc:
            raise NemarDownloadError(
                f"Could not fetch the sourcedata manifest for {nemar_id}: {exc}"
            ) from exc
    try:
        record = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NemarDownloadError(
            f"NEMAR dataset {nemar_id} has an unreadable sourcedata manifest."
        ) from exc

    entries = record.get("files") or []
    if not any("subject" in entry for entry in entries):
        return None
    candidates = subject if isinstance(subject, (list, tuple, set)) else [subject]
    wanted = {str(candidate) for candidate in candidates}
    files = [
        entry["file"]
        for entry in entries
        if entry.get("file") and str(entry.get("subject")) in wanted
    ]
    if not files:
        raise NemarDownloadError(
            f"NEMAR dataset {nemar_id} lists no sourcedata for subject "
            f"{subject!r}. Known subjects: "
            f"{sorted({str(e.get('subject')) for e in entries if e.get('subject')})}."
        )
    # Manifest names are literal file paths, but they are consumed as glob
    # patterns (nemar's include and the local verification below) -- escape
    # them so names containing [, ], * or ? still match.
    return ["sourcedata/" + glob_module.escape(name) for name in files]


@verbose
def nemar_sourcedata_dl(
    nemar_id,
    dataset_code,
    path=None,
    force_update=False,
    subject=None,
    include=None,
    verbose=None,
):
    """Download a NEMAR dataset's ``sourcedata/`` and return its local root.

    ``sourcedata/`` holds the **original, pre-BIDS distribution** as published
    by the dataset authors, stored under the upstream filenames. It is
    therefore a drop-in mirror of whatever :meth:`BaseDataset.data_path` would
    otherwise fetch from the upstream host — useful when that host is slow,
    rate-limited, behind a bot gate, or gone.

    Parameters
    ----------
    nemar_id : str
        NEMAR dataset identifier.
    dataset_code : str
        MOABB dataset code used to choose the local dataset directory.
    path : None | str
        Base path where MOABB stores datasets.
    force_update : bool
        Re-fetch even when a local copy is already present.
    subject : int | str | list | None
        Restrict the download to one subject, resolved through the deposit's
        ``sourcedata_provenance.json`` rather than by guessing at the layout.
        A list is treated as aliases for the same subject (e.g. the raw MOABB
        id and its ``nemar_subject_template`` form) and matches any of them.
        Falls back to fetching the whole tree, with a warning, when the deposit
        was enriched before that manifest recorded subjects.
    include : str | list of str | None
        Explicit path glob(s) relative to the dataset root, for callers that
        know the layout. Overrides ``subject``. Without either, the whole
        ``sourcedata/`` tree is downloaded.
    verbose : bool, str, int, or None
        If not None, override default verbose level.

    Returns
    -------
    str
        Local path to the downloaded ``sourcedata`` directory.

    Raises
    ------
    NemarDownloadError
        If the download fails or the deposit publishes no ``sourcedata/``
        at all.
    """
    # Keyed on the NEMAR id, not the MOABB code: one deposit backs several
    # classes (nm000132 is shared by all seven ErpCore2021 subclasses, nm000250
    # by four Dreyer2023 ones), and a per-code path would download the same
    # tree once per class.
    target_dir = nemar_store(dataset_code, nemar_id, path)

    if include is None and subject is not None:
        include = _sourcedata_files_for_subject(
            target_dir, nemar_id, subject, force_update
        )
        if include is None:
            # warnings.warn, not mne.utils.warn: the latter emits a given
            # warning once per session, which makes it order-dependent for
            # anything asserting on it -- and matches what base.py already does
            # for the sibling upstream fallback.
            warnings.warn(
                f"NEMAR dataset {nemar_id} was enriched before its sourcedata "
                "manifest recorded subjects, so one subject cannot be selected; "
                "downloading the whole sourcedata/ tree instead.",
                RuntimeWarning,
                stacklevel=2,
            )

    kwargs = {"include": include} if include is not None else {}
    sourcedata_dir = target_dir / "sourcedata"
    what = f"sourcedata/ matching {include!r}" if include else "sourcedata/"
    missing = (
        f"NEMAR dataset {nemar_id} published no {what} -- the original "
        "distribution is not mirrored there."
    )
    try:
        nemar.download(
            dataset=nemar_id,
            target_dir=target_dir,
            scope="sourcedata",
            trust_existing=not force_update,
            **kwargs,
        )
    except SelectionError as exc:
        # nemar-py raises SelectionError when the scope or the include glob
        # matches nothing, so "this deposit has no sourcedata/" and "this
        # subject is not in it" both arrive here rather than as a successful
        # empty download.
        raise NemarDownloadError(missing) from exc
    except (NemarError, OSError, ConnectionError, TimeoutError, ValueError) as exc:
        # Transport, verification, or S3 failures are not evidence that the
        # deposit publishes no sourcedata -- report them as what they are.
        raise NemarDownloadError(
            f"Could not download {what} for NEMAR dataset {nemar_id}: {exc}"
        ) from exc

    # Ask about the subset this call requested, not the whole tree. `download()`
    # calls this once per subject into a shared directory, so "the directory has
    # something in it" is true from the second subject on even when this
    # subject's fetch matched nothing -- and a before/after diff of the tree
    # would walk every file twice per subject to find that out.
    if include is not None:
        patterns = [include] if isinstance(include, str) else include
        fetched = any(next(target_dir.glob(p), None) is not None for p in patterns)
    else:
        # The manifest does not count: a deposit that publishes only its
        # provenance has no original distribution to offer.
        manifest = target_dir / SOURCEDATA_PROVENANCE
        fetched = sourcedata_dir.is_dir() and any(
            entry != manifest for entry in sourcedata_dir.rglob("*")
        )
    if not fetched:
        raise NemarDownloadError(missing)
    return str(sourcedata_dir)


# This function is from https://github.com/cognoma/figshare (BSD-3-Clause)
def fs_issue_request(method, url, headers, data=None, binary=False):
    """Wrapper for HTTP request.

    Parameters
    ----------
    method : str
        HTTP method. One of GET, PUT, POST or DELETE
    url : str
        URL for the request
    headers: dict
        HTTP header information
    data: dict
        Figshare article data
    binary: bool
        Whether data is binary or not

    Returns
    -------
    response_data: dict
        JSON response for the request returned as python dict
    """
    if data is not None and not binary:
        data = json.dumps(data)

    response = requests.request(method, url, headers=headers, data=data)

    try:
        response.raise_for_status()
        try:
            response_data = json.loads(response.text)
        except ValueError:
            response_data = response.content
    except HTTPError as error:
        logger.error("Caught an HTTPError: {}".format(error))
        logger.error("Body:\n{}".format(response.text))
        raise

    return response_data


def _fs_paginated_file_list(base_url, headers, page_size=1000):
    files = []
    page = 1

    while True:
        page_url = f"{base_url}?page={page}&page_size={page_size}"
        response = fs_issue_request("GET", page_url, headers=headers)

        if isinstance(response, dict):
            page_files = response.get("files", [])
        else:
            page_files = response

        if not page_files:
            break

        files.extend(page_files)
        page += 1

    return files


@functools.lru_cache(maxsize=None)
def fs_get_file_list(article_id, version=None):
    """List all the files associated with a given article.

    Cached in-process by ``(article_id, version)`` to avoid Figshare's 403
    rate limit when callers iterate (clear with ``cache_clear()``).

    Parameters
    ----------
    article_id : str or int
        Figshare article ID
    version : str or int, default is None
        Figshare article version. If None, selects the most recent version.

    Returns
    -------
    response : dict
        HTTP request response as a python dict
    """
    fsurl = "https://api.figshare.com/v2"
    headers = {"Content-Type": "application/json"}

    if version is None:
        url = fsurl + "/articles/{}/files".format(article_id)
    else:
        url = fsurl + "/articles/{}/versions/{}/files".format(article_id, version)

    return _fs_paginated_file_list(url, headers=headers)


def fs_get_file_hash(filelist):
    """Returns a dict associating figshare file id to MD5 hash.

    Parameters
    ----------
    filelist : list of dict
        HTTP request response from fs_get_file_list

    Returns
    -------
    response : dict
        keys are file_id and values are md5 hash
    """
    return {str(f["id"]): "md5:" + f["supplied_md5"] for f in filelist}


def fs_get_file_id(filelist):
    """Returns a dict associating filename to figshare file id.

    Parameters
    ----------
    filelist : list of dict
        HTTP request response from fs_get_file_list

    Returns
    -------
    response : dict
        keys are filename and values are file_id
    """
    return {f["name"]: str(f["id"]) for f in filelist}


def fs_get_file_name(filelist):
    """Returns a dict associating figshare file id to filename.

    Parameters
    ----------
    filelist : list of dict
        HTTP request response from fs_get_file_list

    Returns
    -------
    response : dict
        keys are file_id and values are file name
    """
    return {str(f["id"]): f["name"] for f in filelist}


def download_if_missing(
    file_path, url, warn_missing=True, verbose=True, force_update=False
):
    """Download file from url to a specified path if it is not already there.

    If ``force_update`` is True, an existing local copy is deleted and
    downloaded again.
    """

    folder_path = osp.dirname(file_path)

    # Ensure the folder exists
    if not osp.exists(folder_path):
        if warn_missing:
            warn(f"Directory {folder_path} not found. Creating directory.")
        os.makedirs(folder_path)

    if force_update and osp.exists(file_path):
        os.remove(file_path)

    # Check if file exists, if not download it
    if not osp.exists(file_path):
        if warn_missing:
            warn(f"{file_path} not found. Downloading from {url}")

        downloader = choose_downloader(url, progressbar=verbose)
        if type(downloader).__name__ in ["HTTPDownloader", "DOIDownloader"]:
            downloader.kwargs.setdefault("verify", False)
        _set_user_agent(downloader)

        path = retrieve(
            url,
            None,
            fname=osp.basename(file_path),
            path=osp.dirname(file_path),
            downloader=downloader,
        )

        return path


def create_metainfo_osf(osf_code: str) -> pd.DataFrame:
    """Create a metadata file for a dataset stored on OSF."""
    # OSF API base URL for the project's OSF storage

    base_url = f"https://api.osf.io/v2/nodes/{osf_code}/files/osfstorage/"

    files = []  # to collect (name, url) tuples
    stack = [base_url + "?page[size]=100"]  # start with base URL, up to 100 results

    while stack:
        url = stack.pop()
        try:
            response = requests.get(url)
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            continue

        # Loop through items in this page
        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            kind = attrs.get("kind")
            if kind == "folder":
                # If folder, add its listing URL to stack for later retrieval
                rel = item.get("relationships", {})
                files_rel = rel.get("files", {}) if rel else {}
                folder_url = files_rel.get("links", {}).get("related", {}).get("href")
                if folder_url:
                    # Append page[size]=100 to folder URL as well for efficiency
                    stack.append(folder_url + "?page[size]=100")
            elif kind == "file":
                name = attrs.get("name")
                download_url = item.get("links", {}).get("download")
                if name and download_url:
                    files.append((name, download_url))

        # If there's a next page, add it to stack to continue pagination
        next_url = data.get("links", {}).get("next")
        if next_url:
            stack.append(next_url)

    metainfo = pd.DataFrame(files, columns=["filename", "url"])

    return metainfo
