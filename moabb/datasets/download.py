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
import re
import shutil
import warnings
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import nemar
import pandas as pd
import requests
from httpx import TransportError
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.utils import _url_to_local_path, verbose, warn
from nemar.errors import NemarError, SelectionError
from pooch import file_hash, retrieve
from pooch.downloaders import choose_downloader
from pooch.utils import unique_file_name
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
    trailing segments of the URL path, longest tail first. An explicit
    ``fname`` is probed first, but the URL tails remain fallbacks: a caller's
    ``fname`` names the *destination* (e.g. Lee2024 prefixes the experiment
    directory and zero-pads subject 8), while the store keeps the upstream
    names, so the two can legitimately differ. A tail of two or more segments
    whose file name has no extension also matches a single stored file that
    adds one: deposits keep some extension-less downloads under their content
    type (Mendeley's ``<uuid>/file_downloaded`` as ``file_downloaded.xdf``).
    A miss returns None and the caller downloads.
    """
    store = _ACTIVE_STORE.get()
    if store is None:
        return None
    tails = [PurePosixPath(fname).parts] if fname is not None else []
    parts = PurePosixPath(urlparse(url).path).parts
    # ponytail: four trailing segments cover every current dataset layout
    # (Thielen2021: offline/sub-01/block_1/trainlabels.mat); deepen if a
    # deposit ever nests further.
    tails += [parts[i:] for i in range(max(len(parts) - 4, 0), len(parts))]
    for tail in tails:
        if not tail:
            continue
        candidate = Path(store).joinpath(*tail)
        if not candidate.is_file() and len(tail) > 1 and not candidate.suffix:
            variants = (
                [
                    path
                    for path in candidate.parent.glob(
                        glob_module.escape(candidate.name) + ".*"
                    )
                    if path.is_file()
                ]
                if candidate.parent.is_dir()
                else []
            )
            if len(variants) == 1:
                candidate = variants[0]
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


#: HTTPS mirror of PhysioNet's ``/files/`` tree on the AWS Open Data
#: registry. Verified byte-identical for ``eegmmidb`` and ``bigp3bci``
#: (spot-checked EDFs plus ``.event``/``SHA256SUMS.txt`` probes, Oct 2026).
#: ``s3://physionet-open/...`` and this HTTPS endpoint serve the same
#: objects; HTTPS is used because pooch only speaks http(s)/ftp/sftp/doi.
PHYSIONET_S3_MIRROR_BASE = "https://physionet-open.s3.amazonaws.com"
_PHYSIONET_UPSTREAM_PREFIX = "https://physionet.org/files/"


def _s3_mirror_enabled() -> bool:
    """Whether the PhysioNet S3 HTTPS mirror may be tried first.

    Enabled by default; set ``MOABB_S3_MIRROR=0`` (or ``false``/``off``/
    ``disabled``) to force direct upstream downloads.
    """
    return os.environ.get("MOABB_S3_MIRROR", "1").lower() not in (
        "0",
        "false",
        "off",
        "no",
        "disabled",
    )


def physionet_s3_mirror(url: str) -> str | None:
    """Return the AWS Open Data HTTPS mirror for a PhysioNet ``/files/`` URL.

    Only rewrites the ``https://physionet.org/files/<dataset>/...`` tree,
    which the ``physionet-open`` bucket mirrors 1:1. Returns ``None`` for
    any other URL (including ``archive.physionet.org``), so callers can
    fall back to the upstream host untouched.
    """
    if not isinstance(url, str):
        return None
    if not url.startswith(_PHYSIONET_UPSTREAM_PREFIX):
        return None
    rest = url[len(_PHYSIONET_UPSTREAM_PREFIX) :]
    if not rest or ".." in rest.split("/"):
        return None
    return PHYSIONET_S3_MIRROR_BASE + "/" + rest


# Set after the first mirror failure in this process: later files go straight
# upstream instead of paying one timeout and one warning per file.
_S3_MIRROR_FAILED = False


def _retrieve_with_physionet_fallback(
    url, *, fname, path, known_hash=None, progressbar=True, downloader=None
):
    """Retrieve ``url`` preferring the PhysioNet S3 mirror on failure-prone hosts.

    The destination is always ``path/fname`` (derived from the upstream URL
    by the caller), so a mirror hit lands exactly where the loader expects
    and pooch skips re-download afterwards. On any mirror transport failure
    this warns, retries the upstream URL and skips the mirror for the rest
    of the process; the upstream error (if it also fails) is the one raised.
    """
    global _S3_MIRROR_FAILED
    use_mirror = _s3_mirror_enabled() and not _S3_MIRROR_FAILED
    mirror = physionet_s3_mirror(url) if use_mirror else None
    if mirror is not None:
        # TLS verification stays on: the ``verify=False`` workaround the
        # callers apply is for upstream hosts with broken chains, not AWS.
        mirror_downloader = choose_downloader(mirror, progressbar=progressbar)
        _set_user_agent(mirror_downloader)
        try:
            return retrieve(
                mirror,
                known_hash,
                fname=fname,
                path=path,
                progressbar=progressbar,
                downloader=mirror_downloader,
            )
        except Exception as exc:
            _S3_MIRROR_FAILED = True
            # warnings.warn, not mne.utils.warn: the latter is silent when the
            # MNE log level is above WARNING, and users should see the fallback.
            warnings.warn(
                f"S3 mirror download failed for {mirror} ({exc!r}); "
                f"falling back to upstream {url} for this and later files. "
                "Set MOABB_S3_MIRROR=0 to skip the mirror.",
                RuntimeWarning,
                stacklevel=2,
            )
    return retrieve(
        url,
        known_hash,
        fname=fname,
        path=path,
        progressbar=progressbar,
        downloader=downloader,
    )


def _sanitize_path(path: Path) -> Path:
    path = Path(path)
    table = {ord(c): "-" for c in '~:*?"<>|'}

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
    # retrieve() below treats `destination` as a directory and stores the file
    # inside it under pooch's unique name; several loaders then listdir() that
    # directory. Fill the store copy at the same wrapped path -- a plain file
    # at `destination` would shadow the directory and break those loaders.
    wrapped = osp.join(destination, unique_file_name(url))
    if not force_update and not osp.isfile(wrapped):
        mirror = _store_lookup(url)
        if mirror is not None:
            os.makedirs(destination, exist_ok=True)
            try:
                os.link(mirror, wrapped)
            except OSError:
                shutil.copy2(mirror, wrapped)
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
        # Raw (unsanitized) on purpose: this lookup exists to FIND
        # pre-sanitization files -- e.g. the literal '~bci' components
        # mirrored from https://lampx.tugraz.at/~bci/... URLs -- and
        # migrate them to the sanitized destination below.
        legacy_destination = Path(_url_to_local_path(url, root))
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
    dlpath = _retrieve_with_physionet_fallback(
        url,
        fname=destination.name,
        path=str(destination.parent),
        known_hash=known_hash,
        progressbar=True,
        downloader=downloader,
    )
    return dlpath


def data_dl_destination(url, sign, path=None, fname=None):
    """Where :func:`data_dl` stores ``url``, computed without any network.

    Lets a loader locate the directory an archive extracts into before deciding
    whether the archive has to be downloaded at all.
    """
    root = Path(get_dataset_path(sign, path)) / "MNE-{:s}-data".format(sign.lower())
    if fname is not None:
        return _sanitize_path(root / fname)
    return _sanitize_path(_normalize_destination(url, root))


def fill_from_store(base_dir, files):
    """Place archive members from the active NEMAR store; True if all are there.

    NEMAR deposits mirror an archive's *extracted* members, not the archive, so
    a URL lookup for the archive (see :func:`data_dl`) never matches. A loader
    about to download an archive only to extract ``files`` below ``base_dir``
    asks here first: each file is looked up in the store under its path
    relative to ``base_dir`` -- the layout the deposit was taken from.

    All or nothing: files are linked only when the store (or the cache) holds
    every one of them, so a partial deposit never leaves a half-extracted tree
    that the loader would mistake for a complete one, and an archive extracted
    later never writes through a link into the store.

    Parameters
    ----------
    base_dir : str | pathlib.Path
        Directory the archive extracts into.
    files : list of str | pathlib.Path
        Absolute paths, below ``base_dir``, of the members the loader reads.

    Returns
    -------
    bool
        True when every file exists afterwards.
    """
    base_dir = Path(base_dir)
    store = _ACTIVE_STORE.get()
    pending = []
    for target in map(Path, files):
        if target.is_file():
            continue
        if store is None:
            return False
        try:
            member = target.relative_to(base_dir)
        except ValueError:
            return False
        source = Path(store, *PurePosixPath(member.as_posix()).parts)
        if not source.is_file():
            return False
        pending.append((source, target))
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    if pending:
        logger.info(
            "Filled %d archive member(s) in %s from the local NEMAR store.",
            len(pending),
            base_dir,
        )
    return True


def fill_tree_from_store(base_dir, member_dir):
    """Place a whole extracted directory from the active NEMAR store.

    :func:`fill_from_store` for loaders that read every file an archive
    extracts below one directory (e.g. one per subject) instead of a fixed
    list: the members are the files the store holds below ``member_dir``.
    Recordings whose headers name files the store lacks (a BrainVision
    ``.vhdr`` without its ``.eeg``/``.vmrk``, an EEGLAB ``.set`` without its
    ``.fdt``) mean the directory is incomplete, and nothing is placed.

    Parameters
    ----------
    base_dir : str | pathlib.Path
        Directory the archive extracts into.
    member_dir : str
        The directory, relative to ``base_dir``, the loader reads.

    Returns
    -------
    bool
        True when ``base_dir / member_dir`` holds every file the store has
        there afterwards; False without an active store or when the store
        does not hold a complete copy of the directory.
    """
    store = _ACTIVE_STORE.get()
    if store is None:
        return False
    store = Path(store)
    source_dir = store.joinpath(*PurePosixPath(Path(member_dir).as_posix()).parts)
    if not source_dir.is_dir():
        return False
    names = sorted(
        path.relative_to(store).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file()
    )
    if not names or _missing_companions(store, names):
        return False
    return fill_from_store(base_dir, [Path(base_dir, name) for name in names])


def active_store():
    """The active local sourcedata store root, or None."""
    store = _ACTIVE_STORE.get()
    return Path(store) if store is not None else None


def store_glob(pattern):
    """Names in the active NEMAR store matching ``pattern``, relative to its root.

    For loaders that locate a member by pattern rather than by a fixed name
    (e.g. a subject directory named after its unknown recording date), before
    handing the match to :func:`fill_tree_from_store`. Empty without an active
    store.
    """
    store = _ACTIVE_STORE.get()
    if store is None:
        return []
    store = Path(store)
    return sorted(path.relative_to(store).as_posix() for path in store.glob(pattern))


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
    except (NemarError, TransportError, OSError, ValueError) as exc:
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
        except (NemarError, TransportError, OSError, ValueError) as exc:
            raise NemarDownloadError(
                f"Could not fetch the sourcedata manifest for {nemar_id}: {exc}"
            ) from exc
    try:
        record = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NemarDownloadError(
            f"NEMAR dataset {nemar_id} has an unreadable sourcedata manifest."
        ) from exc
    files = _manifest_subject_files(record, subject)
    if files is None:
        return None
    if not files:
        known = sorted(
            {
                str(entry["subject"])
                for entry in record.get("files") or []
                if isinstance(entry, dict) and entry.get("subject")
            }
        )
        raise NemarDownloadError(
            f"NEMAR dataset {nemar_id} lists no sourcedata for subject "
            f"{subject!r}. Known subjects: {known}."
        )
    # Files the manifest attributes to no subject (montages, band/interval
    # definitions, a performance table) are read by every subject's load, so
    # they belong to every subject's selection.
    files = files + _manifest_shared_files(record)
    # Manifest names are literal file paths, but they are consumed as glob
    # patterns (nemar's include and the local verification below) -- escape
    # them so names containing [, ], * or ? still match.
    return ["sourcedata/" + glob_module.escape(name) for name in files]


def _sourcedata_store_holds_data(target_dir):
    """Whether a deposit's ``sourcedata/`` holds anything besides its manifest.

    The manifest lands inside the store, so it does not count: a store holding
    only a manifest is not a fetched one.

    Uses :func:`os.walk` and looks only at ``filenames``. ``Path.rglob`` would
    also match directories, so an interrupted fetch that left empty ``sub-*``
    directories behind would count as a fetched store.
    """
    manifest_name = PurePosixPath(SOURCEDATA_PROVENANCE).name
    for _root, _dirs, filenames in os.walk(target_dir / "sourcedata"):
        if any(name != manifest_name for name in filenames):
            return True
    return False


def _manifest_subject_files(record, subject):
    """The files a provenance record attributes to ``subject``.

    ``None`` when the manifest predates the ``subject`` field and so cannot
    answer per subject; an empty list when it can answer but knows nothing
    about this subject. Names are returned raw -- callers that feed them to a
    glob escape them themselves.
    """
    entries = [entry for entry in (record.get("files") or []) if isinstance(entry, dict)]
    if not any("subject" in entry for entry in entries):
        return None
    candidates = subject if isinstance(subject, (list, tuple, set)) else [subject]
    wanted = {str(candidate) for candidate in candidates}
    return [
        entry["file"]
        for entry in entries
        if entry.get("file") and str(entry.get("subject")) in wanted
    ]


def _manifest_shared_files(record):
    """The files a per-subject provenance record attributes to no subject.

    Only meaningful once the manifest records subjects at all: then an entry
    whose ``subject`` is missing or empty is shared by every subject's load
    (Brandl2020's ``mnt.mat`` montage, Dreyer2023's ``performance.csv``).
    Names are returned raw.
    """
    entries = [entry for entry in (record.get("files") or []) if isinstance(entry, dict)]
    if not any("subject" in entry for entry in entries):
        return []
    return [
        entry["file"]
        for entry in entries
        if entry.get("file") and entry.get("subject") in (None, "")
    ]


def _is_stored_file(sourcedata_dir, name):
    """Whether ``name`` from a manifest names a real file inside the store.

    Manifest names are data, so they are not trusted to stay relative: an
    absolute one would make ``/`` discard the store root entirely, and a
    directory must not pass for a downloaded file.
    """
    return not Path(name).is_absolute() and (sourcedata_dir / name).is_file()


def nemar_sourcedata_is_local(target_dir, subject):
    """Whether a subject's ``sourcedata/`` is already on disk, without any network.

    Answering per subject is what lets a caller loading one subject at a time
    skip only what it really has; when the manifest cannot answer, a store that
    already holds data is trusted as-is.

    This has to settle offline because ``nemar.download`` walks index, version,
    metadata and manifest before it ever consults ``trust_existing`` -- a call
    with nothing to do still costs four round-trips.

    Parameters
    ----------
    target_dir : pathlib.Path
        The deposit's local root, as returned by :func:`nemar_store`. Taken
        already resolved because resolving it re-reads MNE's config file, which
        a per-subject caller would otherwise pay for on every subject.
    subject : int | str | list
        Subject identifier, or the identifiers a deposit may file it under.
    """
    target_dir = Path(target_dir)
    try:
        record = json.loads(
            (target_dir / SOURCEDATA_PROVENANCE).read_text(encoding="utf-8")
        )
        files = _manifest_subject_files(record, subject)
    except (OSError, ValueError, AttributeError, TypeError):
        # Unreadable, or a manifest whose schema has drifted -- either way it
        # cannot answer, so degrade to the whole-store rule rather than raising
        # out of get_data().
        files = None
    if files is None:
        return _sourcedata_store_holds_data(target_dir)
    if not files:
        return False
    sourcedata_dir = target_dir / "sourcedata"
    return all(
        _is_stored_file(sourcedata_dir, name)
        for name in files + _manifest_shared_files(record)
    )


_LOADER_CACHE_DIR = re.compile(r"MNE-(.+)-data")


def _loader_cache_root(record, path=None):
    """Where, on this machine, the loader keeps the files a deposit mirrors.

    A deposit's ``sourcedata/`` is a snapshot of the loader's own cache after
    ``data_path()`` ran -- archives already extracted, opaque download URLs
    already resolved to file names -- and its provenance records that cache
    directory as ``upstream_cache_root``, e.g.
    ``/scratch/.../mne_data/MNE-Simoes2020-data/BCIAUT_P300``. The manifest
    names files relative to it. Rebase everything from the ``MNE-<sign>-data``
    component on this machine's dataset path for that sign.

    Returns None when the manifest records no such root, or one that would
    leave the dataset directory.
    """
    root = record.get("upstream_cache_root")
    if not isinstance(root, str) or not root:
        return None
    parts = PurePosixPath(root.replace("\\", "/")).parts
    for index in range(len(parts) - 1, -1, -1):
        match = _LOADER_CACHE_DIR.fullmatch(parts[index])
        if match is None:
            continue
        tail = parts[index:]
        if any(part in ("..", ".") for part in tail):
            return None
        return Path(get_dataset_path(match.group(1), path)).joinpath(*tail)
    return None


_BRAINVISION_COMPANION = re.compile(r"^\s*(?:DataFile|MarkerFile)\s*=\s*(.+?)\s*$", re.M)


def _eeglab_data_file(header):
    """The ``.fdt`` an EEGLAB ``.set`` stores its samples in, or None if embedded.

    Raises when the file cannot be read as a MATLAB v5 ``.set``.
    """
    from scipy.io import loadmat, whosmat

    names = {name for name, _shape, _cls in whosmat(header)}
    if "data" in names:  # EEGLAB >= 2019 saves the EEG fields at top level
        data = loadmat(header, variable_names=["data"], squeeze_me=True)["data"]
    else:
        eeg = loadmat(
            header, variable_names=["EEG"], squeeze_me=True, struct_as_record=False
        )["EEG"]
        data = eeg.data
    return data if isinstance(data, str) else None


def _missing_companions(sourcedata_dir, files):
    """Files that the store's recording headers name but the store lacks.

    Some deposits mirror only part of a recording: the ``.vhdr`` of a
    BrainVision set without the ``.eeg``/``.vmrk`` it points to, an EEGLAB
    ``.set`` without its ``.fdt``, or Curry ``.cdt`` samples without their
    ``.cdt.dpa``/``.cdt.dpo`` header. An EEGLAB header that cannot be read
    counts as incomplete.
    """
    missing = set()
    for name in files:
        if name.lower().endswith(".cdt"):
            # Curry 8 keeps the header next to the samples (.cdt.dpa/.cdt.dpo).
            data = sourcedata_dir / name
            if data.is_file() and not any(
                Path(f"{data}{suffix}").is_file() for suffix in (".dpa", ".dpo")
            ):
                missing.add(f"{name}.dpa")
            continue
        if name.lower().endswith(".set"):
            header = sourcedata_dir / name
            if header.with_suffix(".fdt").is_file() or not header.is_file():
                continue
            try:
                data_file = _eeglab_data_file(header)
            except Exception:  # noqa: BLE001 - unreadable header: do not place it
                missing.add(str(PurePosixPath(name).with_suffix(".fdt")))
                continue
            if data_file and not (header.parent / data_file).is_file():
                missing.add(str(PurePosixPath(name).parent / data_file))
            continue
        if not name.lower().endswith(".vhdr"):
            continue
        header = sourcedata_dir / name
        try:
            text = header.read_text(encoding="latin-1")
        except OSError:
            continue
        for companion in _BRAINVISION_COMPANION.findall(text):
            if (header.parent / companion).is_file():
                continue
            # Some deposits rename the recordings to BIDS names but leave the
            # header naming the pre-rename file (Thapa2025 subject 9). The
            # loaders that read such deposits override the data/marker file
            # names with the header's own stem, so a same-suffix companion
            # under that stem satisfies the reference.
            stem = PurePosixPath(name).stem
            if any(
                PurePosixPath(f).parent == PurePosixPath(name).parent
                and PurePosixPath(f).stem == stem
                and PurePosixPath(f).suffix == PurePosixPath(companion).suffix
                for f in files
            ):
                continue
            missing.add(str(PurePosixPath(name).parent / companion))
    return missing


def seed_loader_cache(target_dir, subject, path=None):
    """Link a subject's NEMAR sourcedata into the loader's own cache tree.

    :func:`_store_lookup` serves a download only when the store holds the file
    under the URL's trailing segments. Loaders that download an archive and
    extract it, or that fetch opaque URLs (OSF file ids, DSpace bitstream
    uuids, figshare ``ndownloader`` ids), never match: the store holds the
    extracted or resolved files instead. Those loaders do check their cache
    before downloading, so placing the store's files where the loader itself
    would have left them -- the layout the deposit was snapshotted from -- lets
    them load without contacting the upstream host.

    Files are hard-linked (copied across filesystems), never overwrite an
    existing cache file, and are confined to the rebased cache root.

    Parameters
    ----------
    target_dir : pathlib.Path
        The deposit's local root, as returned by :func:`nemar_store`.
    subject : int | str | list
        Subject identifier, or the identifiers a deposit may file it under.
    path : None | str
        Base path where MOABB stores datasets.

    Returns
    -------
    int
        Number of files placed in the cache.
    """
    target_dir = Path(target_dir)
    try:
        record = json.loads(
            (target_dir / SOURCEDATA_PROVENANCE).read_text(encoding="utf-8")
        )
        entries = [e for e in record.get("files") or [] if isinstance(e, dict)]
        files = _manifest_subject_files(record, subject)
        if files is None:
            files = [entry["file"] for entry in entries if entry.get("file")]
        else:
            # Files no subject owns (montages, shared tables) serve every
            # subject's load; placed whenever the store holds them.
            files = files + [
                entry["file"]
                for entry in entries
                if entry.get("file") and entry.get("subject") in (None, "")
            ]
        cache_root = _loader_cache_root(record, path)
    except (OSError, ValueError, AttributeError, TypeError):
        return 0
    if cache_root is None:
        return 0
    sourcedata_dir = target_dir / "sourcedata"
    resolved_store = sourcedata_dir.resolve()
    resolved_root = cache_root.resolve()
    pending = []
    for name in files:
        if not _is_stored_file(sourcedata_dir, name):
            continue
        destination = cache_root / name
        # Manifest names are data: keep both ends inside their trees.
        if not (sourcedata_dir / name).resolve().is_relative_to(resolved_store):
            continue
        if not destination.resolve().is_relative_to(resolved_root):
            continue
        if not destination.exists():
            pending.append((sourcedata_dir / name, destination))
    if not pending:
        return 0
    # A partial recording in the cache is worse than none: a loader that only
    # tests "is the extracted tree there?" would skip its own download and then
    # load a fraction of the subject, or fail on an absent file.
    missing = _missing_companions(sourcedata_dir, files)
    if missing:
        logger.warning(
            "Not placing NEMAR store files in %s: the deposit lacks %s, which "
            "its headers reference; the dataset's own downloader is used.",
            cache_root,
            ", ".join(sorted(missing)[:3]) + (" ..." if len(missing) > 3 else ""),
        )
        return 0
    for source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    logger.info(
        "Placed %d file(s) from the local NEMAR store in %s.", len(pending), cache_root
    )
    return len(pending)


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
    except (NemarError, TransportError, OSError, ValueError) as exc:
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
        fetched = _sourcedata_store_holds_data(target_dir)
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
    downloaded again. Like :func:`data_dl`, a missing file is first taken
    from the active NEMAR store when it holds it under the URL's trailing
    path segments (e.g. a ``SHA256SUMS.txt`` index at the root of the
    deposit's ``sourcedata/``).
    """

    folder_path = osp.dirname(file_path)

    # Ensure the folder exists
    if not osp.exists(folder_path):
        if warn_missing:
            warn(f"Directory {folder_path} not found. Creating directory.")
        os.makedirs(folder_path)

    if force_update and osp.exists(file_path):
        os.remove(file_path)

    if not force_update and not osp.exists(file_path):
        mirror = _store_lookup(url)
        if mirror is not None:
            try:
                os.link(mirror, file_path)
            except OSError:
                shutil.copy2(mirror, file_path)
            return str(file_path)

    # Check if file exists, if not download it
    if not osp.exists(file_path):
        if warn_missing:
            warn(f"{file_path} not found. Downloading from {url}")

        downloader = choose_downloader(url, progressbar=verbose)
        if type(downloader).__name__ in ["HTTPDownloader", "DOIDownloader"]:
            downloader.kwargs.setdefault("verify", False)
        _set_user_agent(downloader)

        path = _retrieve_with_physionet_fallback(
            url,
            fname=osp.basename(file_path),
            path=osp.dirname(file_path),
            downloader=downloader,
        )

        return path


def create_metainfo_osf(osf_code: str) -> pd.DataFrame:
    """List every file of an OSF project's OSF storage.

    Parameters
    ----------
    osf_code : str
        OSF project (node) id, e.g. ``"r7s9b"``.

    Returns
    -------
    metainfo : pandas.DataFrame
        One row per file with columns ``filename`` (bare file name), ``url``
        (download link) and ``path`` (the file's full path inside the
        project, e.g. ``/sourcedata/sub-01/ses-01/eeg/sub-01_eeg.vhdr``).
        A project can hold several files with the same name in different
        folders; ``path`` tells them apart, ``filename`` does not. The row
        order follows the OSF listing and is not guaranteed.
    """
    # OSF API base URL for the project's OSF storage

    base_url = f"https://api.osf.io/v2/nodes/{osf_code}/files/osfstorage/"

    files = []  # to collect (name, url, path) tuples
    # (listing URL, folder path) pairs; start with the root, up to 100 results.
    # The folder path is a fallback for items without ``materialized_path``.
    stack = [(base_url + "?page[size]=100", "/")]

    while stack:
        url, folder_path = stack.pop()
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
            name = attrs.get("name")
            item_path = attrs.get("materialized_path")
            if kind == "folder":
                # If folder, add its listing URL to stack for later retrieval
                rel = item.get("relationships", {})
                files_rel = rel.get("files", {}) if rel else {}
                folder_url = files_rel.get("links", {}).get("related", {}).get("href")
                if folder_url:
                    sub_path = item_path or f"{folder_path}{name}/"
                    if not sub_path.endswith("/"):
                        sub_path += "/"
                    # Append page[size]=100 to folder URL as well for efficiency
                    stack.append((folder_url + "?page[size]=100", sub_path))
            elif kind == "file":
                download_url = item.get("links", {}).get("download")
                if name and download_url:
                    files.append(
                        (name, download_url, item_path or f"{folder_path}{name}")
                    )

        # If there's a next page, add it to stack to continue pagination
        next_url = data.get("links", {}).get("next")
        if next_url:
            stack.append((next_url, folder_path))

    metainfo = pd.DataFrame(files, columns=["filename", "url", "path"])

    return metainfo
