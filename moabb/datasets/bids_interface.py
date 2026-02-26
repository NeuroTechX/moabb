"""BIDS Interface for MOABB.

========================
This module contains the BIDS interface for MOABB, which allows to convert
any MOABB dataset to BIDS with Cache.
We can convert at the Raw, Epochs or Array level.
"""

# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import abc
import datetime
import json
import logging
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type

import mne
import mne_bids
from numpy import load as np_load
from numpy import save as np_save

import moabb
from moabb.analysis.results import get_digest
from moabb.datasets import download as dl


if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)


def get_bids_root(code, path=None):
    """Path to the root of the BIDS structure used for caching.

    See :class:`moabb.datasets.base.BaseDataset` and
    :class:`moabb.datasets.base.CacheConfig` for more information
     on the MOABB caching mechanism.

    Parameters
    ----------
    code : str
        The dataset code from the MOABB dataset.
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.

    Returns
    -------
    root : Path
        Path to the root of the BIDS structure.
    """

    mne_path = Path(dl.get_dataset_path(code, path))
    cache_dir = f"MNE-BIDS-{camel_to_kebab_case(code)}"
    root = mne_path / cache_dir
    return root


def camel_to_kebab_case(name):
    """Converts a CamelCase string to kebab-case."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1-\2", name).lower()


def subject_moabb_to_bids(subject: int):
    """Convert the subject number to string (subject)."""
    return str(subject)


def subject_bids_to_moabb(subject: str):
    """Convert the subject string to int(subject)."""
    return int(subject)


def run_moabb_to_bids(run: str):
    """Convert the run to run index plus eventually description."""
    p = r"([0-9]+)(|[a-zA-Z]+[a-zA-Z0-9]*)"
    idx, desc = re.fullmatch(p, run).groups()
    out = {"run": idx}
    if desc:
        out["recording"] = desc
    return out


def run_bids_to_moabb(path: mne_bids.BIDSPath):
    """Extracts the run index plus eventually description from a path."""
    if path.recording is None:
        return path.run
    return f"{path.run}{path.recording}"


@dataclass
class BIDSInterfaceBase(abc.ABC):
    """Base class for BIDSInterface.

    This dataclass is used to convert a MOABB dataset to MOABB BIDS.
    It is used by the ``get_data`` method of any MOABB dataset.

    Parameters
    ----------
    dataset : BaseDataset
        The dataset to convert.
    subject : int
        The subject to convert.
    path : str
        The path to the BIDS dataset.
    process_pipeline : Pipeline
        The processing pipeline used to convert the data.
    verbose : str
        The verbosity level.

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    dataset: "BaseDataset"
    subject: int
    path: str = None
    process_pipeline: "Pipeline" = None
    verbose: str = None
    _dataset_type: str = "derivative"

    @property
    def processing_params(self):
        """Return the processing parameters."""
        # TODO: add dataset kwargs
        return self.process_pipeline

    @property
    def desc(self):
        """Return the description of the processing pipeline."""
        return get_digest(self.processing_params)

    def __repr__(self):
        """Return the representation of the BIDSInterface."""
        desc = self.desc
        desc_str = f"{desc:.7}" if desc is not None else "None"
        return (
            f"{self.dataset.code!r} sub-{self.subject} "
            f"suffix-{self._suffix} desc-{desc_str}"
        )

    @property
    def root(self):
        """Return the root path of the BIDS dataset."""
        return get_bids_root(self.dataset.code, self.path)

    def _lock_file(self, session):
        """Return the lock file path for a specific session.

        This file is saved after writing all runs for a session to ensure
        the session's data was completely saved. It is stored in the
        ``code/`` folder of the BIDS dataset root, which is
        BIDS-validator exempt.
        """
        return (
            self.root
            / "code"
            / f"sub-{subject_moabb_to_bids(self.subject)}_ses-{session}_desc-{self.desc}_lockfile.json"
        )

    @property
    def _migration_lock_file(self):
        """Per-subject lock file used for backward compatibility.

        This was the lock file format used between the initial BIDS caching
        implementation migration to the ``code/`` folder and the switch to
        per-session lock files.
        """
        return (
            self.root
            / "code"
            / f"sub-{subject_moabb_to_bids(self.subject)}_desc-{self.desc}_lockfile.json"
        )

    @property
    def _legacy_lock_file(self):
        """Return the legacy lock file path for backward compatibility.

        In the original implementation, the lock file was stored inside the
        subject folder of the BIDS structure. This property allows loading
        caches that were created with the old path.
        """
        return mne_bids.BIDSPath(
            root=self.root,
            subject=subject_moabb_to_bids(self.subject),
            description=self.desc,
            extension=".json",
            suffix="lockfile",  # necessary for unofficial files
            check=False,
        )

    def erase(self):
        """Erase the cache of the subject if it exists."""
        log.info("Starting erasing cache of %s...", repr(self))

        if not self.root.exists():
            log.info("No cache directory at %s, nothing to erase.", self.root)
            return

        # Find all matching paths to determine which sessions exist
        paths = mne_bids.find_matching_paths(
            root=self.root,
            subjects=subject_moabb_to_bids(self.subject),
            descriptions=self.desc,
            check=self._check,
            suffixes=self._suffix,
            extensions=self._extension,
        )
        sessions = set(p.session for p in paths)

        # Remove lock files FIRST, before calling session_path.rm(). In some
        # versions of mne_bids, rm() globs all files under root and finds our
        # lock files (named with BIDS entity syntax). It then derives a wrong
        # "canonical" BIDS path and tries to unlink a non-existent file.
        code_dir = self.root / "code"
        if code_dir.exists():
            pattern = f"sub-{subject_moabb_to_bids(self.subject)}_ses-*_desc-{self.desc}_lockfile.json"
            for lock_file in code_dir.glob(pattern):
                lock_file.unlink()
        # Remove migration-style per-subject lock file if present
        if self._migration_lock_file.exists():
            self._migration_lock_file.unlink()
        # Remove original legacy lock file if present
        legacy = self._legacy_lock_file
        if legacy.fpath.exists():
            legacy.fpath.unlink()

        # Remove data files per session to avoid mne_bids failing when
        # looking up scans.tsv across multiple sessions.  Note: mne_bids
        # rm() automatically calls rmtree on the subject directory when
        # the last session is removed (i.e. no remaining files under
        # sub-{subject}/), so empty directories are cleaned up.
        for session in sessions:
            session_path = mne_bids.BIDSPath(
                root=self.root,
                subject=subject_moabb_to_bids(self.subject),
                session=session,
                description=self.desc,
                check=False,
            )
            session_path.rm(safe_remove=False)
        log.info("Finished erasing cache of %s.", repr(self))

    def load(self, preload=False):
        """Load the cache of the subject if it exists and returns it as
        a nested dictionary with the following structure::

            sessions_data = {'session_id':
                        {'run_id': run}
                    }

        If the cache is not present, returns None.
        """
        log.info("Attempting to retrieve cache of %s...", repr(self))
        code_dir = self.root / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # Check for non-session-aware legacy lock files (backward compatibility)
        legacy_lock_exists = (
            self._migration_lock_file.exists() or self._legacy_lock_file.fpath.exists()
        )
        # Ensure the legacy BIDSPath directory exists for mne_bids compatibility
        self._legacy_lock_file.mkdir(exist_ok=True)

        paths = mne_bids.find_matching_paths(
            root=self.root,
            subjects=subject_moabb_to_bids(self.subject),
            descriptions=self.desc,
            extensions=self._extension,
            check=self._check,
            # datatypes="eeg", # commented for compatibility with cache saved in previous versions
            suffixes=self._suffix,
        )

        if not paths:
            log.info("No cache found at %s.", str(code_dir))
            return None

        # Check per-session lock files unless a legacy (non-session-aware) lock
        # file exists, which indicates the whole subject was already cached.
        if not legacy_lock_exists:
            found_sessions = {path.session for path in paths}
            missing = [s for s in found_sessions if not self._lock_file(s).exists()]
            if missing:
                log.info("No cache found at %s.", str(code_dir))
                return None

        sessions_data = {}
        for path in paths:
            session_moabb = path.session
            session = sessions_data.setdefault(session_moabb, {})
            run = self._load_file(path, preload=preload)
            session[run_bids_to_moabb(path)] = run
        log.info("Finished reading cache of %s", repr(self))
        return sessions_data

    def save(self, sessions_data):
        """Save the cache of the subject.
        The data to be saved should be a nested dictionary
        with the following structure::

            sessions_data = {'session_id':
                        {'run_id': run}
                    }

        If a ``run`` is None, it will be skipped.

        The type of the ``run`` object can vary (see the subclases).
        """
        log.info("Starting caching %s", repr(self))
        mne_bids.BIDSPath(root=self.root).mkdir(exist_ok=True)

        lock_data = dict(processing_params=str(self.processing_params))
        for session, runs in sessions_data.items():
            for run, obj in runs.items():
                if obj is None:
                    log.warning(
                        "Skipping caching %s session %s run %s because it is None.",
                        repr(self),
                        session,
                        run,
                    )
                    continue

                run_kwargs = run_moabb_to_bids(run)
                bids_path = mne_bids.BIDSPath(
                    root=self.root,
                    subject=subject_moabb_to_bids(self.subject),
                    session=session,
                    task=self.dataset.paradigm,
                    **run_kwargs,
                    description=self.desc,
                    extension=self._extension,
                    datatype="eeg",
                    suffix=self._suffix,
                    check=self._check,
                )

                bids_path.mkdir(exist_ok=True)
                self._write_file(bids_path, obj)

            self._write_lock_file(session, lock_data)

        # Write dataset_description.json after all files so that it
        # overwrites any version created internally by mne_bids.write_raw_bids.
        source_datasets = []
        if self.dataset.doi is not None:
            source_datasets = [dict(DOI=self.dataset.doi)]
        mne_bids.make_dataset_description(
            path=str(self.root),
            name=self.dataset.code,
            dataset_type=self._dataset_type,
            generated_by=[
                dict(
                    CodeURL="https://github.com/NeuroTechX/moabb",
                    Name="moabb",
                    Description="Mother of All BCI Benchmarks",
                    Version=moabb.__version__,
                )
            ],
            source_datasets=source_datasets,
            overwrite=True,
            verbose=self.verbose,
        )
        log.info("Finished caching %s to disk.", repr(self))

    def _write_lock_file(self, session, lock_data):
        """Write the lock file for a session to signal that saving is complete."""
        lock_file = self._lock_file(session)
        log.debug("Writing %s", lock_file)
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with lock_file.open("w") as file:
            json.dump(lock_data, file)

    @abc.abstractmethod
    def _load_file(self, bids_path, preload):
        pass

    @abc.abstractmethod
    def _write_file(self, bids_path, obj):
        pass

    @property
    @abc.abstractmethod
    def _extension(self):
        pass

    @property
    @abc.abstractmethod
    def _check(self):
        pass

    @property
    @abc.abstractmethod
    def _suffix(self):
        pass


_FORMAT_EXTENSION_MAP = {
    "EDF": ".edf",
    "BrainVision": ".vhdr",
    "BDF": ".bdf",
    "EEGLAB": ".set",
}


class BIDSInterfaceRawEDF(BIDSInterfaceBase):
    """BIDS Interface for Raw EEG files.

    In this case, the ``run`` object (see the ``save()`` method)
    is expected to be an ``mne.io.BaseRaw`` instance."""

    _format = "EDF"

    @property
    def _extension(self):
        return _FORMAT_EXTENSION_MAP[self._format]

    @property
    def _check(self):
        return True

    @property
    def _suffix(self):
        return "eeg"

    def _load_file(self, bids_path, preload):
        raw = mne_bids.read_raw_bids(
            bids_path, extra_params=dict(preload=preload), verbose=self.verbose
        )
        return raw

    def _write_file(self, bids_path, raw):
        if not raw.annotations:
            raise ValueError(
                "Raw object must have annotations to be saved in BIDS format."
                "Use the SetRawAnnotations pipeline for this."
            )
        datetime_now = datetime.datetime.now(tz=datetime.timezone.utc)
        if raw.info.get("line_freq", None) is None:
            # specify line frequency if not present as required by BIDS
            raw.info["line_freq"] = 50
        if raw.info.get("subject_info", None) is None:
            # specify subject info as required by BIDS
            raw.info["subject_info"] = {
                "his_id": subject_moabb_to_bids(self.subject),
            }
        if raw.info.get("device_info", None) is None:
            # specify device info as required by BIDS
            raw.info["device_info"] = {"type": "eeg"}
        raw.set_meas_date(datetime_now)

        # Otherwise, the montage would still have the stim channel
        # which is dropped by mne_bids.write_raw_bids:
        picks = mne.pick_types(info=raw.info, eeg=True, stim=False)
        raw.pick(picks)

        # By using the same anonymization `daysback` number we can
        # preserve the longitudinal structure of multiple sessions for a
        # single subject and the relation between subjects. Be sure to
        # change or delete this number before putting code online, you
        # wouldn't want to inadvertently de-anonymize your data.
        #
        # Note that we do not need to pass any events, as the dataset
        # is already equipped with annotations, which will be converted to
        # BIDS events automatically.
        mne_bids.write_raw_bids(
            raw,
            bids_path,
            format=self._format,
            allow_preload=True,
            montage=raw.get_montage(),
            overwrite=False,
            verbose=self.verbose,
        )


class BIDSInterfaceEpochs(BIDSInterfaceBase):
    """This interface is used to cache mne-epochs to disk.

    Pseudo-BIDS format is used to store the data.


    In this case, the ``run`` object (see the ``save()`` method)
    is expected to be an ``mne.Epochs`` instance.
    """

    @property
    def _extension(self):
        return ".fif"

    @property
    def _check(self):
        return False

    @property
    def _suffix(self):
        return "epo"

    def _load_file(self, bids_path, preload):
        epochs = mne.read_epochs(bids_path.fpath, preload=preload, verbose=self.verbose)
        return epochs

    def _write_file(self, bids_path, epochs):
        epochs.save(bids_path.fpath, overwrite=False, verbose=self.verbose)


class BIDSInterfaceNumpyArray(BIDSInterfaceBase):
    """This interface is used to cache numpy arrays to disk.

    MOABB Pseudo-BIDS format is used to store the data.

    In this case, the ``run`` object (see the ``save()`` method)
    is expected to be an ``OrderedDict`` with keys ``"X"`` and
    ``"events"``. Both values are expected to be ``numpy.ndarray``.
    """

    @property
    def _extension(self):
        return ".npy"

    @property
    def _check(self):
        return False

    @property
    def _suffix(self):
        return "array"

    def _load_file(self, bids_path, preload):
        if preload:
            raise ValueError("preload must be False for numpy arrays")
        events_fname = mne_bids.write._find_matching_sidecar(
            bids_path,
            suffix="events",
            extension=".eve",  # mne convention
            on_error="raise",
        )
        log.debug("Reading %s", bids_path.fpath)
        X = np_load(bids_path.fpath)
        events = mne.read_events(events_fname, verbose=self.verbose)
        return OrderedDict([("X", X), ("events", events)])

    def _write_file(self, bids_path, obj):
        events_path = bids_path.copy().update(
            suffix="events",
            extension=".eve",
        )
        log.debug("Writing %s", bids_path.fpath)
        np_save(bids_path.fpath, obj["X"])
        log.debug("Wrote %s", bids_path.fpath)
        mne.write_events(
            filename=events_path.fpath,
            events=obj["events"],
            overwrite=False,
            verbose=self.verbose,
        )


class StepType(Enum):
    """Enum corresponding to the type of data returned
    by a pipeline step."""

    RAW = "raw"
    EPOCHS = "epochs"
    ARRAY = "array"


_interface_map: Dict[StepType, Type[BIDSInterfaceBase]] = {
    StepType.RAW: BIDSInterfaceRawEDF,
    StepType.EPOCHS: BIDSInterfaceEpochs,
    StepType.ARRAY: BIDSInterfaceNumpyArray,
}


@dataclass
class _BIDSInterfaceRawEDFNoDesc(BIDSInterfaceRawEDF):
    """BIDSInterfaceRawEDF variant that saves without a description hash.

    Used internally by :meth:`~moabb.datasets.base.BaseDataset.convert_to_bids` to produce BIDS files
    whose names do not contain a ``desc-<hash>`` entity.
    """

    _dataset_type: str = "raw"
    _format: str = "EDF"

    @property
    def desc(self):
        return None

    def _write_lock_file(self, session, lock_data):
        """Do not write a lock file for public BIDS conversion."""

    def erase(self):
        """Remove the subject's BIDS directory entirely."""
        subject_dir = self.root / f"sub-{subject_moabb_to_bids(self.subject)}"
        if subject_dir.exists():
            log.info("Starting erasing BIDS data of %s...", repr(self))
            shutil.rmtree(subject_dir)
            log.info("Finished erasing BIDS data of %s.", repr(self))
