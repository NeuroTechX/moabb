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
        return (
            f"{self.dataset.code!r} sub-{self.subject} "
            f"datatype-{self._datatype} desc-{self.desc:.7}"
        )

    @property
    def root(self):
        """Return the root path of the BIDS dataset."""
        code = self.dataset.code
        mne_path = Path(dl.get_dataset_path(code, self.path))
        cache_dir = f"MNE-BIDS-{camel_to_kebab_case(code)}"
        cache_path = mne_path / cache_dir
        return cache_path

    @property
    def lock_file(self):
        """Return the lock file path.

        this file was saved last to ensure that the subject's data was
        completely saved this is not an official bids file
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
        path = mne_bids.BIDSPath(
            root=self.root,
            subject=subject_moabb_to_bids(self.subject),
            description=self.desc,
            check=False,
        )
        path.rm(safe_remove=False)
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
        self.lock_file.mkdir(exist_ok=True)
        if not self.lock_file.fpath.exists():
            log.info("No cache found at %s.", str(self.lock_file.directory))
            return None
        paths = mne_bids.find_matching_paths(
            root=self.root,
            subjects=subject_moabb_to_bids(self.subject),
            descriptions=self.desc,
            extensions=self._extension,
            check=self._check,
            datatypes=self._datatype,
            suffixes=self._datatype,
        )
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
        mne_bids.make_dataset_description(
            path=str(self.root),
            name=self.dataset.code,
            dataset_type="derivative",
            generated_by=[
                dict(
                    CodeURL="https://github.com/NeuroTechX/moabb",
                    Name="moabb",
                    Description="Mother of All BCI Benchmarks",
                    Version=moabb.__version__,
                )
            ],
            source_datasets=[
                dict(
                    DOI=self.dataset.doi,
                )
            ],
            overwrite=False,
            verbose=self.verbose,
        )

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
                    datatype=self._datatype,
                    suffix=self._datatype,
                    check=self._check,
                )

                bids_path.mkdir(exist_ok=True)
                self._write_file(bids_path, obj)
        log.debug("Writing", self.lock_file)
        self.lock_file.mkdir(exist_ok=True)
        with self.lock_file.fpath.open("w") as file:
            dic = dict(processing_params=str(self.processing_params))
            json.dump(dic, file)
        log.info("Finished caching %s to disk.", repr(self))

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
    def _datatype(self):
        pass


class BIDSInterfaceRawEDF(BIDSInterfaceBase):
    """BIDS Interface for Raw EDF files. Selected .edf type only.

    In this case, the ``run`` object (see the ``save()`` method)
    is expected to be an ``mne.io.BaseRaw`` instance."""

    @property
    def _extension(self):
        return ".edf"

    @property
    def _check(self):
        return True

    @property
    def _datatype(self):
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
            format="EDF",
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
    def _datatype(self):
        # because of mne conventions, we need the suffix to be "epo"
        # because of mne_bids conventions, we need datatype and suffix to match
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
    def _datatype(self):
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
