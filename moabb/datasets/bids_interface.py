import abc
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mne
import mne_bids
from numpy import load as np_load
from numpy import save as np_save

import moabb
from moabb.analysis.results import get_digest
from moabb.datasets import download as dl
from moabb.paradigms.utils import _find_events


if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)


def subject_moabb_to_bids(subject):
    return str(subject)


def subject_bids_to_moabb(subject):
    return int(subject)


def session_moabb_to_bids(session):
    return session.replace("session_", "")


def session_bids_to_moabb(session):
    return "session_" + session


def run_moabb_to_bids(run):
    # Note: the runs are expected to be indexes in the BIDS standard.
    #       This is not always the case in MOABB.
    # See: https://bids-specification.readthedocs.io/en/stable/glossary.html#run-entities
    return run.replace("run_", "")


def run_bids_to_moabb(run):
    return "run_" + run


# @total_ordering
@dataclass
class BIDSInterfaceBase(abc.ABC):
    dataset: "BaseDataset"
    subject: int
    path: str = None
    process_pipeline: "Pipeline" = None
    verbose: str = None

    # @classmethod
    # def from_lockfile(cls, lock_file, other_interface):
    #     with lock_file.fpath.open("r") as f:
    #         processing_params = json.load(f)
    #     return cls(
    #         dataset=other_interface.dataset,  # TODO set correct kwargs of dataset
    #         subject=other_interface.subject,
    #         path=other_interface.path,
    #         fmin=processing_params["fmin"],
    #         fmax=processing_params["fmax"],
    #     )

    # def _base_comp(self, other):
    #     return self.dataset == other.dataset and self.subject == other.subject

    # def __eq__(self, other):
    #     return (
    #             self._base_comp(other) and self.fmin == other.fmin and self.fmax == other.fmax
    #     )

    # def __le__(self, other):
    #     return (
    #             self._base_comp(other)
    #             and (
    #                     other.fmin is None or (self.fmin is not None and self.fmin >= other.fmin)
    #             )
    #             and (
    #                     other.fmax is None or (self.fmax is not None and self.fmax <= other.fmax)
    #             )
    #     )

    @property
    def processing_params(self):
        # TODO: add dataset kwargs
        return self.process_pipeline

    @property
    def desc(self):
        return get_digest(self.processing_params)

    def __repr__(self):
        return f"dataset {self.dataset.code}, subject {self.subject}, processing pipeline {self.desc}"

    @property
    def root(self):
        code = self.dataset.code + "-BIDS"
        mne_path = Path(dl.get_dataset_path(code, self.path))
        cache_dir = f"MNE-{code.lower()}-cache"
        cache_path = mne_path / cache_dir
        # if not cache_path.is_dir():
        #     cache_path.mkdir(parents=True)
        return cache_path

    @property
    def lock_file(self):
        # this file was saved last to ensure that the subject's data was completely saved
        # this is not an official bids file
        return mne_bids.BIDSPath(
            root=self.root,
            subject=subject_moabb_to_bids(self.subject),
            description=self.desc,
            extension=".json",
            suffix="lockfile",  # necessary for unofficial files
            check=False,
        )

    # def find_compatible_interface(self):
    #     lock_files = mne_bids.find_matching_paths(
    #         root=self.root,
    #         subjects=subject_moabb_to_bids(self.subject),
    #         # descriptions=self.desc,
    #         extensions=".json",
    #         suffixes="lockfile",  # necessary for unofficial files
    #         check=False,
    #     )
    #     interfaces = [
    #         self.__class__.from_lockfile(lock_file, self) for lock_file in lock_files
    #     ]
    #     interfaces = [interface for interface in interfaces if interface >= self]
    #     if len(interfaces) == 0:
    #         return None  # No compatible interface found
    #     return interfaces[0]  # We just return one at random

    def erase(self):
        log.info(f"Starting erasing cache of {repr(self)}...")
        path = mne_bids.BIDSPath(
            root=self.root,
            subject=subject_moabb_to_bids(self.subject),
            description=self.desc,
            check=False,
        )
        path.rm(safe_remove=False)
        log.info(f"Finished erasing cache of {repr(self)}.")

    def load(self, preload=False):
        log.info(f"Attempting to retrieve cache of {repr(self)}...")
        self.lock_file.mkdir(exist_ok=True)
        if not self.lock_file.fpath.exists():
            log.info(f"No cache found at {str(self.lock_file.directory)}.")
            return None
        paths = mne_bids.find_matching_paths(
            root=self.root,
            subjects=subject_moabb_to_bids(self.subject),
            descriptions=self.desc,
            extensions=self._extension,
            check=self._check,
        )
        sessions_data = {}
        for p in paths:
            session = sessions_data.setdefault(session_bids_to_moabb(p.session), {})
            # log.debug(f"Reading {p.fpath}")
            run = self._load_file(p, preload=preload)
            session[run_bids_to_moabb(p.run)] = run
        log.info(f"Finished reading cache of {repr(self)}.")
        return sessions_data

    def save(self, sessions_data):
        log.info(
            f"Starting caching dataset {self.dataset.code}, subject {self.subject} to disk..."
        )
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
                bids_path = mne_bids.BIDSPath(
                    root=self.root,
                    subject=subject_moabb_to_bids(self.subject),
                    session=session_moabb_to_bids(session),
                    task=self.dataset.paradigm,
                    run=run_moabb_to_bids(run),
                    description=self.desc,
                    datatype="eeg",
                    extension=self._extension,
                    check=self._check,
                )

                bids_path.mkdir(exist_ok=True)
                self._write_file(bids_path, obj)
        # log.debug(f"Writing {self.lock_file}")
        self.lock_file.mkdir(exist_ok=True)
        with self.lock_file.fpath.open("w") as f:
            d = dict(processing_params=str(self.processing_params))
            json.dump(d, f)
        log.info(f"Finished caching {repr(self)} to disk.")

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


class BIDSInterfaceRawEDF(BIDSInterfaceBase):
    @property
    def _extension(self):
        return ".edf"

    @property
    def _check(self):
        return True

    def _load_file(self, bids_path, preload):
        raw = mne_bids.read_raw_bids(
            bids_path, extra_params=dict(preload=preload), verbose=self.verbose
        )
        return raw

    def _write_file(self, bids_path, raw):
        datetime_now = datetime.datetime.now(tz=datetime.timezone.utc)
        if raw.info.get("line_freq", None) is None:
            # specify line frequency if not present as required by BIDS
            raw.info["line_freq"] = 50
        if raw.info.get("subject_info", None) is None:
            # specify subject info as required by BIDS
            raw.info["subject_info"] = {
                "his_id": self.subject,
            }
        if raw.info.get("device_info", None) is None:
            # specify device info as required by BIDS
            raw.info["device_info"] = {"type": "eeg"}
        if raw.info.get("meas_date", None) is None:
            raw.set_meas_date(datetime_now)

        events = _find_events(raw, self.dataset.event_id)
        picks = mne.pick_types(info=raw.info, eeg=True, stim=False)
        raw.pick(picks)
        # By using the same anonymization `daysback` number we can
        # preserve the longitudinal structure of multiple sessions for a
        # single subject and the relation between subjects. Be sure to
        # change or delete this number before putting code online, you
        # wouldn't want to inadvertently de-anonymize your data.
        #
        # Note that we do not need to pass any events, as the dataset is already
        # equipped with annotations, which will be converted to BIDS events
        # automatically.
        # log.debug(f"Writing {bids_path}")
        mne_bids.write_raw_bids(
            raw,
            bids_path,
            # anonymize=dict(daysback=daysback_min + 2117),
            events=events,
            event_id=self.dataset.event_id,
            format="EDF",
            allow_preload=True,
            montage=raw.get_montage(),
            overwrite=False,  # files should be deleted by _delete_cache in case overwrite_cache is True
            verbose=self.verbose,
        )


class BIDSInterfaceEpochs(BIDSInterfaceBase):
    @property
    def _extension(self):
        return "-epo.fif"

    @property
    def _check(self):
        return False

    def _load_file(self, bids_path, preload):
        epochs = mne.read_epochs(bids_path.fpath, preload=preload, verbose=self.verbose)
        return epochs

    def _write_file(self, bids_path, epochs):
        epochs.save(bids_path.fpath, overwrite=False, verbose=self.verbose)


class BIDSInterfaceNumpyArray(BIDSInterfaceBase):
    @property
    def _extension(self):
        return ".npy"

    @property
    def _check(self):
        return False

    def _load_file(self, bids_path, preload):
        if preload:
            raise ValueError("preload must be False for numpy arrays")
        x = np_load(bids_path.fpath)
        return x

    def _write_file(self, bids_path, x):
        np_save(bids_path.fpath, x)
