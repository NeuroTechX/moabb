import json
import logging
from abc import ABCMeta, abstractmethod

# import datetime
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pandas as pd

import moabb
from moabb.analysis.results import get_digest
from moabb.datasets import download as dl


log = logging.getLogger(__name__)


class BaseParadigm(metaclass=ABCMeta):
    """Base Paradigm."""

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def scoring(self):
        """Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        """
        pass

    @property
    @abstractmethod
    def datasets(self):
        """Property that define the list of compatible datasets"""
        pass

    @abstractmethod
    def is_valid(self, dataset):
        """Verify the dataset is compatible with the paradigm.

        This method is called to verify dataset is compatible with the
        paradigm.

        This method should raise an error if the dataset is not compatible
        with the paradigm. This is for example the case if the
        dataset is an ERP dataset for motor imagery paradigm, or if the
        dataset does not contain any of the required events.

        Parameters
        ----------
        dataset : dataset instance
            The dataset to verify.
        """
        pass

    def prepare_process(self, dataset):
        """Prepare processing of raw files

                This function allows to set parameter of the paradigm class prior to
                the preprocessing (process_raw). Does nothing by default and could be
                overloaded if needed.

                Parameters
                ----------

                dataset : dataset instance
                    The dataset corresponding to the raw file. mainly use to access
                    dataset specific i
        nformation.
        """
        if dataset is not None:
            pass

    @staticmethod
    def _find_events(raw, event_id):
        # find the events, first check stim_channels then annotations
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        return events

    @property
    def cached_processing_params(self):
        return [
            ("filter", self.filters),
        ]

    def pre_cache_process_raw(self, raw, dataset):
        """
        Part of the processing that is done before caching the data on disk.
        See pricess_raw for the rest of the processing.

        Parameters
        ----------
        raw: mne.Raw instance
            the raw EEG data.
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.

        Returns
        -------
        raw: mne.Raw instance
        """
        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter data
            raw_f = raw.copy().filter(fmin, fmax, method="iir", verbose=False)
            X.append(raw_f)
        return mne.concatenate_raws(X)

    def process_raw(  # noqa: C901
        self, raw, dataset, return_epochs=False, return_raws=False
    ):
        """
        Process one raw data file.

        This function apply the preprocessing and eventual epoching on the
        individual run, and return the data, labels and a dataframe with
        metadata.

        metadata is a dataframe with as many row as the length of the data
        and labels.

        Parameters
        ----------
        raw: mne.Raw instance
            the raw EEG data.
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs
        return_raws: boolean
            To return raw files and events, to ensure compatibility with braindecode.
            Mutually exclusive with return_epochs

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
            if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata

        """

        if return_epochs and return_raws:
            message = "Select only return_epochs or return_raws, not both"
            raise ValueError(message)

        # get events id
        event_id = self.used_events(dataset)

        try:
            events = self._find_events(raw, event_id)
        except ValueError:
            log.warning(f"No matching annotations in {raw.filenames}")
            return

        # picks channels
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                raw.info["ch_names"], include=self.channels, ordered=True
            )

        # pick events, based on event_id
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        if return_raws:
            raw = raw.pick(picks)
        else:
            # get interval
            tmin = self.tmin + dataset.interval[0]
            if self.tmax is None:
                tmax = dataset.interval[1]
            else:
                tmax = self.tmax + dataset.interval[0]

            # epoch data
            baseline = self.baseline
            if baseline is not None:
                baseline = (
                    self.baseline[0] + dataset.interval[0],
                    self.baseline[1] + dataset.interval[0],
                )
                bmin = baseline[0] if baseline[0] < tmin else tmin
                bmax = baseline[1] if baseline[1] > tmax else tmax
            else:
                bmin = tmin
                bmax = tmax
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=bmin,
                tmax=bmax,
                proj=False,
                baseline=baseline,
                preload=True,
                verbose=False,
                picks=picks,
                event_repeated="drop",
                on_missing="ignore",
            )
            if bmin < tmin or bmax > tmax:
                epochs.crop(tmin=tmin, tmax=tmax)
            if self.resample is not None:
                epochs = epochs.resample(self.resample)

            # overwrite events in case epochs have been dropped:
            # (assuming all filters produce the same number of epochs...)
            events = epochs.events

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in events[:, -1]])

        if return_epochs:
            X = epochs
        elif return_raws:
            X = raw
        else:
            # rescale to work with uV
            X = dataset.unit_factor * epochs.get_data()
            if len(self.filters) > 1:
                # if more than one band, return a 4D array
                X = X.reshape(
                    len(self.filters), len(events), X.shape[1], X.shape[2]
                ).transpose((1, 2, 3, 0))

        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata

    @staticmethod
    def _subject_moabb_to_bids(subject):
        return str(subject)

    @staticmethod
    def _subject_bids_to_moabb(subject):
        return int(subject)

    @staticmethod
    def _session_moabb_to_bids(session):
        return session.removeprefix("session_")

    @staticmethod
    def _session_bids_to_moabb(session):
        return "session_" + session

    @staticmethod
    def _run_moabb_to_bids(run):
        # Note: the runs are expected to be indexes in the BIDS standard.
        #       This is not always the case in MOABB.
        # See: https://bids-specification.readthedocs.io/en/stable/glossary.html#run-entities
        return run.removeprefix("run_")

    @staticmethod
    def _run_bids_to_moabb(run):
        return "run_" + run

    @property
    def bids_desc(self):
        return get_digest(self.cached_processing_params)

    @staticmethod
    def _get_cache_root(dataset, path):
        code = dataset.code + "-BIDS"
        mne_path = Path(dl.get_dataset_path(code, path))
        cache_dir = f"MNE-{code.lower()}-cache"
        cache_path = mne_path / cache_dir
        if not cache_path.is_dir():
            cache_path.mkdir(parents=True)
        return cache_path

    def _get_bids_lock_path(self, dataset, subject, path):
        # this file was saved last to ensure that the subject's data was completely saved
        # this is not an official bids file
        return mne_bids.BIDSPath(
            root=self._get_cache_root(dataset, path),
            subject=self._subject_moabb_to_bids(subject),
            description=self.bids_desc,
            extension=".json",
            suffix="lockfile",  # necessary for unofficial files
            check=False,
        )

    def _delete_cache(self, dataset, subject, path):
        # TODO: this function does not update the scans.tsv files
        #       should be fixed by https://github.com/mne-tools/mne-bids/pull/547
        log.info(
            f"Starting erasing cache of dataset {dataset.code}, subject {subject}..."
        )
        paths = mne_bids.find_matching_paths(
            root=self._get_cache_root(dataset, path),
            subjects=self._subject_moabb_to_bids(subject),
            descriptions=self.bids_desc,
        )
        for p in paths:
            log.debug(f"Erasing {p}")
            p.fpath.unlink()  # remove file
        log.info(f"Finished erasing cache of dataset {dataset.code}, subject {subject}.")

    def _load_cache(self, dataset, subject, path):
        log.info(
            f"Attempting to retrieve cache of dataset {dataset.code}, subject {subject}..."
        )
        lock_file = self._get_bids_lock_path(dataset, subject, path)
        lock_file.mkdir(exist_ok=True)
        if not lock_file.fpath.exists():
            log.info(f"No cache found at {str(lock_file.directory)}.")
            return None
        paths = mne_bids.find_matching_paths(
            root=self._get_cache_root(dataset, path),
            subjects=self._subject_moabb_to_bids(subject),
            descriptions=self.bids_desc,
            extensions=".edf",
        )
        sessions_data = {}
        for p in paths:
            session = sessions_data.setdefault(self._session_bids_to_moabb(p.session), {})
            log.debug(f"Reading {p.fpath}")
            run = mne_bids.read_raw_bids(p)
            session[self._run_bids_to_moabb(p.run)] = run
        log.info(f"Finished reading cache of dataset {dataset.code}, subject {subject}.")
        return sessions_data

    def _save_cache(self, dataset, subject, sessions_data, path):
        log.info(f"Starting caching dataset {dataset.code}, subject {subject} to disk...")
        mne_bids.make_dataset_description(
            path=str(self._get_cache_root(dataset, path)),
            name=dataset.code,
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
                    DOI=dataset.doi,
                )
            ],
            verbose=False,
            overwrite=False,
        )

        # datetime_now = datetime.datetime.now(tz=datetime.timezone.utc)
        raws = []
        for runs in sessions_data.values():
            for raw in runs.values():
                raws.append(raw)
                if raw.info.get("line_freq", None) is None:
                    # specify line frequency if not present as required by BIDS
                    raw.info["line_freq"] = 50
                if raw.info.get("subject_info", None) is None:
                    # specify subject info as required by BIDS
                    raw.info["subject_info"] = {
                        "his_id": subject,
                    }
                if raw.info.get("device_info", None) is None:
                    # specify device info as required by BIDS
                    raw.info["device_info"] = {"type": "eeg"}
                if raw.info.get("meas_date", None) is None:
                    raw.set_meas_date(None)
        # daysback_min, daysback_max = mne_bids.get_anonymization_daysback(raws)
        for session, runs in sessions_data.items():
            for run, raw in runs.items():
                bids_path = mne_bids.BIDSPath(
                    root=self._get_cache_root(dataset, path),
                    subject=self._subject_moabb_to_bids(subject),
                    session=self._session_moabb_to_bids(session),
                    task=dataset.paradigm,
                    run=self._run_moabb_to_bids(run),
                    description=self.bids_desc,
                    datatype="eeg",
                )

                events = self._find_events(raw, dataset.event_id)
                # By using the same anonymization `daysback` number we can
                # preserve the longitudinal structure of multiple sessions for a
                # single subject and the relation between subjects. Be sure to
                # change or delete this number before putting code online, you
                # wouldn't want to inadvertently de-anonymize your data.
                #
                # Note that we do not need to pass any events, as the dataset is already
                # equipped with annotations, which will be converted to BIDS events
                # automatically.
                log.debug(f"Writing {bids_path}")
                bids_path.mkdir(exist_ok=True)
                mne_bids.write_raw_bids(
                    raw,
                    bids_path,
                    # anonymize=dict(daysback=daysback_min + 2117),
                    events=events,
                    event_id=dataset.event_id,
                    format="EDF",
                    allow_preload=True,
                    montage=raw.get_montage(),
                    overwrite=False,  # files should be deleted by _delete_cache in case overwrite_cache is True
                    verbose="ERROR",
                )
        lock_file = self._get_bids_lock_path(dataset, subject, path)
        log.debug(f"Writing {lock_file}")
        lock_file.mkdir(exist_ok=True)
        with lock_file.fpath.open("w") as f:
            json.dump(self.cached_processing_params, f)
        log.info(f"Finished caching dataset {dataset.code}, subject {subject} to disk.")

    def _get_single_subject_data(
        self, dataset, subject, save_cache, use_cache, overwrite_cache, path
    ):
        """
        Either load the data of a single subject from disk cache or from the dataset object,
        then eventually saves or overwrites the cache version depending on the parameters.
        """
        if overwrite_cache:
            self._delete_cache(dataset, subject, path)
            use_cache = False  # can't load if it was just erased
        sessions_data = None
        if use_cache:
            sessions_data = self._load_cache(dataset, subject, path)
        if sessions_data is not None:
            save_cache = False  # no need to save if we just loaded it
        else:
            sessions_data = dataset._get_single_subject_data(subject)
            sessions_data = {
                session: {
                    run: self.pre_cache_process_raw(raw, dataset)
                    for run, raw in runs.items()
                }
                for session, runs in sessions_data.items()
            }
        if save_cache:
            try:
                self._save_cache(dataset, subject, sessions_data, path)
            except Exception as ex:
                # ex_type, ex_value, ex_traceback = sys.exc_info()
                log.warning(
                    f"Failed to save dataset {dataset.code}, subject {subject} to BIDS format:\n{ex}"
                )
                # self._delete_cache(dataset, subject, path)  # remove partial cache
        return sessions_data

    def get_data(
        self,
        dataset,
        subjects=None,
        return_epochs=False,
        return_raws=False,
        save_cache=True,
        use_cache=True,
        overwrite_cache=True,
        path=None,
    ):
        """
        Return the data for a list of subject.

        return the data, labels and a dataframe with metadata. the dataframe
        will contain at least the following columns

        - subject : the subject indice
        - session : the session indice
        - run : the run indice

        parameters
        ----------
        dataset:
            A dataset instance.
        subjects: List of int
            List of subject number
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs
        return_raws: boolean
            To return raw files and events, to ensure compatibility with braindecode.
            Mutually exclusive with return_epochs
        save_cache: boolean
            This flag specifies whether to save the processed mne.io.Raw to disk
        use_cache: boolean
            This flag specifies whether to use the processed mne.io.Raw from disk
            in case they exist
        overwrite_cache: boolean
            This flag specifies whether to overwrite the processed mne.io.Raw on disk
            in case they exist
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
            if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata.
        """

        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)

        if return_epochs and return_raws:
            message = "Select only return_epochs or return_raws, not both"
            raise ValueError(message)

        if subjects is None:
            subjects = dataset.subject_list
        self.prepare_process(dataset)

        X = [] if (return_epochs or return_raws) else np.array([])
        labels = []
        metadata = []
        for subject in subjects:
            sessions = self._get_single_subject_data(
                dataset, subject, save_cache, use_cache, overwrite_cache, path
            )
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    proc = self.process_raw(raw, dataset, return_epochs, return_raws)

                    if proc is None:
                        # this mean the run did not contain any selected event
                        # go to next
                        continue

                    x, lbs, met = proc
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run
                    metadata.append(met)

                    # grow X and labels in a memory efficient way. can be slow
                    if return_epochs:
                        x.metadata = (
                            met.copy()
                            if len(self.filters) == 1
                            else pd.concat(
                                [met.copy()] * len(self.filters), ignore_index=True
                            )
                        )
                        X.append(x)
                    elif return_raws:
                        X.append(x)
                    else:
                        X = np.append(X, x, axis=0) if len(X) else x
                    labels = np.append(labels, lbs, axis=0)

        metadata = pd.concat(metadata, ignore_index=True)
        if return_epochs:
            X = mne.concatenate_epochs(X)
        return X, labels, metadata
