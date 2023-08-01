import abc
import logging
from operator import methodcaller
from typing import List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from moabb.datasets.preprocessing import (
    EpochsToEvents,
    EventsToLabels,
    ForkPipelines,
    RawToEpochs,
    RawToEvents,
    get_crop_pipeline,
    get_filter_pipeline,
    get_resample_pipeline,
)


log = logging.getLogger(__name__)


class BaseProcessing(metaclass=abc.ABCMeta):
    """Base Processing.

    Please use one of the child classes


    Parameters
    ----------

    filters: list of list (defaults [[7, 35]])
        bank of bandpass filter to apply.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline
            correction. If None, do not apply baseline correction.
            If a tuple (a, b), the interval is between a and b (in seconds),
            including the endpoints.
            Correction is applied by computing the mean of the baseline period
            and subtracting it from the data (see mne.Epochs)

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided."""

    def __init__(
        self,
        filters: List[Tuple[float, float]],
        tmin: float = 0.0,
        tmax: Optional[float] = None,
        baseline: Optional[Tuple[float, float]] = None,
        channels: Optional[List[str]] = None,
        resample: Optional[float] = None,
    ):
        if tmax is not None:
            if tmin >= tmax:
                raise (ValueError("tmax must be greater than tmin"))
        self.filters = filters
        self.channels = channels
        self.baseline = baseline
        self.resample = resample
        self.tmin = tmin
        self.tmax = tmax

    @property
    @abc.abstractmethod
    def datasets(self):
        """Property that define the list of compatible datasets."""
        pass

    @abc.abstractmethod
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
        """Prepare processing of raw files.

        This function allows to set parameter of the paradigm class prior to
        the preprocessing (process_raw). Does nothing by default and could be
        overloaded if needed.

        Parameters
        ----------
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.
        """
        if dataset is not None:
            pass


    @abc.abstractmethod
    def used_events(self, dataset):
        pass

    def get_data(  # noqa: C901
        self,
        dataset,
        subjects=None,
        return_epochs=False,
        return_raws=False,
        processing_pipeline=None,
        cache_config=None,
    ):
        """
        Return the data for a list of subject.

        return the data, labels and a dataframe with metadata. the dataframe
        will contain at least the following columns

        - subject : the subject indice
        - session : the session indice
        - run : the run indice

        Parameters
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
        cache_config: dict | CacheConfig
            Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.

        Eeturns
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
        raw_pipelines = self._get_raw_pipelines()
        epochs_pipeline = self._get_epochs_pipeline(return_epochs, return_raws, dataset)
        array_pipeline = self._get_array_pipeline(
            return_epochs, return_raws, dataset, processing_pipeline
        )
        if return_epochs:
            labels_pipeline = make_pipeline(
                EpochsToEvents(),
                EventsToLabels(event_id=self.used_events(dataset)),
            )
        elif return_raws:
            labels_pipeline = make_pipeline(
                self._get_events_pipeline(dataset),
                EventsToLabels(event_id=self.used_events(dataset)),
            )
        else:  # return array
            labels_pipeline = EventsToLabels(event_id=self.used_events(dataset))

        if array_pipeline is not None:
            events_pipeline = (
                self._get_events_pipeline(dataset) if return_raws else EpochsToEvents()
            )
        else:
            events_pipeline = None

        data = [
            dataset.get_data(
                subjects=subjects,
                cache_config=cache_config,
                raw_pipeline=raw_pipeline,
                epochs_pipeline=epochs_pipeline,
                array_pipeline=array_pipeline,
                events_pipeline=events_pipeline,
            )
            for raw_pipeline in raw_pipelines
        ]

        X = []
        labels = []
        metadata = []
        for subject, sessions in data[0].items():
            for session, runs in sessions.items():
                for run in runs.keys():
                    proc = [data_i[subject][session][run] for data_i in data]
                    if any(obj is None for obj in proc):
                        # this mean the run did not contain any selected event
                        # go to next
                        assert all(obj is None for obj in proc)  # sanity check
                        continue

                    if return_epochs:
                        assert all(len(proc[0]) == len(p) for p in proc[1:])
                        n = len(proc[0])
                        lbs = labels_pipeline.transform(proc[0])
                        x = (
                            proc[0]
                            if len(self.filters) == 1
                            else mne.concatenate_epochs(proc)
                        )
                    elif return_raws:
                        assert all(len(proc[0]) == len(p) for p in proc[1:])
                        n = 1
                        lbs = labels_pipeline.transform(
                            proc[0]
                        )  # XXX does it make sense to return labels for raws?
                        x = proc[0] if len(self.filters) == 1 else proc
                    else:  # return array
                        assert all(
                            np.array_equal(proc[0]["X"].shape, p["X"].shape)
                            for p in proc[1:]
                        )
                        assert all(
                            np.array_equal(proc[0]["events"], p["events"])
                            for p in proc[1:]
                        )
                        n = proc[0]["X"].shape[0]
                        events = proc[0]["events"]
                        lbs = labels_pipeline.transform(events)
                        x = (
                            proc[0]["X"]
                            if len(self.filters) == 1
                            else np.array([p["X"] for p in proc]).transpose((1, 2, 3, 0))
                        )

                    met = pd.DataFrame(index=range(n))
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run
                    metadata.append(met)

                    if return_epochs:
                        x.metadata = (
                            met.copy()
                            if len(self.filters) == 1
                            else pd.concat(
                                [met.copy()] * len(self.filters), ignore_index=True
                            )
                        )
                    X.append(x)
                    labels.append(lbs)

        metadata = pd.concat(metadata, ignore_index=True)
        labels = np.concatenate(labels)
        if return_epochs:
            X = mne.concatenate_epochs(X)
        elif return_raws:
            pass
        else:
            X = np.concatenate(X, axis=0)
        return X, labels, metadata

    def _get_raw_pipelines(self):
        return [get_filter_pipeline(fmin, fmax) for fmin, fmax in self.filters]

    def _get_epochs_pipeline(self, return_epochs, return_raws, dataset):
        if return_raws:
            return None

        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

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
        steps = []
        steps.append(
            (
                "epoching",
                make_pipeline(
                    ForkPipelines(
                        [
                            ("raw", make_pipeline(None)),
                            ("events", self._get_events_pipeline(dataset)),
                        ]
                    ),
                    RawToEpochs(
                        event_id=self.used_events(dataset),
                        tmin=bmin,
                        tmax=bmax,
                        baseline=baseline,
                        channels=self.channels,
                    ),
                ),
            )
        )
        if bmin < tmin or bmax > tmax:
            steps.append(("crop", get_crop_pipeline(tmin=tmin, tmax=tmax)))
        if self.resample is not None:
            steps.append(("resample", get_resample_pipeline(self.resample)))
        if return_epochs:  # needed to concatenate epochs
            steps.append(("load_data", FunctionTransformer(methodcaller("load_data"))))
        return Pipeline(steps)

    def _get_array_pipeline(
        self, return_epochs, return_raws, dataset, processing_pipeline
    ):
        steps = []
        if not return_epochs and not return_raws:
            steps.append(("get_data", FunctionTransformer(methodcaller("get_data"))))
            steps.append(
                (
                    "scaling",
                    FunctionTransformer(methodcaller("__mul__", dataset.unit_factor)),
                )
            )
        if processing_pipeline is not None:
            steps.append(("processing_pipeline", processing_pipeline))
        if len(steps) == 0:
            return None
        return Pipeline(steps)

    @abc.abstractmethod
    def _get_events_pipeline(self, dataset):
        pass


class BaseParadigm(BaseProcessing):
    """Base class for paradigms.

    Parameters
    ----------

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.
    """

    def __init__(
        self,
        filters,
        events: Optional[List[str]] = None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
    ):
        super().__init__(
            filters=filters,
            channels=channels,
            baseline=baseline,
            resample=resample,
            tmin=tmin,
            tmax=tmax,
        )
        self.events = events

    @property
    @abc.abstractmethod
    def scoring(self):
        """Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        """
        pass

    def _get_events_pipeline(self, dataset):
        event_id = self.used_events(dataset)
        return RawToEvents(event_id=event_id)
