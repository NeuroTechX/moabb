from __future__ import annotations

import abc
import logging
from operator import methodcaller
from typing import List, Literal, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import check_scoring, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from moabb.datasets.base import BaseDataset
from moabb.datasets.bids_interface import StepType
from moabb.datasets.preprocessing import (
    EpochsToEvents,
    EventsToLabels,
    FixedPipeline,
    ForkPipelines,
    RawToEpochs,
    RawToEvents,
    SetRawAnnotations,
    get_crop_pipeline,
    get_filter_pipeline,
    get_resample_pipeline,
    make_fixed_pipeline,
)
from moabb.utils import MoabbMetaClass


log = logging.getLogger(__name__)


def _normalize_scorer(scorer):
    """Normalize scorer, converting list-style scorers to a dict.

    This handles lists of metric functions or scorer objects and converts
    them to a dict format that sklearn's check_scoring can handle.

    Parameters
    ----------
    scorer : str, callable, dict, list, or None
        The scoring specification. Can be:
        - None: returns None (use default)
        - str: returns as-is
        - callable: returns as-is
        - dict: returns as-is
        - list of str: returns as-is (sklearn handles this)
        - list of callable/scorer/tuple: converts to dict with metric names as keys.
          Each element can be:
          - a callable metric function (assumes greater_is_better=True)
          - a scorer object (e.g., make_scorer/get_scorer output), passed through
          - a tuple of (callable, greater_is_better)
          - a tuple of (callable, scorer_kwargs) where scorer_kwargs is a dict
            (e.g., needs_proba/needs_threshold; may include greater_is_better)
          - a tuple of (callable, greater_is_better, scorer_kwargs)

    Returns
    -------
    normalized_scorer : str, callable, dict, list, or None
        The normalized scorer specification.

    Raises
    ------
    ValueError
        If list is empty or contains invalid types.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score, mean_squared_error
    >>> # Simple list of metrics (all assume higher is better)
    >>> scorer = [accuracy_score, balanced_accuracy_score]
    >>> # Mix of metrics with explicit greater_is_better control
    >>> scorer = [
    ...     accuracy_score,                  # greater_is_better=True (default)
    ...     (mean_squared_error, False),     # greater_is_better=False (loss)
    ... ]
    >>> # Metrics needing probability/threshold based scoring
    >>> scorer = [
    ...     (roc_auc_score, {"needs_threshold": True}),
    ... ]
    """
    if scorer is None or isinstance(scorer, (str, dict)):
        return scorer

    if isinstance(scorer, list):
        if len(scorer) == 0:
            raise ValueError("scorer list cannot be empty")
        if all(isinstance(s, str) for s in scorer):
            # List of strings - sklearn handles this natively
            return scorer

        def _is_scorer_object(obj):
            # Detect sklearn scorer objects (e.g., make_scorer/get_scorer output)
            return callable(obj) and hasattr(obj, "_score_func") and hasattr(obj, "_sign")

        # Check if list contains valid scorer items
        def _is_valid_scorer_item(item):
            if _is_scorer_object(item):
                return True
            if callable(item):
                return True
            if isinstance(item, tuple):
                if len(item) == 2:
                    func, second = item
                    return callable(func) and (
                        isinstance(second, bool) or isinstance(second, dict)
                    )
                if len(item) == 3:
                    func, greater, kwargs = item
                    return (
                        callable(func)
                        and isinstance(greater, bool)
                        and isinstance(kwargs, dict)
                    )
            return False

        if all(_is_valid_scorer_item(s) for s in scorer):
            # Convert list of metric functions/tuples to dict
            result = {}
            seen = {}
            for i, item in enumerate(scorer):
                # Pass through scorer objects unchanged
                if _is_scorer_object(item):
                    scorer_obj = item
                    func = getattr(item, "_score_func", None)
                else:
                    scorer_kwargs = {}
                    # Extract function and greater_is_better/kwargs
                    if isinstance(item, tuple):
                        if len(item) == 2:
                            func, second = item
                            if isinstance(second, bool):
                                greater_is_better = second
                            else:
                                scorer_kwargs = dict(second)
                                greater_is_better = scorer_kwargs.pop(
                                    "greater_is_better", True
                                )
                        else:
                            func, greater_is_better, scorer_kwargs = item
                            if "greater_is_better" in scorer_kwargs:
                                raise ValueError(
                                    "greater_is_better should not be provided in "
                                    "scorer_kwargs when passed as a separate argument"
                                )
                    else:
                        func = item
                        greater_is_better = True

                    # Handle deprecated parameters for sklearn 1.4+ and 1.6+
                    if "needs_threshold" in scorer_kwargs and scorer_kwargs.pop(
                        "needs_threshold"
                    ):
                        scorer_kwargs.setdefault(
                            "response_method", ("decision_function", "predict_proba")
                        )
                    if "needs_proba" in scorer_kwargs and scorer_kwargs.pop(
                        "needs_proba"
                    ):
                        scorer_kwargs.setdefault("response_method", "predict_proba")

                    scorer_obj = make_scorer(
                        func, greater_is_better=greater_is_better, **scorer_kwargs
                    )

                # Generate unique name
                name = getattr(func, "__name__", f"scorer_{i}")
                if name == "<lambda>":
                    name = f"scorer_{i}"
                if name in seen:
                    seen[name] += 1
                    name = f"{name}_{seen[name]}"
                else:
                    seen[name] = 0

                result[name] = scorer_obj
            return result

        raise ValueError(
            "scorer list must contain all strings, all callables/scorers, "
            "or all tuples with (callable, bool), (callable, kwargs), "
            "or (callable, bool, kwargs)"
        )

    # callable passes through
    return scorer


class BaseProcessing(metaclass=MoabbMetaClass):
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
        If not None, resample the eeg data with the sampling rate provided.
    """

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
        self.interpolate_missing_channels = False

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

    def make_process_pipelines(
        self, dataset, return_epochs=False, return_raws=False, postprocess_pipeline=None
    ):
        """Create pre-processing pipelines for the data.

        Return the pre-processing pipelines corresponding to this paradigm (one per frequency band).

        Parameters
        ----------
        dataset : BaseDataset
            The dataset instance.
        return_epochs : bool, default is False
            Specify if needed to return epochs instead of ndarray.
        return_raws : bool, default is False
            Specify if needed to return raws instead of ndarray.
        postprocess_pipeline : Pipeline | None, default is None
            Optional pipeline to apply to the data after the preprocessing.
            This pipeline will either receive :class:`mne.io.BaseRaw`, :class:`mne.Epochs`
            or :func:`np.ndarray` as input, depending on the values of ``return_epochs``
            and ``return_raws``.
            This pipeline must return an ``np.ndarray``.
            This pipeline must be "fixed" because it will not be trained,
            i.e. no call to ``fit`` will be made.
        """
        if return_epochs and return_raws:
            message = "Select only return_epochs or return_raws, not both"
            raise ValueError(message)

        self.prepare_process(dataset)

        raw_pipelines = self._get_raw_pipelines()
        epochs_pipeline = self._get_epochs_pipeline(return_epochs, return_raws, dataset)
        array_pipeline = self._get_array_pipeline(
            return_epochs, return_raws, dataset, postprocess_pipeline
        )

        if array_pipeline is not None:
            events_pipeline = (
                self._get_events_pipeline(dataset) if return_raws else EpochsToEvents()
            )
        else:
            events_pipeline = None

        if events_pipeline is None and array_pipeline is not None:
            log.warning(
                f"event_id not specified, using all the dataset's "
                f"events to generate labels: {dataset.event_id}"
            )
            events_pipeline = (
                RawToEvents(dataset.event_id, interval=dataset.interval)
                if epochs_pipeline is None
                else EpochsToEvents()
            )

        process_pipelines = []
        for raw_pipeline in raw_pipelines:
            steps = []
            steps.append(
                (
                    StepType.RAW,
                    SetRawAnnotations(
                        dataset.event_id,
                        interval=dataset.interval,
                    ),
                )
            )
            if raw_pipeline is not None:
                steps.append((StepType.RAW, raw_pipeline))
            if epochs_pipeline is not None:
                steps.append((StepType.EPOCHS, epochs_pipeline))
            if array_pipeline is not None:
                array_events_pipeline = ForkPipelines(
                    [
                        ("X", array_pipeline),
                        ("events", events_pipeline),
                    ]
                )
                steps.append((StepType.ARRAY, array_events_pipeline))
            process_pipelines.append(FixedPipeline(steps))
        return process_pipelines

    def make_labels_pipeline(self, dataset, return_epochs=False, return_raws=False):
        """Returns the pipeline that extracts the labels from the
        output of the postprocess_pipeline.
        Refer to the arguments of :func:`get_data` for more information."""
        if return_epochs:
            labels_pipeline = make_fixed_pipeline(
                EpochsToEvents(),
                EventsToLabels(event_id=self.used_events(dataset)),
            )
        elif return_raws:
            labels_pipeline = make_fixed_pipeline(
                self._get_events_pipeline(dataset),
                EventsToLabels(event_id=self.used_events(dataset)),
            )
        else:  # return array
            labels_pipeline = EventsToLabels(event_id=self.used_events(dataset))
        return labels_pipeline

    def get_data(  # noqa: C901
        self,
        dataset,
        subjects=None,
        return_epochs=False,
        return_raws=False,
        cache_config=None,
        postprocess_pipeline=None,
        process_pipelines=None,
        additional_metadata: Literal["all"] | list[str] = None,
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
        postprocess_pipeline: Pipeline | None
            Optional pipeline to apply to the data after the preprocessing.
            This pipeline will either receive :class:`mne.io.BaseRaw`, :class:`mne.Epochs`
            or :func:`np.ndarray` as input, depending on the values of ``return_epochs``
            and ``return_raws``.
            This pipeline must return an ``np.ndarray``.
            This pipeline must be "fixed" because it will not be trained,
            i.e. no call to ``fit`` will be made.
        process_pipelines: Pipeline | None
            Optional pipeline to apply to the data after the preprocessing.
            You must set the ``return_epochs`` and ``return_raws` parameters
            accordingly, i.e., if your custom pipeline returns raw objects,
            you must also set ``return_raws=True``, otherwise you will get unexpected results.
            Only use it if you know what you are doing.
        additional_metadata: Literal["all"] | list[str] | None
            Additional metadata to be loaded from the dataset.
            If None, the default metadata will be loaded containing
            `subject`, `session` and `run`. If "all", all columns of the `events.tsv`
            file will be loaded. A list of column names can be passed to just
            select these columns in addition to the three default values mentioned
            before. This parameter works regardless of the return type
            (epochs, raws, or array).

        Returns
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

        if process_pipelines is not None:
            assert isinstance(process_pipelines, list)
            assert isinstance(process_pipelines[0], Pipeline)
            output_step_type, _ = process_pipelines[0].steps[-1]
            if (
                (output_step_type == StepType.ARRAY and (return_epochs or return_raws))
                or (output_step_type == StepType.EPOCHS and not return_epochs)
                or (output_step_type == StepType.RAW and not return_raws)
            ):
                raise ValueError(
                    f"process_pipeline output step type {output_step_type} incompatible with "
                    f"arguments {return_epochs=} and {return_raws=}."
                )

        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)

        if subjects is None:
            subjects = dataset.subject_list

        if process_pipelines is None:
            process_pipelines = self.make_process_pipelines(
                dataset, return_epochs, return_raws, postprocess_pipeline
            )

        labels_pipeline = self.make_labels_pipeline(dataset, return_epochs, return_raws)

        data = [
            dataset.get_data(
                subjects=subjects,
                cache_config=cache_config,
                process_pipeline=process_pipeline,
            )
            for process_pipeline in process_pipelines
        ]

        X = []
        labels = []
        metadata = []
        for subject, sessions in data[0].items():
            for session, runs in sessions.items():
                for run in runs.keys():
                    proc = [data_i[subject][session][run] for data_i in data]

                    if additional_metadata:
                        ext_metadata = [
                            dataset.get_additional_metadata(
                                subject=subject, session=session, run=run
                            )
                        ] * len(process_pipelines)

                        if isinstance(additional_metadata, list):
                            ext_metadata = [
                                dm[["session", "subject", "run"] + additional_metadata]
                                for dm in ext_metadata
                            ]
                    else:
                        ext_metadata = [None] * len(process_pipelines)

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

                    # overwrite if additional is required
                    if additional_metadata:
                        # extend the metadata according to the filters

                        dmeta_ext = (
                            ext_metadata[0].copy()
                            if isinstance(ext_metadata[0], pd.DataFrame)
                            else pd.DataFrame()
                        )
                        metadata[-1] = dmeta_ext

                    if return_epochs:
                        x.metadata = (
                            metadata[-1].copy()
                            if len(self.filters) == 1
                            else pd.concat(
                                [metadata[-1].copy()] * len(self.filters),
                                ignore_index=True,
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
                make_fixed_pipeline(
                    ForkPipelines(
                        [
                            ("raw", make_fixed_pipeline(None)),
                            ("events", self._get_events_pipeline(dataset)),
                        ]
                    ),
                    RawToEpochs(
                        event_id=self.used_events(dataset),
                        tmin=bmin,
                        tmax=bmax,
                        baseline=baseline,
                        channels=self.channels,
                        interpolate_missing_channels=self.interpolate_missing_channels,
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
        return FixedPipeline(steps)

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
            steps.append(("postprocess_pipeline", processing_pipeline))
        if len(steps) == 0:
            return None
        return FixedPipeline(steps)

    def match_all(
        self,
        datasets: List[BaseDataset],
        shift=-0.5,
        channel_merge_strategy: str = "intersect",
        ignore=["stim"],
    ):
        """
        Initialize this paradigm to match all datasets in parameter:

        - `self.resample` is set to match the minimum frequency in all datasets, minus `shift`.
          If the frequency is 128 for example, then MNE can return 128 or 129 samples
          depending on the dataset, even if the length of the epochs is 1s
          Setting `shift=-0.5` solves this particular issue.
        - `self.channels` is initialized with the channels which are common to all datasets.

        Parameters
        ----------
        datasets: List[BaseDataset]
            A dataset instance.
        shift: List[BaseDataset]
            Shift the sampling frequency by this value
            E.g.: if sampling=128 and shift=-0.5, then it returns 127.5 Hz
        channel_merge_strategy: str (default: 'intersect')
            Accepts two values:
            - 'intersect': keep only channels common to all datasets
            - 'union': keep all channels from all datasets, removing duplicate
        ignore: List[string]
            A list of channels to ignore

        ..versionadded:: 0.6.0
        """
        resample = None
        channels: set = None
        for dataset in datasets:
            first_subject = dataset.subject_list[0]
            data = dataset.get_data(subjects=[first_subject])[first_subject]
            first_session = list(data.keys())[0]
            session = data[first_session]
            first_run = list(session.keys())[0]
            X = session[first_run]
            info = X.info
            sfreq = info["sfreq"]
            ch_names = info["ch_names"]
            # get the minimum sampling frequency between all datasets
            resample = sfreq if resample is None else min(resample, sfreq)
            # get the channels common to all datasets
            if channels is None:
                channels = set(ch_names)
            elif channel_merge_strategy == "intersect":
                channels = channels.intersection(ch_names)
                self.interpolate_missing_channels = False
            else:
                channels = channels.union(ch_names)
                self.interpolate_missing_channels = True
        # If resample=128 for example, then MNE can returns 128 or 129 samples
        # depending on the dataset, even if the length of the epochs is 1s
        # `shift=-0.5` solves this particular issue.
        self.resample = resample + shift

        # exclude ignored channels
        self.channels = list(channels.difference(ignore))

    @abc.abstractmethod
    def _get_events_pipeline(self, dataset):
        pass


class BaseParadigm(BaseProcessing):
    """Base class for paradigms.

    Parameters
    ----------

    events: List of str | None (default None)
        events to use for epoching. If None, default to all events defined in
        the dataset.

    scorer: sklearn-compatible string or a compatible sklearn scorer | None (default None)
        If None, and n_classes==2 use the roc_auc, else use accuracy.
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
        scorer=None,
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

        # Normalize scorer (convert list of callables to dict)
        scorer = _normalize_scorer(scorer)

        if scorer is not None:
            try:
                check_scoring(None, scoring=scorer)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid scorer: {e}") from e

        self.scorer = scorer

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
        return RawToEvents(event_id=event_id, interval=dataset.interval)
