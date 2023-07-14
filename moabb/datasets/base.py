"""
Base class for a dataset
"""
import abc
import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from pathlib import Path
from typing import Type, Union

from sklearn.pipeline import Pipeline, make_pipeline

from moabb.datasets.bids_interface import (
    BIDSInterfaceBase,
    BIDSInterfaceEpochs,
    BIDSInterfaceNumpyArray,
    BIDSInterfaceRawEDF,
)
from moabb.datasets.preprocessing import (
    EpochsToEvents,
    EventsToLabels,
    ForkPipelines,
    RawToEvents,
    _is_none_pipeline,
)


log = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """
    Configuration for caching of datasets.

    Parameters
    ----------
    save: boolean
        This flag specifies whether to save the processed mne.io.Raw to disk
    use: boolean
        This flag specifies whether to use the processed mne.io.Raw from disk
        in case they exist. If True, the Raw objects returned will not be preloaded
        (this saves some time). Otherwise, they will be preloaded.
    overwrite: boolean
        This flag specifies whether to overwrite the processed mne.io.Raw on disk
        in case they exist
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    verbose:
        Verbosity level. See mne.verbose.
    """

    save_raw: bool = False
    save_epochs: bool = False
    save_array: bool = False

    # use_raw: bool = True
    # use_epochs: bool = True
    # use_array: bool = True
    use: bool = False

    overwrite_raw: bool = False
    overwrite_epochs: bool = False
    overwrite_array: bool = False

    path: Union[str, Path] = None
    verbose = None

    @classmethod
    def make(cls, d: Union[None, dict, "CacheConfig"] = None) -> "CacheConfig":
        """
        Create a CacheConfig object from a dict or another CacheConfig object.

        Examples
        -------
        Using default parameters:

        >>> CacheConfig.make()
        CacheConfig(save=True, use=True, overwrite=True, path=None)

        From a dict:

        >>> d = {'save': False}
        >>> CacheConfig.make(d)
        CacheConfig(save=False, use=True, overwrite=True, path=None)
        """
        if d is None:
            return cls()
        elif isinstance(d, dict):
            return cls(**d)
        elif isinstance(d, cls):
            return d
        else:
            raise ValueError(f"Expected dict or CacheConfig, got {type(d)}")


class StepType(Enum):
    RAW = "raw"
    EPOCHS = "epochs"
    ARRAY = "array"


_interface_map: dict[StepType, Type[BIDSInterfaceBase]] = {
    StepType.RAW: BIDSInterfaceRawEDF,
    StepType.EPOCHS: BIDSInterfaceEpochs,
    StepType.ARRAY: BIDSInterfaceNumpyArray,
}


def apply_step(pipeline, obj):
    if obj is None:
        return None
    try:
        return pipeline.transform(obj)
    except ValueError as e:
        # no events received by RawToEpochs:
        if "No events found" == str(e):
            return None
        raise e


class BaseDataset(metaclass=abc.ABCMeta):
    """BaseDataset

    Parameters required for all datasets

    parameters
    ----------
    subjects: List of int
        List of subject number (or tuple or numpy array)

    sessions_per_subject: int
        Number of sessions per subject (if varying, take minimum)

    events: dict of strings
        String codes for events matched with labels in the stim channel.
        Currently imagery codes codes can include:
        - left_hand
        - right_hand
        - hands
        - feet
        - rest
        - left_hand_right_foot
        - right_hand_left_foot
        - tongue
        - navigation
        - subtraction
        - word_ass (for word association)

    code: string
        Unique identifier for dataset, used in all plots

    interval: list with 2 entries
        Imagery interval as defined in the dataset description

    paradigm: ['p300','imagery', 'ssvep']
        Defines what sort of dataset this is

    doi: DOI for dataset, optional (for now)
    """

    def __init__(
        self,
        subjects,
        sessions_per_subject,
        events,
        code,
        interval,
        paradigm,
        doi=None,
        unit_factor=1e6,
    ):
        try:
            _ = iter(subjects)
        except TypeError:
            raise ValueError("subjects must be a iterable, like a list") from None

        self.subject_list = subjects
        self.n_sessions = sessions_per_subject
        self.event_id = events
        self.code = code
        self.interval = interval
        self.paradigm = paradigm
        self.doi = doi
        self.unit_factor = unit_factor

    def get_data(
        self,
        subjects=None,
        cache_config=None,
        raw_pipeline=None,
        epochs_pipeline=None,
        array_pipeline=None,
        event_id=None,
    ):
        """Return the data correspoonding to a list of subjects.

        The returned data is a dictionary with the folowing structure::

            data = {'subject_id' :
                        {'session_id':
                            {'run_id': run}
                        }
                    }

        subjects are on top, then we have sessions, then runs.
        A sessions is a recording done in a single day, without removing the
        EEG cap. A session is constitued of at least one run. A run is a single
        contigous recording. Some dataset break session in multiple runs.

        Processing steps can optionally be applied to the data using the
        ``*_pipeline`` arguments. These pipelines are applied in the following order:
        ``raw_pipeline`` -> ``epochs_pipeline`` -> ``array_pipeline``. If a ``*_pipeline`` argument
        is ``None``, the step will be skipped. Therefor, the ``array_pipeline`` may either
        receive a ``mne.io.Raw`` or a ``mne.Epochs`` object as input depending on whether
        ``epochs_pipeline`` is ``None`` or not.

        Parameters
        ----------
        subjects: List of int
            List of subject number
        cache_config: dict
            Configuration for caching of datasets. See CacheConfig for details.
        raw_pipeline: sklearn.pipeline.Pipeline | sklearn.base.TransformerMixin | None
            Pipeline that necessarily takes a mne.io.Raw as input,
            and necessarily returns a ``mne.io.Raw`` as output.
        epochs_pipeline: sklearn.pipeline.Pipeline | sklearn.base.TransformerMixin | None
            Pipeline that necessarily takes a mne.io.Raw as input,
            and necessarily returns a ``mne.Epochs`` as output.
        array_pipeline: sklearn.pipeline.Pipeline | sklearn.base.TransformerMixin | None
            Pipeline either takes as input a ``mne.Epochs`` if epochs_pipeline
            is not ``None``, or a ``mne.io.Raw`` otherwise. It necessarily returns
            a ``numpy.ndarray`` as output.
            If array_pipeline is not None, each run will be a dict with keys "X" and "y"
            corresponding respectively to the array itself and the corresponding labels.
        event_id: dict
            Event ids to use for generating labels. Only used if ``array_pipeline``
            is not ``None``. If ``None``, all the events of the dataset will be used.

        Returns
        -------
        data: Dict
            dict containing the raw data
        """
        if subjects is None:
            subjects = self.subject_list

        if not isinstance(subjects, list):
            raise (ValueError("subjects must be a list"))

        if event_id is None and array_pipeline is not None:
            log.warning(
                f"event_id not specified, using all the dataset's "
                f"events to generate labels: {self.event_id}"
            )
            event_id = self.event_id

        cache_config = CacheConfig.make(cache_config)

        steps = []
        if raw_pipeline is not None:
            steps.append((StepType.RAW, raw_pipeline))
        if epochs_pipeline is not None:
            steps.append((StepType.EPOCHS, epochs_pipeline))
        if array_pipeline is not None:
            labels_pipeline = Pipeline(
                [
                    (
                        "events",
                        RawToEvents(event_id)
                        if epochs_pipeline is None
                        else EpochsToEvents(),
                    ),
                    ("labels", EventsToLabels(event_id)),
                ]
            )
            array_labels_pipeline = ForkPipelines(
                [
                    ("X", array_pipeline),
                    ("y", labels_pipeline),  # todo: only used events
                ]
            )
            steps.append((StepType.ARRAY, array_labels_pipeline))
        if len(steps) == 0:
            steps.append((StepType.RAW, make_pipeline(None)))

        data = dict()
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError("Invalid subject {:d} given".format(subject))
            data[subject] = self._get_single_subject_data_using_cache(
                subject,
                cache_config,
                steps,
            )

        return data

    def download(
        self,
        subject_list=None,
        path=None,
        force_update=False,
        update_path=None,
        accept=False,
        verbose=None,
    ):
        """Download all data from the dataset.

        This function is only usefull to download all the dataset at once.


        Parameters
        ----------
        subject_list : list of int | None
            List of subjects id to download, if None all subjects
            are downloaded.
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted.
        accept: bool
            Accept licence term to download the data, if any. Default: False
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see :func:`mne.verbose`).
        """
        if subject_list is None:
            subject_list = self.subject_list
        for subject in subject_list:
            # check if accept is needed
            sig = signature(self.data_path)
            if "accept" in [str(p) for p in sig.parameters]:
                self.data_path(
                    subject=subject,
                    path=path,
                    force_update=force_update,
                    update_path=update_path,
                    verbose=verbose,
                    accept=accept,
                )
            else:
                self.data_path(
                    subject=subject,
                    path=path,
                    force_update=force_update,
                    update_path=update_path,
                    verbose=verbose,
                )

    def _get_single_subject_data_using_cache(
        self,
        subject,
        cache_config,
        steps,
    ):
        """
        Either load the data of a single subject from disk cache or from the dataset object,
        then eventually saves or overwrites the cache version depending on the parameters.
        """
        splitted_steps = []  # list of (cached_steps, remaining_steps)
        if cache_config.use:
            splitted_steps += [(steps[:i], steps[i:]) for i in range(len(steps), 0, -1)]
            if _is_none_pipeline(steps[0]):  # case where step was not already "empty"
                splitted_steps.append(([(StepType.RAW, make_pipeline(None))], steps))
        splitted_steps.append(([], steps))  # last option: we don't use cache at all

        for cached_steps, remaining_steps in splitted_steps:
            sessions_data = None
            # Load and eventually overwrite:
            if len(cached_steps) == 0:
                sessions_data = self._get_single_subject_data(subject)
                assert sessions_data is not None  # should not happen
            else:
                cache_type = cached_steps[-1][0]
                interface = _interface_map[cache_type](
                    self,
                    subject,
                    path=cache_config.path,
                    process_pipeline=Pipeline(cached_steps),
                    verbose=cache_config.verbose,
                )

                if (
                    (cache_config.overwrite_raw and cache_type is StepType.RAW)
                    or (cache_config.overwrite_epochs and cache_type is StepType.EPOCHS)
                    or (cache_config.overwrite_array and cache_type is StepType.ARRAY)
                ):
                    interface.erase()
                elif cache_config.use:  # can't load if it was just erased
                    sessions_data = interface.load(
                        preload=False
                    )  # None if cache inexistant

            # If no cache was found or if it was erased, try the next option:
            if sessions_data is None:
                continue

            # Apply remaining steps and save:
            for step_idx, (step_type, process_pipeline) in enumerate(remaining_steps):
                # apply one step:
                sessions_data = {
                    session: {
                        run: apply_step(process_pipeline, raw)
                        for run, raw in runs.items()
                    }
                    for session, runs in sessions_data.items()
                }

                # save:
                if (
                    (cache_config.save_raw and step_type is StepType.RAW)
                    or (cache_config.save_epochs and step_type is StepType.EPOCHS)
                    or (cache_config.save_array and step_type is StepType.ARRAY)
                ):
                    interface = _interface_map[step_type](
                        self,
                        subject,
                        path=cache_config.path,
                        process_pipeline=Pipeline(
                            cached_steps + remaining_steps[: step_idx + 1]
                        ),
                        verbose=cache_config.verbose,
                    )
                    try:
                        interface.save(sessions_data)
                    except Exception:
                        log.warning(
                            f"Failed to save {interface.__repr__()} to BIDS format:\n"
                            f"{' Pipeline: '.center(50, '#')}\n"
                            f"{interface.process_pipeline.__repr__()}\n"
                            f"{' Exception: '.center(50, '#')}\n"
                            f"{''.join(traceback.format_exc())}{'#' * 50}"
                        )
                        interface.erase()  # remove partial cache
            return sessions_data
        raise ValueError("should not happen")

    @abc.abstractmethod
    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        The returned data is a dictionary with the folowing structure

        data = {'session_id':
                    {'run_id': raw}
                }

        parameters
        ----------
        subject: int
            subject number

        returns
        -------
        data: Dict
            dict containing the raw data
        """
        pass

    @abc.abstractmethod
    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Get path to local copy of a subject data.

        Parameters
        ----------
        subject : int
            Number of subject to use
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None **Deprecated**
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see :func:`mne.verbose`).

        Returns
        -------
        path : list of str
            Local path to the given data file. This path is contained inside a
            list of length one, for compatibility.
        """  # noqa: E501
        pass
