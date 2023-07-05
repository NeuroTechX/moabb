"""
Base class for a dataset
"""
import abc
import logging
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Union

from sklearn.pipeline import make_pipeline

from moabb.datasets.bids_interface import BIDSInterface


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

    save: bool = True
    use: bool = True
    overwrite: bool = False
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
        process_pipeline=None,
    ):
        """Return the data correspoonding to a list of subjects.

        The returned data is a dictionary with the folowing structure::

            data = {'subject_id' :
                        {'session_id':
                            {'run_id': raw}
                        }
                    }

        subjects are on top, then we have sessions, then runs.
        A sessions is a recording done in a single day, without removing the
        EEG cap. A session is constitued of at least one run. A run is a single
        contigous recording. Some dataset break session in multiple runs.

        Parameters
        ----------
        subjects: List of int
            List of subject number
        cache_config: dict
            Configuration for caching of datasets. See CacheConfig for details.
        process_pipeline: sklearn.pipeline.Pipeline | None
            Pipeline that takes a mne.io.Raw as input, used to process the data.
            If None, no processing is applied.

        Returns
        -------
        data: Dict
            dict containing the raw data
        """
        if subjects is None:
            subjects = self.subject_list

        if not isinstance(subjects, list):
            raise (ValueError("subjects must be a list"))

        cache_config = CacheConfig.make(cache_config)

        process_pipeline = (
            make_pipeline(None) if process_pipeline is None else process_pipeline
        )

        data = dict()
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError("Invalid subject {:d} given".format(subject))
            data[subject] = self._get_single_subject_data_using_cache(
                subject,
                cache_config,
                process_pipeline,
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
        self, subject, cache_config, process_pipeline
    ):
        """
        Either load the data of a single subject from disk cache or from the dataset object,
        then eventually saves or overwrites the cache version depending on the parameters.
        """
        save_cache = cache_config.save

        interface = BIDSInterface(
            self,
            subject,
            path=cache_config.path,
            process_pipeline=process_pipeline,
            verbose=cache_config.verbose,
        )

        # Overwrite:
        if cache_config.overwrite:
            interface.erase()

        # Load:
        sessions_data = None
        if (
            cache_config.use and not cache_config.overwrite
        ):  # can't load if it was just erased
            sessions_data = interface.load(preload=False)
        if sessions_data is not None:
            save_cache = False  # no need to save if we just loaded it
        else:
            # interface2 = (
            #     interface.find_compatible_interface() if cache_config.use else None
            # )
            # if interface2 is not None:
            #     sessions_data = interface2.load(preload=True)
            # else:
            sessions_data = self._get_single_subject_data(subject)
            sessions_data = {
                session: {
                    run: process_pipeline.transform(raw) for run, raw in runs.items()
                }
                for session, runs in sessions_data.items()
            }

        # Save:
        if save_cache:
            try:
                interface.save(sessions_data)
            except Exception as ex:
                # ex_type, ex_value, ex_traceback = sys.exc_info()
                log.warning(
                    f"Failed to save dataset {self.code}, subject {subject} to BIDS format:\n{ex}"
                )
                interface.erase()  # remove partial cache
            else:
                if cache_config.use:
                    sessions_data = interface.load(preload=False)
                    assert (
                        sessions_data is not None
                    )  # should not happen because save succeeded
        return sessions_data

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
