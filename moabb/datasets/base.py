"""Base class for a dataset."""

from __future__ import annotations

import abc
import logging
import re
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any, Dict, Union

import mne_bids
import pandas as pd
from sklearn.pipeline import Pipeline

from moabb.datasets.bids_interface import StepType, _interface_map
from moabb.datasets.preprocessing import SetRawAnnotations


log = logging.getLogger(__name__)

_RAW_EXTENSIONS = [
    ".con",
    ".sqd",
    ".pdf",
    ".fif",
    ".ds",
    ".vhdr",
    ".set",
    ".edf",
    ".bdf",
    ".EDF",
    ".snirf",
    ".cdt",
    ".mef",
    ".nwb",
]


def get_summary_table(paradigm: str, dir_name: str | None = None):
    if dir_name is None:
        dir_name = Path(__file__).parent
    path = Path(dir_name) / f"summary_{paradigm}.csv"
    df = pd.read_csv(
        path,
        header=0,
        index_col="Dataset",
        skipinitialspace=True,
        dtype={"PapersWithCode leaderboard": str},
    )
    return df


_summary_table_imagery = get_summary_table("imagery")
_summary_table_p300 = get_summary_table("p300")
_summary_table_ssvep = get_summary_table("ssvep")
_summary_table_cvep = get_summary_table("cvep")
_summary_table_rstate = get_summary_table("rstate")
_summary_table = pd.concat(
    [
        _summary_table_imagery,
        _summary_table_p300,
        _summary_table_ssvep,
        _summary_table_cvep,
        _summary_table_rstate,
    ],
)


@dataclass
class CacheConfig:
    """
    Configuration for caching of datasets.

    Parameters
    ----------
    save_*: bool
        This flag specifies whether to save the output of the corresponding
        step to disk.
    use: bool
        This flag specifies whether to use the disk cache in case it exists.
        If True, the Raw or Epochs objects returned will not be preloaded
        (this saves some time). Otherwise, they will be preloaded.
        If use is False, the save_* and overwrite_* keys will be ignored.
    overwrite_*: bool
        This flag specifies whether to overwrite the disk cache in
        case it exist.
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    verbose:
        Verbosity level. See mne.verbose.

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    save_raw: bool = False
    save_epochs: bool = False
    save_array: bool = False

    use: bool = False

    overwrite_raw: bool = False
    overwrite_epochs: bool = False
    overwrite_array: bool = False

    path: Union[str, Path] = None
    verbose: str = None

    @classmethod
    def make(cls, dic: Union[None, Dict, "CacheConfig"] = None) -> "CacheConfig":
        """
        Create a CacheConfig object from a dict or another CacheConfig object.

        Examples
        --------
        Using default parameters:

        >>> CacheConfig.make()
        CacheConfig(save=True, use=True, overwrite=True, path=None)

        From a dict:

        >>> dic = {'save': False}
        >>> CacheConfig.make(dic)
        CacheConfig(save=False, use=True, overwrite=True, path=None)
        """
        if dic is None:
            return cls()
        elif isinstance(dic, dict):
            return cls(**dic)
        elif isinstance(dic, cls):
            return dic
        else:
            raise ValueError(f"Expected dict or CacheConfig, got {type(dic)}")


def apply_step(pipeline, obj):
    """Apply a pipeline to an object."""
    if obj is None:
        return None
    try:
        return pipeline.transform(obj)
    except ValueError as error:
        # no events received by RawToEpochs:
        if str(error) == "No events found":
            return None
        raise error


def is_camel_kebab_case(name: str):
    """Check if a string is in CamelCase but can also contain dashes."""
    return re.fullmatch(r"[a-zA-Z0-9\-]+", name) is not None


def is_abbrev(abbrev_name: str, full_name: str):
    """Check if abbrev_name is an abbreviation of full_name,
    i.e. ifthe characters in abbrev_name are all in full_name
    and in the same order. They must share the same capital letters."""
    pattern = re.sub(r"([A-Za-z])", r"\1[a-z0-9\-]*", re.escape(abbrev_name))
    return re.fullmatch(pattern, full_name) is not None


def check_subject_names(data):
    for subject in data.keys():
        if not isinstance(subject, (int, str)):
            raise ValueError(
                f"Subject names must be integers or strings, found {type(subject)}: {subject!r}. "
                f"If you used cache, you may need to erase it using overwrite=True."
            )


def session_run_pattern():
    return r"([0-9]+)(|[a-zA-Z]+[a-zA-Z0-9]*)"  # g1: index, g2: description


constraint_message = (
    "names must be strings starting with an integer "
    "identifying the order in which they were recorded, "
    "optionally followed by a description only containing "
    "letters and numbers."
)


def check_session_names(data):
    pattern = session_run_pattern()
    for subject, sessions in data.items():
        indexes = []
        for session in sessions.keys():
            match = re.fullmatch(pattern, session)
            if not isinstance(session, str) or not match:
                raise ValueError(
                    f"Session {constraint_message} Found key {session!r} instead. "
                    f"If you used cache, you may need to erase it using overwrite=True."
                )
            indexes.append(int(match.groups()[0]))
        if not len(indexes) == len(set(indexes)):
            raise ValueError(
                f"Session {constraint_message} Found duplicate index {list(sessions.keys())}."
            )


def check_run_names(data):
    pattern = session_run_pattern()
    for subject, sessions in data.items():
        for session, runs in sessions.items():
            indexes = []
            for run in runs.keys():
                match = re.fullmatch(pattern, run)
                if not isinstance(run, str) or not match:
                    raise ValueError(
                        f"Run {constraint_message} Found key {run!r} instead. "
                        f"If you used cache, you may need to erase it using overwrite=True."
                    )
                indexes.append(int(match.groups()[0]))
            if not len(indexes) == len(set(indexes)):
                raise ValueError(
                    f"Run {constraint_message} Found duplicate index {list(runs.keys())}."
                )


def _transfer_unit(key: str, value: str):
    pattern = r"( ?\((\w+)\))$"
    match = re.search(pattern, key)
    if match:
        suffix, unit = match.groups()
        return key[: -len(suffix)], f"{value} {unit}"
    return key, value


def format_row(row: pd.Series, horizontal: bool = True):
    pwc_key = "PapersWithCode leaderboard"
    tab_prefix = " " * 8
    tab_sep = "="
    row = row[~row.isna()]
    pwc_link = row.get(pwc_key, None)
    if pwc_link is not None:
        row = row.drop(pwc_key)

    def to_int(x):
        try:
            i = int(x)
            if i == x:
                return i
            return x
        except ValueError:
            return x

    # append the eventual units to the values:
    keys, values = zip(
        *[_transfer_unit(str(key), str(to_int(val))) for key, val in row.items()]
    )
    # make columns bold:
    keys: Sequence[str] = [f"**{key}**" for key in keys]
    # transpose the table if vertical:
    rows: Sequence[Sequence[str]] = (
        [keys, values] if horizontal else list(zip(keys, values))
    )
    # compute the width of each column:
    widths = [max(map(len, col)) for col in zip(*rows)]
    # pad each column with spaces:
    rows = [[str(col).rjust(width) for col, width in zip(row, widths)] for row in rows]
    # add separator rows:
    sep_row = [tab_sep * width for width in widths]
    if horizontal:
        rows.insert(1, sep_row)
    rows.insert(0, sep_row)
    rows.append(sep_row)
    # join the columns and rows into one string:
    rows_str = "\n".join([f"{tab_prefix}{' '.join(row)}" for row in rows])
    # add the header:
    out = f"    .. admonition:: Dataset summary\n\n{rows_str}"
    # add the PapersWithCode link if it exists:
    if pwc_link is not None:
        out = f"    **{pwc_key}:** {pwc_link}\n\n" + out
    return out, row


class MetaclassDataset(abc.ABCMeta):
    def __new__(cls, name, bases, attrs):
        doc = attrs.get("__doc__", "")
        try:
            row = _summary_table.loc[name]
            row_str, row = format_row(row, horizontal=False)
            doc_list = doc.split("\n\n")
            if len(doc_list) >= 2:
                doc_list = [doc_list[0], row_str] + doc_list[1:]
            else:
                doc_list.append(row_str)
            attrs["__doc__"] = "\n\n".join(doc_list)
            attrs["_summary_table"] = row.to_dict()
        except KeyError:
            log.debug(
                f"No description found for dataset {name}. "
                f"Complete the appropriate moabb/datasets/summary_*.csv file"
            )
        return super().__new__(cls, name, bases, attrs)


class BaseDataset(metaclass=MetaclassDataset):
    """Abstract Moabb BaseDataset.

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
        Unique identifier for dataset, used in all plots.
        The code should be in CamelCase.

    interval: list with 2 entries
        Imagery interval as defined in the dataset description

    paradigm: ['p300','imagery', 'ssvep']
        Defines what sort of dataset this is

    doi: DOI for dataset, optional (for now)
    """

    _summary_table: dict[str, Any]

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
        """Initialize function for the BaseDataset."""
        try:
            _ = iter(subjects)
        except TypeError:
            raise ValueError("subjects must be a iterable, like a list") from None

        if not is_camel_kebab_case(code):
            raise ValueError(
                f"code {code!r} must be in Camel-KebabCase; "
                "i.e. use CamelCase, and add dashes where absolutely necessary. "
                "See moabb.datasets.base.is_camel_kebab_case for more information."
            )
        class_name = self.__class__.__name__.replace("_", "-")
        if not is_abbrev(class_name, code):
            log.warning(
                f"The dataset class name {class_name!r} must be an abbreviation "
                f"of its code {code!r}. "
                "See moabb.datasets.base.is_abbrev for more information."
            )

        self.subject_list = subjects
        self.n_sessions = sessions_per_subject
        self.event_id = events
        self.code = code
        self.interval = interval
        self.paradigm = paradigm
        self.doi = doi
        self.unit_factor = unit_factor

    def _create_process_pipeline(self):
        return Pipeline(
            [
                (
                    StepType.RAW,
                    SetRawAnnotations(
                        self.event_id,
                        interval=self.interval,
                    ),
                ),
            ]
        )

    def get_data(
        self,
        subjects=None,
        cache_config=None,
        process_pipeline=None,
    ):
        """
        Return the data corresponding to a list of subjects.

        The returned data is a dictionary with the following structure::

            data = {'subject_id' :
                        {'session_id':
                            {'run_id': run}
                        }
                    }

        subjects are on top, then we have sessions, then runs.
        A sessions is a recording done in a single day, without removing the
        EEG cap. A session is constitued of at least one run. A run is a single
        contiguous recording. Some dataset break session in multiple runs.

        Processing steps can optionally be applied to the data using the
        ``*_pipeline`` arguments. These pipelines are applied in the
        following order: ``raw_pipeline`` -> ``epochs_pipeline`` ->
        ``array_pipeline``. If a ``*_pipeline`` argument is ``None``,
        the step will be skipped. Therefore, the ``array_pipeline`` may
        either receive a :class:`mne.io.Raw` or a :class:`mne.Epochs` object
        as input depending on whether ``epochs_pipeline`` is ``None`` or not.

        Parameters
        ----------
        subjects: List of int
            List of subject number
        cache_config: dict | CacheConfig
            Configuration for caching of datasets. See ``CacheConfig``
            for details.
        process_pipeline: Pipeline | None
            Optional processing pipeline to apply to the data.
            To generate an adequate pipeline, we recommend using
            :func:`moabb.utils.make_process_pipelines`.
            This pipeline will receive :class:`mne.io.BaseRaw` objects.
            The steps names of this pipeline should be elements of :class:`StepType`.
            According to their name, the steps should either return a
            :class:`mne.io.BaseRaw`, a :class:`mne.Epochs`, or a :func:`numpy.ndarray`.
            This pipeline must be "fixed" because it will not be trained,
            i.e. no call to ``fit`` will be made.

        Returns
        -------
        data: Dict
            dict containing the raw data
        """
        if subjects is None:
            subjects = self.subject_list

        if not isinstance(subjects, list):
            raise ValueError("subjects must be a list")

        cache_config = CacheConfig.make(cache_config)

        if process_pipeline is None:
            process_pipeline = self._create_process_pipeline()

        data = dict()
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError("Invalid subject {:d} given".format(subject))
            data[subject] = self._get_single_subject_data_using_cache(
                subject,
                cache_config,
                process_pipeline,
            )
        check_subject_names(data)
        check_session_names(data)
        check_run_names(data)
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

        This function is only useful to download all the dataset at once.


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
                # pylint: disable-next=unexpected-keyword-arg
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
        """Load a single subject's data using cache.

        Either load the data of a single subject from disk cache or from the
        dataset object,
        then eventually saves or overwrites the cache version depending on the
        parameters.
        """
        steps = list(process_pipeline.steps)
        splitted_steps = []  # list of (cached_steps, remaining_steps)
        if cache_config.use:
            splitted_steps += [
                (steps[:i], steps[i:]) for i in range(len(steps), 0, -1)
            ]  # [len(steps)...1]
        splitted_steps.append(
            ([], steps)
        )  # last option:  if cached_steps is [], we don't use cache, i.e. i=0

        for cached_steps, remaining_steps in splitted_steps:
            sessions_data = None
            # Load and eventually overwrite:
            if len(cached_steps) == 0:  # last option: we don't use cache
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
                    )  # None if cache inexistent

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
                    (
                        cache_config.save_raw
                        and step_type is StepType.RAW
                        and (
                            (step_idx == len(remaining_steps) - 1)
                            or (remaining_steps[step_idx + 1][0] is not StepType.RAW)
                        )
                    )  # we only save the last raw step
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
                            f"Failed to save {interface.__repr__()} "
                            f"to BIDS format:\n"
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

        The returned data is a dictionary with the following structure

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
    ) -> list[str | Path]:
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


class BaseBIDSDataset(BaseDataset):
    """Abstract BIDS dataset class.

    This abstract class can be used to facilitate the integration of datasets which are
    provided in the Brain Imaging Data Structure (BIDS) format into MOABB.

    More information about BIDS can be found at https://bids.neuroimaging.io/.

    The method ``_download_subject`` must be implemented in each subclass
    (see its docstring for more details).

    If necessary, the methods ``_get_path_search_params`` and
    ``_get_read_extra_params`` can be implemented in the subclass.
    """

    def _get_path_search_params(self, subject: int | None) -> dict[str, Any]:
        """Return the kwargs for the :func:`mne_bids.find_matching_paths` function."""
        out = {"extensions": _RAW_EXTENSIONS}
        if subject is not None:
            out["subjects"] = str(subject)
        return out

    def _get_read_extra_params(
        self,
        subject: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any] | None:
        """Return the ``extra_params`` argument for the :func:`mne_bids.read_raw_bids` function."""
        return None

    @staticmethod
    def _find_matching_paths(root, **kwargs) -> list[mne_bids.BIDSPath]:
        bids_paths = mne_bids.find_matching_paths(root=root, **kwargs)
        # Remove JSON files manually (the ignore_json argument only arrives in mne-bids=0.16)
        return [bids_path for bids_path in bids_paths if bids_path.extension != ".json"]

    @abc.abstractmethod
    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download the data of a single subject and return the local path to the ROOT of the BIDS dataset.

        Returns
        -------
        root : str
            Path to the ROOT of the BIDS dataset.
        """
        pass

    def bids_paths(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ) -> list[mne_bids.BIDSPath]:
        root = self._download_subject(subject, path, force_update, update_path, verbose)
        return self._find_matching_paths(
            root=root, **self._get_path_search_params(subject)
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        bids_paths = self.bids_paths(subject, path, force_update, update_path, verbose)
        return [bids_path.fpath for bids_path in bids_paths]

    def _get_single_subject_data(self, subject):
        bids_paths = self.bids_paths(subject)
        data = {}
        for bids_path in bids_paths:
            raw = mne_bids.read_raw_bids(
                bids_path, extra_params=self._get_read_extra_params(subject)
            )
            # Data needs to be preloaded for the filtering step of paradigms
            raw.load_data()

            if bids_path.session is None:
                log.warning(
                    "Session not found for subject='%s'. Using session='0'", subject
                )
                session = "0"
            else:
                session = bids_path.session
            if bids_path.run is None:
                log.warning(
                    "Run not found for subject='%s', session='%s'. Using run='0'",
                    subject,
                    session,
                )
                run = "0"
            else:
                run = bids_path.run
            data.setdefault(session, {})[run] = raw
        return data


class LocalBIDSDataset(BaseBIDSDataset):
    """Generic local/private BIDS datasets.

    This class is useful if you have a local/private dataset in BIDS format
    and you want to use it with MOABB, without having to create a new dataset class.

    Parameters
    ----------
    bids_root : str | Path
        Local path to the root of the BIDS dataset.
    path_search_params : dict[str, Any] | None
        Additional kwargs for the :func:`mne_bids.find_matching_paths` function.
    read_extra_params : dict[str, Any] | None
        Additional kwargs for the :func:`mne_bids.read_raw_bids` function.
    subjects : list[int] | None
        Optional list of subjects. If None, the subjects are inferred from the dataset.
    sessions_per_subject : int | None
        Optional number of sessions per subject. If None, the number is inferred from the dataset.
    events : dict[str, str]
        String codes for events matched with labels in the stim channel.
    interval : list with 2 entries
        Imagery interval as defined in the dataset description.
    paradigm : str
        Defines what sort of dataset this is.
    doi : str | None
        Optional DOI for dataset.
    code : str
        Unique identifier for the dataset. for compatibility reasons,
        it should start with ``"LocalBIDSDataset"``
    unit_factor : float
        Factor to convert units to microvolts (default: 1e6).
    """

    def __init__(
        self,
        bids_root: Path | str,
        path_search_params: dict[str, Any] | None = None,
        read_extra_params: dict[str, Any] | None = None,
        *,
        subjects: list[int] | None = None,
        sessions_per_subject: int | None = None,
        events,
        code="LocalBIDSDataset-",
        interval,
        paradigm,
        doi=None,
        unit_factor=1e6,
    ):
        self.bids_root = bids_root
        self.path_search_params = path_search_params
        self.read_extra_params = read_extra_params
        bids_paths = self._find_matching_paths(
            root=bids_root, **self._get_path_search_params(None)
        )
        if len(bids_paths) == 0:
            raise ValueError(f"No BIDS dataset found in {bids_root}")
        if subjects is None or sessions_per_subject is None:
            if subjects is None:
                subjects = sorted(set(path.subject for path in bids_paths))
                log.warning(f"Found subjects: {subjects}")
            if sessions_per_subject is None:
                sessions_per_subject = min(
                    len(
                        set(
                            bids_path.session
                            for bids_path in bids_paths
                            if bids_path.subject == subject
                        )
                    )
                    for subject in subjects
                )
                log.warning(f"Found {sessions_per_subject=}")

        super().__init__(
            subjects,
            sessions_per_subject,
            events,
            code,
            interval,
            paradigm,
            doi,
            unit_factor,
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        return self.bids_root

    def _get_path_search_params(self, subject):
        return dict(
            super()._get_path_search_params(subject), **(self.path_search_params or {})
        )

    def _get_read_extra_params(self, subject):
        return self.read_extra_params
