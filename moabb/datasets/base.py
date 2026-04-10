"""Base class for a dataset."""

from __future__ import annotations

import abc
import logging
import re
import traceback
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Union
from urllib.parse import quote

import mne_bids
import numpy as np
import pandas as pd
from mne_bids import events_file_to_annotation_kwargs

from moabb.datasets.bids_interface import (
    _FORMAT_EXTENSION_MAP,
    StepType,
    _BIDSInterfaceRawEDFNoDesc,
    _enrich_raw_info_from_metadata,
    _interface_map,
    get_bids_root,
)
from moabb.datasets.preprocessing import FixedPipeline, SetRawAnnotations


if TYPE_CHECKING:
    pass


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
    df = pd.read_csv(path, header=0, index_col="Dataset", skipinitialspace=True)
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
    ]
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

        >>> dic = {"save": False}
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


def _is_event_int(v):
    """Return True if v is int or np.integer but not bool."""
    return not isinstance(v, bool) and isinstance(v, (int, np.integer))


_KWARG_HINT = (
    "Check that keyword arguments were not accidentally included inside the events dict."
)


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
    for _subject, sessions in data.items():
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
    for _subject, sessions in data.items():
        for _session, runs in sessions.items():
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
    tab_prefix = " " * 8
    tab_sep = "="
    row = row[~row.isna()]

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
    return out, row


def _has_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_metadata_value(v) for v in value)
    return str(value)


def _metadata_admonition_block(
    title: str, items: list[tuple[str, Any]], existing_doc: str
) -> str | None:
    if f".. admonition:: {title}" in existing_doc:
        return None

    lines = []
    for label, value in items:
        if not _has_nonempty(value):
            continue
        lines.append(f"        - **{label}**: {_format_metadata_value(value)}")

    if not lines:
        return None
    return "\n".join([f"    .. admonition:: {title}", "", *lines])


def _format_age(participants) -> str | None:
    age_mean = getattr(participants, "age_mean", None)
    age_min = getattr(participants, "age_min", None)
    age_max = getattr(participants, "age_max", None)
    if age_mean is None:
        return None
    age_text = _format_metadata_value(age_mean)
    if age_min is not None and age_max is not None:
        age_text += f" (range: {_format_metadata_value(age_min)}-{_format_metadata_value(age_max)})"
    return f"{age_text} years"


def _format_bandpass(preprocessing) -> str | None:
    bandpass = getattr(preprocessing, "bandpass", None)
    if isinstance(bandpass, dict):
        low = bandpass.get(
            "low",
            bandpass.get(
                "highpass", bandpass.get("low_cutoff_hz", bandpass.get("highpass_hz"))
            ),
        )
        high = bandpass.get(
            "high",
            bandpass.get(
                "lowpass", bandpass.get("high_cutoff_hz", bandpass.get("lowpass_hz"))
            ),
        )
        if low is not None and high is not None:
            return f"{_format_metadata_value(low)}-{_format_metadata_value(high)} Hz"
    elif isinstance(bandpass, (list, tuple)) and len(bandpass) >= 2:
        return (
            f"{_format_metadata_value(bandpass[0])}"
            f"-{_format_metadata_value(bandpass[1])} Hz"
        )

    highpass = getattr(preprocessing, "highpass_hz", None)
    lowpass = getattr(preprocessing, "lowpass_hz", None)
    if highpass is not None and lowpass is not None:
        return f"{_format_metadata_value(highpass)}-{_format_metadata_value(lowpass)} Hz"
    return None


def _metadata_doc_sections(metadata: Any, existing_doc: str) -> str:
    if metadata is None:
        return ""

    participants = getattr(metadata, "participants", None)
    acquisition = getattr(metadata, "acquisition", None)
    experiment = getattr(metadata, "experiment", None)
    documentation = getattr(metadata, "documentation", None)
    preprocessing = getattr(metadata, "preprocessing", None)
    external_links = getattr(metadata, "external_links", None)

    blocks = []

    if participants is not None:
        blocks.append(
            _metadata_admonition_block(
                "Participants",
                [
                    ("Population", getattr(participants, "health_status", None)),
                    (
                        "Clinical population",
                        getattr(participants, "clinical_population", None),
                    ),
                    ("Age", _format_age(participants)),
                    ("Handedness", getattr(participants, "handedness", None)),
                    ("BCI experience", getattr(participants, "bci_experience", None)),
                ],
                existing_doc,
            )
        )

    if acquisition is not None:
        blocks.append(
            _metadata_admonition_block(
                "Equipment",
                [
                    ("Amplifier", getattr(acquisition, "hardware", None)),
                    ("Electrodes", getattr(acquisition, "sensor_type", None)),
                    ("Montage", getattr(acquisition, "montage", None)),
                    ("Reference", getattr(acquisition, "reference", None)),
                ],
                existing_doc,
            )
        )

    if preprocessing is not None:
        steps = getattr(preprocessing, "preprocessing_steps", None)
        steps_text = ", ".join(steps) if isinstance(steps, list) else steps
        blocks.append(
            _metadata_admonition_block(
                "Preprocessing",
                [
                    ("Data state", getattr(preprocessing, "data_state", None)),
                    ("Bandpass filter", _format_bandpass(preprocessing)),
                    ("Steps", steps_text),
                    ("Re-reference", getattr(preprocessing, "re_reference", None)),
                    ("Notes", getattr(preprocessing, "notes", None)),
                ],
                existing_doc,
            )
        )

    data_url = None
    if documentation is not None:
        data_url = getattr(documentation, "data_url", None)
    if data_url is None and external_links is not None:
        data_url = (
            external_links.get("source") if isinstance(external_links, dict) else None
        )

    if documentation is not None or _has_nonempty(data_url):
        blocks.append(
            _metadata_admonition_block(
                "Data Access",
                [
                    (
                        "DOI",
                        getattr(documentation, "doi", None) if documentation else None,
                    ),
                    ("Data URL", data_url),
                    (
                        "Repository",
                        (
                            getattr(documentation, "repository", None)
                            if documentation
                            else None
                        ),
                    ),
                ],
                existing_doc,
            )
        )

    if experiment is not None:
        blocks.append(
            _metadata_admonition_block(
                "Experimental Protocol",
                [
                    ("Paradigm", getattr(experiment, "paradigm", None)),
                    ("Task type", getattr(experiment, "task_type", None)),
                    ("Tasks", getattr(experiment, "tasks", None)),
                    ("Feedback", getattr(experiment, "feedback_type", None)),
                    ("Stimulus", getattr(experiment, "stimulus_type", None)),
                ],
                existing_doc,
            )
        )

    blocks = [block for block in blocks if block is not None]
    return "\n\n".join(blocks)


def _format_feedback_section(dataset_id: str) -> str:
    """Generate a feedback section with a button to report issues on GitHub."""
    issue_title = quote(f"[Dataset] Issue with {dataset_id}")
    issue_body = quote(
        f"## Dataset\n\n"
        f"- **Dataset ID:** {dataset_id}\n\n"
        f"## Issue Description\n\n"
        f"Please describe the issue you encountered with this dataset:\n\n"
        f"## Steps to Reproduce\n\n"
        f"1. \n2. \n3. \n\n"
        f"## Expected Behavior\n\n\n"
        f"## Additional Context\n\n"
    )
    github_url = (
        f"https://github.com/NeuroTechX/moabb/issues/new"
        f"?title={issue_title}&body={issue_body}&labels=dataset"
    )

    return (
        f"    .. admonition:: Found an issue with this dataset?\n"
        f"       :class: tip\n"
        f"\n"
        f"       If you encounter any problems with this dataset (missing files,\n"
        f"       incorrect metadata, loading errors, etc.), please let us know!\n"
        f"\n"
        f"       .. button-link:: {github_url}\n"
        f"          :color: primary\n"
        f"          :outline:\n"
        f"\n"
        f"          Report an Issue on GitHub"
    )


class MetaclassDataset(abc.ABCMeta):
    def __new__(cls, name, bases, attrs):
        doc = attrs.get("__doc__", "") or ""
        insert_blocks = []

        try:
            row = _summary_table.loc[name]
            row_str, row = format_row(row, horizontal=False)
            insert_blocks.append(row_str)
            attrs["_summary_table"] = row.to_dict()
        except KeyError:
            log.debug(
                f"No description found for dataset {name}. "
                f"Complete the appropriate moabb/datasets/summary_*.csv file"
            )

        metadata_sections = _metadata_doc_sections(attrs.get("METADATA"), doc)
        if metadata_sections:
            insert_blocks.append(metadata_sections)

        # Note: feedback "Report Issue" button is now part of the enhanced
        # dataset card header injected by dataset_timeline_ext.py, so the
        # standalone feedback admonition is no longer injected here.

        if insert_blocks:
            if doc.strip():
                doc_list = doc.split("\n\n")
                doc_list = [doc_list[0], *insert_blocks, *doc_list[1:]]
                attrs["__doc__"] = "\n\n".join(doc_list)
            else:
                attrs["__doc__"] = "\n\n".join(insert_blocks)

        return super().__new__(cls, name, bases, attrs)


class BaseDataset(metaclass=MetaclassDataset):
    """Abstract Moabb BaseDataset.

    Parameters required for all datasets.

    Parameters
    ----------
    subjects : list of int
        List of subject number (or tuple or numpy array).

    sessions_per_subject : int
        Number of sessions per subject (if varying, take minimum).

    events : dict of str
        String codes for events matched with labels in the stim channel.
        Currently imagery codes can include:
        ``left_hand``, ``right_hand``, ``hands``, ``feet``, ``rest``,
        ``left_hand_right_foot``, ``right_hand_left_foot``, ``tongue``,
        ``navigation``, ``subtraction``, ``word_ass`` (for word association).

    code : str
        Unique identifier for dataset, used in all plots.
        The code should be in CamelCase.

    interval : list
        Imagery interval as defined in the dataset description,
        with 2 entries.

    paradigm : str
        Defines what sort of dataset this is.
        One of ``'p300'``, ``'imagery'``, or ``'ssvep'``.

    doi : str, optional
        DOI for the dataset.

    return_all_modalities : bool | dict, optional
        Controls which channel types are retained when data is picked:

        - ``False`` (default): only EEG channels are kept.
        - ``True``: all channels except stim are kept.
        - ``dict``: keyword arguments forwarded to :func:`mne.pick_types`,
          e.g. ``dict(eeg=True, eog=True)`` keeps EEG and EOG channels.
          ``stim`` is always forced to ``False``.
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
        *,
        selected_subjects=None,
        selected_sessions=None,
        return_all_modalities=False,
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

        self.return_all_modalities = return_all_modalities

        self._all_subjects = list(subjects)
        if selected_subjects is not None:
            selected_subjects = list(selected_subjects)
            # Warn on duplicate subjects and deduplicate preserving order
            if len(selected_subjects) != len(set(selected_subjects)):
                unique = dict.fromkeys(selected_subjects)
                dupes = [s for s in unique if selected_subjects.count(s) > 1]
                warnings.warn(
                    f"Duplicate subjects detected: {dupes}. "
                    "Duplicates will be removed, preserving order.",
                    stacklevel=2,
                )
                selected_subjects = list(unique)
            invalid = [s for s in selected_subjects if s not in self._all_subjects]
            if invalid:
                raise ValueError(
                    f"Invalid subjects: {invalid}. "
                    f"Valid subjects are: {self._all_subjects}"
                )
            self.subject_list = selected_subjects
        else:
            self.subject_list = list(subjects)
        self.n_sessions = sessions_per_subject

        # Validate selected_sessions
        if selected_sessions is not None:
            try:
                selected_sessions = list(selected_sessions)
            except TypeError:
                raise TypeError(
                    f"selected_sessions must be an iterable, "
                    f"got {type(selected_sessions).__name__}"
                ) from None
            bad = [s for s in selected_sessions if not isinstance(s, (int, str))]
            if bad:
                raise TypeError(
                    f"selected_sessions elements must be int or str, "
                    f"got: {[(type(s).__name__, s) for s in bad]}"
                )
        self._selected_sessions = selected_sessions

        # Validate events dict integrity
        if not isinstance(events, dict):
            raise TypeError(f"events must be a dict, got {type(events).__name__}")
        for key, value in events.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"All event dict keys must be strings, but got "
                    f"{type(key).__name__}: {key!r}. {_KWARG_HINT}"
                )
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if not _is_event_int(v):
                        raise TypeError(
                            f"Event {key!r} list element {i} is {v!r} "
                            f"({type(v).__name__}), expected int. {_KWARG_HINT}"
                        )
            elif not _is_event_int(value):
                raise TypeError(
                    f"Event {key!r} has value {value!r} ({type(value).__name__}), "
                    f"expected int or list of int. {_KWARG_HINT}"
                )
        self.event_id = events
        self.code = code
        self.interval = interval
        self.paradigm = paradigm
        self.doi = doi
        self.unit_factor = unit_factor

    @property
    def all_subjects(self):
        """Full list of subjects available in this dataset (unfiltered)."""
        return list(self._all_subjects)

    @cached_property
    def metadata(self):
        """Return structured metadata for this dataset.

        Returns the DatasetMetadata object from the centralized catalog,
        or None if metadata is not available for this dataset.

        Returns
        -------
        ``DatasetMetadata`` | None
            The metadata object containing acquisition parameters,
            participant demographics, experiment details, and documentation.
            Returns None if no metadata is registered for this dataset.

        Examples
        --------
        >>> from moabb.datasets import BNCI2014_001
        >>> dataset = BNCI2014_001()
        >>> dataset.metadata.participants.n_subjects
        9
        >>> dataset.metadata.acquisition.sampling_rate
        250.0
        """
        from moabb.datasets.metadata import get_dataset_metadata

        try:
            return get_dataset_metadata(self.__class__.__name__)
        except KeyError:
            return None

    def _create_process_pipeline(self):
        return FixedPipeline(
            [(StepType.RAW, SetRawAnnotations(self.event_id, interval=self.interval))]
        )

    def _block_rep(self, block, repetition):
        raise NotImplementedError()

    def get_block_repetition(self, paradigm, subjects, block_list, repetition_list):
        """Select data for all provided subjects, blocks and repetitions.

        subject -> session -> run -> block -> repetition

        See Also
        --------
        get_data

        Parameters
        ----------
        subjects: List of int
            List of subject number
        block_list: List of int
            List of block number
        repetition_list: List of int
            List of repetition number inside a block

        Returns
        -------
        data: Dict
            dict containing the raw data
        """
        X, labels, meta = paradigm.get_data(self, subjects)
        X_select = []
        labels_select = []
        meta_select = []
        for block in block_list:
            for repetition in repetition_list:
                run = self._block_rep(block, repetition)
                X_select.append(X[meta["run"] == run])
                labels_select.append(labels[meta["run"] == run])
                meta_select.append(meta[meta["run"] == run])
        X_select = np.concatenate(X_select)
        labels_select = np.concatenate(labels_select)
        meta_select = np.concatenate(meta_select)
        df = pd.DataFrame(meta_select, columns=meta.columns)
        meta_select = df

        return X_select, labels_select, meta_select

    def get_data(self, subjects=None, cache_config=None, process_pipeline=None):
        """
        Return the data corresponding to a list of subjects.

        The returned data is a dictionary with the following structure::

            data = {"subject_id": {"session_id": {"run_id": run}}}

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
        cache_config: dict | :class:`~moabb.datasets.base.CacheConfig`
            Configuration for caching of datasets. See
            :class:`~moabb.datasets.base.CacheConfig` for details.
        process_pipeline: :class:`sklearn.pipeline.Pipeline` | None
            Optional processing pipeline to apply to the data.
            To generate an adequate pipeline, we recommend using
            :func:`moabb.make_process_pipelines`.
            This pipeline will receive :class:`mne.io.BaseRaw` objects.
            The steps names of this pipeline should be elements of
            ``StepType``.
            According to their name, the steps should either return a
            :class:`mne.io.BaseRaw`, a :class:`mne.Epochs`, or a
            :class:`numpy.ndarray`.
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

        effective_sessions = self._selected_sessions

        cache_config = CacheConfig.make(cache_config)

        if process_pipeline is None:
            process_pipeline = self._create_process_pipeline()

        if effective_sessions is not None:
            str_sessions = {str(s) for s in effective_sessions}
            pat = session_run_pattern()
        else:
            str_sessions = pat = None

        data = {}
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError("Invalid subject {:d} given".format(subject))
            subject_data = self._get_single_subject_data_using_cache(
                subject, cache_config, process_pipeline
            )
            if str_sessions is not None:
                subject_data = {
                    k: v
                    for k, v in subject_data.items()
                    if k in str_sessions
                    or ((m := re.fullmatch(pat, k)) and m.group(1) in str_sessions)
                }
            data[subject] = subject_data
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

    def convert_to_bids(
        self,
        path=None,
        subjects=None,
        overwrite=False,
        format="EDF",
        verbose=None,
        generate_figures=False,
    ):
        """Convert the dataset to BIDS format.

        Saves the raw EEG data in a BIDS-compliant directory structure.
        Unlike the caching mechanism (see :class:`~moabb.datasets.base.CacheConfig`), the files
        produced here do **not** contain a processing-pipeline hash
        (``desc-<hash>``) in their names, making the output a clean,
        shareable BIDS dataset.

        Parameters
        ----------
        path : str | :class:`~pathlib.Path` | None
            Directory under which the BIDS dataset will be written.
            If ``None`` the default MNE data directory is used (same default
            as the rest of MOABB).
        subjects : list of int | None
            Subject numbers to convert.  If ``None``, all subjects in
            ``subject_list`` are converted.
        overwrite : bool
            If ``True``, existing BIDS files for a subject are removed before
            saving.  Default is ``False``.
        format : str
            The file format for the raw EEG data.  Supported values are
            ``"EDF"`` (default), ``"BrainVision"``, and ``"EEGLAB"``.
        verbose : str | None
            Verbosity level forwarded to MNE/MNE-BIDS.
        generate_figures : bool
            If ``True``, generate interactive neural signature HTML figures
            in ``{bids_root}/derivatives/neural_signatures/``.  Requires
            ``plotly`` (``pip install moabb[interactive]``).  Default is
            ``False``.

        Returns
        -------
        bids_root : pathlib.Path
            Path to the root of the written BIDS dataset.

        Examples
        --------
        >>> from moabb.datasets import AlexMI
        >>> dataset = AlexMI()
        >>> bids_root = dataset.convert_to_bids(path="/tmp/bids", subjects=[1])

        Notes
        -----
        Use :class:`~moabb.datasets.base.CacheConfig` to configure caching
        for :meth:`get_data`. Use
        ``moabb.datasets.bids_interface.get_bids_root`` to get the BIDS root
        path.

        .. versionadded:: 1.5
        """
        if format not in _FORMAT_EXTENSION_MAP:
            raise ValueError(
                f"Unsupported format {format!r}. "
                f"Allowed formats are {tuple(_FORMAT_EXTENSION_MAP)}"
            )
        if subjects is None:
            subjects = self.subject_list

        invalid = [s for s in subjects if s not in self.subject_list]
        if invalid:
            raise ValueError(
                f"Invalid subject(s) {invalid}. Valid subjects are {self.subject_list}"
            )

        ext = _FORMAT_EXTENSION_MAP[format]

        for subject in subjects:
            interface = _BIDSInterfaceRawEDFNoDesc(
                dataset=self,
                subject=subject,
                path=path,
                process_pipeline=None,
                verbose=verbose,
                _format=format,
            )
            if overwrite:
                interface.erase()
            else:
                subject_dir = interface.root / f"sub-{subject}"
                if any(subject_dir.rglob(f"*{ext}")):
                    log.info(
                        "BIDS data already exists for %s, skipping "
                        "(use overwrite=True to overwrite).",
                        repr(interface),
                    )
                    continue
            sessions_data = self.get_data(subjects=[subject])
            interface.save(sessions_data[subject])

        bids_root = get_bids_root(self.code, path)

        if generate_figures:
            try:
                from moabb.analysis.neural_signatures import generate_neural_signature

                fig_dir = bids_root / "derivatives" / "neural_signatures"
                generate_neural_signature(self, subjects=subjects, output_dir=fig_dir)
            except ImportError:
                log.warning(
                    "plotly not installed, skipping figure generation. "
                    "Install with: pip install moabb[interactive]"
                )
            except (RuntimeError, ValueError, OSError) as e:
                log.warning("Neural signature generation failed: %s", e)

        return bids_root

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
                # Enrich raw.info from METADATA (sex, hand, age, line_freq)
                metadata = getattr(self, "METADATA", None)
                if metadata is not None:
                    for runs in sessions_data.values():
                        for raw in runs.values():
                            _enrich_raw_info_from_metadata(raw, metadata, subject)
            else:
                cache_type = cached_steps[-1][0]
                interface = _interface_map[cache_type](
                    self,
                    subject,
                    path=cache_config.path,
                    process_pipeline=FixedPipeline(cached_steps),
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
                        process_pipeline=FixedPipeline(
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

    def get_additional_metadata(self, subject: str, session: str, run: str):
        """
        Load additional metadata for a specific subject, session, and run.

        This method is intended to be overridden by subclasses to provide
        additional metadata specific to the dataset. The metadata is typically
        loaded from an `events.tsv` file or similar data source.

        Parameters
        ----------
        subject : str
            The identifier for the subject.
        session : str
            The identifier for the session.
        run : str
            The identifier for the run.

        Returns
        -------
        None | :class:`pandas.DataFrame`
            A DataFrame containing the additional metadata if available,
            otherwise None.
        """

        return None


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
        """Return the kwargs for the ``mne_bids.find_matching_paths`` function."""
        out = {"extensions": _RAW_EXTENSIONS}
        if subject is not None:
            out["subjects"] = str(subject)
        return out

    def _get_read_extra_params(
        self,
        subject: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any] | None:
        """Return the ``extra_params`` argument for the ``mne_bids.read_raw_bids`` function."""
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

    def get_additional_metadata(self, subject: str, session: str, run: str):
        """
        Load additional metadata for a specific subject, session, and run.

        Parameters
        ----------
        subject : str
            The identifier for the subject.
        session : str
            The identifier for the session.
        run : str
            The identifier for the run.

        Returns
        -------
        None | :class:`pandas.DataFrame`
            A DataFrame containing the additional metadata if available,
            otherwise None.
        """
        bids_paths = self.bids_paths(subject)

        # select only with matching session and run
        bids_path_selected = [
            pth
            for pth in bids_paths
            if f"ses-{session}" in pth.basename and f"run-{run}" in pth.basename
        ]

        if len(bids_path_selected) > 1:
            raise ValueError("More than one matching BIDS path found.")
        bids_path = bids_path_selected[0]

        events_fname = bids_path.find_matching_sidecar(
            suffix="events", extension=".tsv", on_error="warn"
        )
        if events_fname is None:
            return None

        # Use official mne-bids API — handles n/a filtering, stim_type compat, etc.
        annot_kwargs = events_file_to_annotation_kwargs(events_fname)

        # Build DataFrame from API output
        dm = pd.DataFrame(
            {
                "onset": annot_kwargs["onset"],
                "duration": annot_kwargs["duration"],
                "trial_type": annot_kwargs["description"],
            }
        )

        # Reconstruct 'value' from event_id mapping (description -> integer)
        dm["value"] = dm["trial_type"].map(annot_kwargs["event_id"])

        # Add extras (custom columns beyond standard BIDS columns)
        extras = annot_kwargs.get("extras")
        if extras and len(extras) > 0:
            extras_df = pd.DataFrame(extras)
            dm = pd.concat([dm, extras_df], axis=1)

        # Filter by dataset's event_id
        dm = dm[dm["trial_type"].isin(self.event_id.keys())]

        dm = dm.assign(subject=subject, session=session, run=run)
        return dm


class LocalBIDSDataset(BaseBIDSDataset):
    """Generic local/private BIDS datasets.

    This class is useful if you have a local/private dataset in BIDS format
    and you want to use it with MOABB, without having to create a new dataset class.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        Local path to the root of the BIDS dataset.
    path_search_params : dict[str, Any] | None
        Additional kwargs for the ``mne_bids.find_matching_paths`` function.
    read_extra_params : dict[str, Any] | None
        Additional kwargs for the ``mne_bids.read_raw_bids`` function.
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
        Factor to convert units to microvolts. Defaults to ``1e6``.
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
        return_all_modalities=False,
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
                subjects = sorted({path.subject for path in bids_paths})
                log.warning(f"Found subjects: {subjects}")
            if sessions_per_subject is None:
                sessions_per_subject = min(
                    len(
                        {
                            bids_path.session
                            for bids_path in bids_paths
                            if bids_path.subject == subject
                        }
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
            return_all_modalities=return_all_modalities,
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        return self.bids_root

    def _get_path_search_params(self, subject):
        return dict(
            super()._get_path_search_params(subject), **(self.path_search_params or {})
        )

    def _get_read_extra_params(self, subject):
        return self.read_extra_params
