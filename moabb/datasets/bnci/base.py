"""Shared helpers for legacy BNCI datasets (2003-2019)."""

import io
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from mne import Annotations, create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

from .utils import convert_units


BNCI_URL = "https://lampx.tugraz.at/~bci/database/"
BBCI_URL = "http://doc.ml.tu-berlin.de/bbci/"


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


@verbose
def load_data(
    subject,
    dataset="BNCI2014-001",
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):  # noqa: D301
    """Get paths to local copies of a BNCI dataset files.

    This will fetch data for a given BNCI dataset. Report to the bnci website
    for a complete description of the experimental setup of each dataset.

    Parameters
    ----------
    subject : int
        The subject to load.
    dataset : string
        The bnci dataset name.
    path : None | str
        Location of where to look for the BNCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BNCI_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the BNCI dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI_PATH in mne-python
        config to the given path. If None, the user is prompted.
    only_filenames : bool
        If True, return only the local path of the files without
        loading the data.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raws : list
        List of raw instances for each non consecutive recording. Depending
        on the dataset it could be a BCI run or a different recording session.
    event_id: dict
        dictionary containing events and their code.
    """
    from .bnci_2003 import _load_data_iva_2003
    from .bnci_2014 import (
        _load_data_001_2014,
        _load_data_002_2014,
        _load_data_004_2014,
        _load_data_008_2014,
        _load_data_009_2014,
    )
    from .bnci_2015 import (
        _load_data_001_2015,
        _load_data_003_2015,
        _load_data_004_2015,
        _load_data_006_2015,
        _load_data_007_2015,
        _load_data_008_2015,
        _load_data_009_2015,
        _load_data_010_2015,
        _load_data_012_2015,
        _load_data_013_2015,
    )

    dataset_list = {
        "BNCI2003-004": _load_data_iva_2003,
        "BNCI2014-001": _load_data_001_2014,
        "BNCI2014-002": _load_data_002_2014,
        "BNCI2014-004": _load_data_004_2014,
        "BNCI2014-008": _load_data_008_2014,
        "BNCI2014-009": _load_data_009_2014,
        "BNCI2015-001": _load_data_001_2015,
        "BNCI2015-003": _load_data_003_2015,
        "BNCI2015-004": _load_data_004_2015,
        "BNCI2015-006": _load_data_006_2015,
        "BNCI2015-007": _load_data_007_2015,
        "BNCI2015-008": _load_data_008_2015,
        "BNCI2015-009": _load_data_009_2015,
        "BNCI2015-010": _load_data_010_2015,
        "BNCI2015-012": _load_data_012_2015,
        "BNCI2015-013": _load_data_013_2015,
    }

    baseurl_list = {
        "BNCI2003-004": "https://www.bbci.de/competition/",
        "BNCI2014-001": BNCI_URL,
        "BNCI2014-002": BNCI_URL,
        "BNCI2015-001": BNCI_URL,
        "BNCI2014-004": BNCI_URL,
        "BNCI2014-008": BNCI_URL,
        "BNCI2014-009": BNCI_URL,
        "BNCI2015-003": BNCI_URL,
        "BNCI2015-004": BNCI_URL,
        "BNCI2015-006": BBCI_URL,
        "BNCI2015-007": BBCI_URL,
        "BNCI2015-008": BBCI_URL,
        "BNCI2015-009": BBCI_URL,
        "BNCI2015-010": BBCI_URL,
        "BNCI2015-012": BBCI_URL,
        "BNCI2015-013": BNCI_URL,
    }

    if dataset not in dataset_list.keys():
        raise ValueError(
            "Dataset '%s' is not a valid BNCI dataset ID. "
            "Valid dataset are %s." % (dataset, ", ".join(dataset_list.keys()))
        )

    return dataset_list[dataset](
        subject,
        path,
        force_update,
        update_path,
        baseurl_list[dataset],
        only_filenames,
        verbose,
    )


# ----------------------------------------------------------------------------
# Shared conversion utilities
# ----------------------------------------------------------------------------


def _finalize_raw(raw, dataset_code, subject_id):
    """Finalize raw object with montage, measurement date, and subject ID.

    This function should be called by each conversion function after creating
    the raw object to ensure all required metadata is set for BIDS compliance.

    Parameters
    ----------
    raw : instance of RawArray
        Raw object to finalize.
    dataset_code : str
        Dataset code (e.g., 'BNCI2014-001').
    subject_id : int
        Subject number.
    """
    # Set montage if not already set and we have standard EEG channels
    if raw.get_montage() is None:
        eeg_picks = [
            ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == "eeg"
        ]

        if eeg_picks:
            montage = make_standard_montage("standard_1005")
            if any(ch in montage.ch_names for ch in eeg_picks):
                raw.set_montage(montage, on_missing="ignore")

    # Set measurement date if not already set (required for BIDS)
    if raw.info["meas_date"] is None:
        year = MNEBNCI._dataset_years.get(dataset_code, 2010)
        raw.set_meas_date(datetime(year, 1, 1, tzinfo=timezone.utc))

    # Ensure subject_info has an ID (required for BIDS)
    subject_info = raw.info.get("subject_info")
    if subject_info is not None:
        if "his_id" not in str(subject_info):
            subject_info["his_id"] = f"sub-{subject_id:02d}"

    # Ensure description is set consistently (required for Raw concatenation)
    # If description is None, set to empty string to allow MNE to merge Raws
    if raw.info.get("description") is None:
        raw.info["description"] = ""


def _extract_common_demographics(run, rec_year):
    """Extracts common demographic information from a run object."""
    subject_info = {}

    # Extract age
    age = int(run.age)
    if rec_year:
        birth_year = rec_year - age
        subject_info["birthday"] = date(birth_year, 1, 1)

    # Extract gender
    gender_str = str(run.gender).lower()
    if gender_str in ["male", "m"]:
        subject_info["sex"] = 1
    elif gender_str in ["female", "f"]:
        subject_info["sex"] = 2
    else:
        subject_info["sex"] = 0  # Unknown

    return subject_info


def _enrich_run_with_metadata(raw, run, dataset_code, subject_id):
    """Extract metadata from run object and enrich raw object.

    This function extracts subject-specific metadata (age, gender, etc.) from
    MAT file run objects and enriches the raw object. It also sets the
    measurement date based on when the data was collected. After metadata
    extraction, it calls _finalize_raw() to ensure BIDS compliance.

    Parameters
    ----------
    raw : instance of RawArray
        Raw object to enrich.
    run : MAT file run object
        Run object containing metadata.
    dataset_code : str
        Dataset code to determine which fields to extract.
    subject_id : int
        Subject number for BIDS subject_info.
    """
    rec_year = MNEBNCI._dataset_years.get(dataset_code)

    # BNCI2014-001 and BNCI2014-004: have age, gender, artifacts
    if dataset_code in ["BNCI2014-001", "BNCI2014-004"]:
        if rec_year:
            raw.set_meas_date(datetime(rec_year, 1, 1, tzinfo=timezone.utc))

        subject_info = _extract_common_demographics(run, rec_year)
        subject_info["hand"] = 1  # right-handed
        raw.info["subject_info"] = subject_info

        # Extract artifacts information
        artifacts = run.artifacts
        if len(artifacts) > 0:
            n_artifacts = len(np.nonzero(artifacts)[0])
            if n_artifacts > 0:
                current_desc = raw.info.get("description") or ""
                raw.info["description"] = (
                    current_desc + f"Artifacts: {n_artifacts}/{len(artifacts)} trials; "
                )

    # BNCI2014-008: has age (string!), gender, ALSfrs, onsetALS
    elif dataset_code == "BNCI2014-008":
        if rec_year:
            raw.set_meas_date(datetime(rec_year, 1, 1, tzinfo=timezone.utc))

        subject_info = _extract_common_demographics(run, rec_year)
        subject_info["hand"] = 0  # Unknown
        raw.info["subject_info"] = subject_info

        # Extract ALS-specific information
        alsfrs = str(run.ALSfrs)
        onset = str(run.onsetALS)
        current_desc = raw.info.get("description") or ""
        # ALS = Amyotrophic Lateral Sclerosis (medical condition, not a typo)
        raw.info["description"] = current_desc + f"ALSfrs: {alsfrs}; ALS onset: {onset}; "

    # Finalize raw object (montage, measurement date fallback, subject ID)
    _finalize_raw(raw, dataset_code, subject_id)


def _convert_mi(filename, ch_names, ch_types, dataset_code=None, subject_id=None):
    """Process (Graz) motor imagery data from MAT files.

    Parameters
    ----------
    filename : str
        Path to the MAT file.
    ch_names : list of str
        List of channel names.
    ch_types : list of str
        List of channel types.
    dataset_code : str, optional
        Dataset code for metadata extraction.
    subject_id : int, optional
        Subject number for BIDS compliance.

    Returns
    -------
    raw : instance of RawArray
        returns list of recording runs."""
    runs = []
    event_id = {}
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    if isinstance(data["data"], np.ndarray):
        run_array = data["data"]
    else:
        run_array = [data["data"]]

    for run in run_array:
        raw, evd = _convert_run(run, ch_names, ch_types, None)
        if raw is None:
            continue

        # Enrich with metadata if dataset code is provided
        if dataset_code and subject_id:
            _enrich_run_with_metadata(raw, run, dataset_code, subject_id)
        elif subject_id:
            # No dataset-specific metadata, but still need BIDS finalization
            _finalize_raw(raw, dataset_code or "unknown", subject_id)

        runs.append(raw)
        event_id.update(evd)
    # change labels to match rest
    standardize_keys(event_id)
    return runs, event_id


def standardize_keys(d):
    master_list = [
        ["both feet", "feet"],
        ["left hand", "left_hand"],
        ["right hand", "right_hand"],
        ["FEET", "feet"],
        ["HAND", "right_hand"],
        ["NAV", "navigation"],
        ["SUB", "subtraction"],
        ["WORD", "word_ass"],
    ]
    for old, new in master_list:
        if old in d.keys():
            d[new] = d.pop(old)


@verbose
def _convert_run(run, ch_names=None, ch_types=None, verbose=None):
    """Convert one run to raw."""

    # parse eeg data
    event_id = {}
    n_chan = run.X.shape[1]
    montage = make_standard_montage("standard_1005")
    eeg_data = convert_units(run.X, from_unit="uV", to_unit="V")
    sfreq = run.fs

    if not ch_names:
        ch_names = ["EEG%d" % ch for ch in range(1, n_chan + 1)]
        montage = None  # no montage

    if not ch_types:
        ch_types = ["eeg"] * n_chan

    trigger = np.zeros((len(eeg_data), 1))
    # some runs does not contains trials i.e baseline runs
    if len(run.trial) > 0:
        trigger[run.trial - 1, 0] = run.y
    else:
        return None, None

    eeg_data = np.c_[eeg_data, trigger]
    # Use 'STI' instead of 'stim' to avoid ambiguity with channel type
    ch_names = ch_names + ["STI"]
    ch_types = ch_types + ["stim"]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)

    # Set line frequency (50 Hz for European datasets)
    raw.info["line_freq"] = 50.0

    return raw, event_id


@verbose
def _convert_run_p300_sl(run, verbose=None):
    """Convert one p300 run from santa lucia file format."""

    montage = make_standard_montage("standard_1005")
    eeg_data = convert_units(run.X, from_unit="uV", to_unit="V")
    sfreq = 256
    ch_names = list(run.channels) + ["Target stim", "Flash stim"]
    ch_types = ["eeg"] * len(run.channels) + ["stim"] * 2

    flash_stim = run.y_stim
    flash_stim[flash_stim > 0] += 2
    eeg_data = np.c_[eeg_data, run.y, flash_stim]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    event_id.update({ev: (ii + 3) for ii, ev in enumerate(run.classes_stim)})
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    # Montage is now set in _get_single_subject_data() method
    raw.set_montage(montage)

    # Set line frequency (50 Hz for European datasets)
    raw.info["line_freq"] = 50.0

    return raw, event_id


@verbose
def _convert_bbci(filename, ch_types, verbose=None):
    """Convert one file in bbci format."""
    raws = []
    event_id = {}

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    for run in data["data"]:
        raw, evd = _convert_run_bbci(run, ch_types, verbose)
        raws.append(raw)
        event_id.update(evd)

    return raws, event_id


@verbose
def _convert_run_bbci(run, ch_types=None, verbose=None):
    """Convert one run to raw."""

    # parse eeg data
    eeg_data = convert_units(run.X, from_unit="uV", to_unit="V")
    sfreq = run.fs

    ch_names = list(run.channels)

    # Dynamically determine channel types if not provided
    if ch_types is None:
        ch_types = ["eeg"] * len(ch_names)

    trigger = np.zeros((len(eeg_data), 1))
    trigger[run.trial - 1, 0] = run.y
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}

    flash = np.zeros((len(eeg_data), 1))
    flash[run.trial - 1, 0] = run.y_stim + 2
    ev_fl = {"Stim%d" % (stim): (stim + 2) for stim in np.unique(run.y_stim)}
    event_id.update(ev_fl)

    eeg_data = np.c_[eeg_data, trigger, flash]
    ch_names = ch_names + ["Target", "Flash"]
    ch_types = ch_types + ["stim"] * 2

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)

    # Set line frequency (50 Hz for European datasets)
    raw.info["line_freq"] = 50.0

    return raw, event_id


def _convert_bbci2003(filename, ch_names, ch_type):
    """
    Process motor imagery data from MAT files.

     Parameters
     ----------
        filename (str):
            Path to the MAT file.
        ch_names (list of str):
            List of channel names.
        ch_type (list of str):
            List of channel types.

    Returns
    -------
        raw (instance of RawArray):
            returns MNE Raw object.
    """
    zip_path = Path(filename)

    with zipfile.ZipFile(zip_path, "r") as z:
        mat_files = [f for f in z.namelist() if f.endswith(".mat")]

        if not mat_files:
            raise FileNotFoundError("No .mat file found in zip archive.")

        with z.open(mat_files[0]) as f:
            data = loadmat(io.BytesIO(f.read()))

        run = data
        raw, ev = _convert_run_bbci2003(run, ch_names, ch_type)
        return raw, ev


@verbose
def _convert_run_bbci2003(run, ch_names, ch_types, verbose=None):
    """
    Converts one run to a raw MNE object.

    Parameters
    ----------
        run (ndarray):
            The continuous EEG signal.
        ch_names (list of str):
            List of channel names.
        ch_types (list of str):
            List of channel types.
        verbose (bool, str, int, or None):
            If not None, override default verbose level (see :func:`mne.verbose`
            and :ref:`Logging documentation <tut_logging>` for more).

    Returns:
        raw (instance of RawArray):
            MNE Raw object.
        event_id (dict):
            Dictionary containing class names.
    """
    class_map = {
        "right": "right_hand",
        "foot": "feet",
    }

    raw_labels = run["mrk"]["y"][0, 0][0]
    labels_mask = ~np.isnan(raw_labels)
    valid_labels = raw_labels[labels_mask]
    labels = valid_labels.astype(int) - 1

    raw_positions = run["mrk"][0][0]["pos"][0]
    positions = raw_positions[labels_mask]

    sfreq = float(run["nfo"][0, 0]["fs"][0, 0])
    eeg_data = run["cnt"]
    raw_classes = run["mrk"]["className"]

    while isinstance(raw_classes, (list, np.ndarray)) and len(raw_classes) == 1:
        raw_classes = raw_classes[0]
    class_names = [cls[0] for cls in raw_classes]

    for i, word in enumerate(class_names):
        if word in class_map:
            class_names[i] = class_map[word]

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    onset = positions / sfreq
    duration = 0
    description = [class_names[i] for i in labels]
    annotations = Annotations(onset=onset, duration=duration, description=description)

    event_id = {name: i for i, name in enumerate(class_names)}
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_annotations(annotations)

    # Enrich raw object with additional metadata
    raw.info["line_freq"] = 50.0

    return raw, event_id


@verbose
def _convert_run_epfl(run, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}

    eeg_data = convert_units(run.eeg, from_unit="uV", to_unit="V")
    sfreq = run.header.SampleRate

    ch_names = list(run.header.Label[:-1])
    ch_types = ["eeg"] * len(ch_names)

    trigger = np.zeros((len(eeg_data), 1))

    for ii, typ in enumerate(run.header.EVENT.TYP):
        if typ in [6, 9]:  # Error
            trigger[run.header.EVENT.POS[ii] - 1, 0] = 2
        elif typ in [5, 10]:  # correct
            trigger[run.header.EVENT.POS[ii] - 1, 0] = 1

    eeg_data = np.c_[eeg_data, trigger]
    # Use 'STI' instead of 'stim' to avoid ambiguity with channel type
    ch_names = ch_names + ["STI"]
    ch_types = ch_types + ["stim"]
    event_id = {"correct": 1, "error": 2}

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)

    # Enrich raw object with additional metadata
    raw.info["line_freq"] = 50.0

    return raw, event_id


class MNEBNCI(BaseDataset):
    """Base class for BNCI Horizon 2020 datasets.

    The BNCI Horizon 2020 project (Brain/Neural Computer Interaction Horizon 2020)
    was an EU-funded initiative to advance brain-computer interface research. This
    base class provides common functionality for loading and processing datasets
    from the BNCI database.

    **Data Sources**

    - Primary: http://bnci-horizon-2020.eu/database/data-sets/
    - Secondary: http://doc.ml.tu-berlin.de/bbci/ (BBCI archive for some datasets)

    **Common Features**

    All BNCI datasets share these characteristics:

    - Automatic electrode montage setting (standard_1005)
    - BIDS-compliant metadata (measurement date, subject info)
    - Consistent event naming conventions (Target/NonTarget for P300, class names for MI)
    - Line frequency set to 50 Hz (European recordings)
    - Data stored in MAT format (MATLAB)

    **Supported Paradigms**

    BNCI datasets cover multiple BCI paradigms:

    - Motor Imagery (MI): BNCI2003_004, BNCI2014_001, BNCI2014_002, BNCI2014_004,
      BNCI2015_001, BNCI2015_004, BNCI2019_001
    - P300/ERP: BNCI2014_008, BNCI2014_009, BNCI2015_003, BNCI2015_006,
      BNCI2015_007, BNCI2015_008, BNCI2015_009, BNCI2015_010, BNCI2015_012
    - Error-Related Potentials: BNCI2015_013

    Attributes
    ----------
    _dataset_years : dict
        Mapping from dataset codes to recording years, used for setting
        measurement dates for BIDS compliance.

    See Also
    --------
    BNCI2014_001 : 4-class motor imagery (BCI Competition IV Dataset 2a)
    BNCI2014_008 : P300 speller for ALS patients
    BNCI2015_004 : 5-class mental tasks for users with disability
    BNCI2015_009 : AMUSE auditory spatial P300 paradigm
    BNCI2015_010 : RSVP gaze-independent visual P300
    """

    # Dataset collection/publication years for measurement dates
    _dataset_years = {
        "BNCI2003-004": 2003,
        "BNCI2014-001": 2008,
        "BNCI2014-002": 2008,
        "BNCI2014-004": 2008,
        "BNCI2014-008": 2012,
        "BNCI2014-009": 2012,
        "BNCI2015-001": 2010,
        "BNCI2015-003": 2009,
        "BNCI2015-004": 2013,
        "BNCI2015-007": 2012,  # Motion VEP Speller dataset (Schaeff2012)
        "BNCI2015-009": 2015,  # AMUSE dataset from BNCI Horizon 2020
        "BNCI2015-010": 2015,  # RSVP dataset from BNCI Horizon 2020
        "BNCI2015-012": 2015,  # PASS2D dataset from BNCI Horizon 2020
        "BNCI2015-013": 2015,  # Error-related potentials dataset
    }

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        sessions = load_data(subject=subject, dataset=self.code, verbose=False)
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return load_data(
            subject=subject,
            dataset=self.code,
            verbose=verbose,
            update_path=update_path,
            path=path,
            force_update=force_update,
            only_filenames=True,
        )


class BNCIBaseDataset(BaseDataset):
    """Base dataset with shared BNCI loading behavior."""

    def __init__(self, *, load_fn, base_url=BNCI_URL, **kwargs):
        self._load_fn = load_fn
        self._base_url = base_url
        super().__init__(**kwargs)

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return self._load_fn(
            subject=subject,
            path=None,
            force_update=False,
            update_path=None,
            base_url=self._base_url,
            only_filenames=False,
            verbose=False,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return paths to data files for a single subject."""
        return self._load_fn(
            subject=subject,
            path=path,
            force_update=force_update,
            update_path=update_path,
            base_url=self._base_url,
            only_filenames=True,
            verbose=verbose,
        )
