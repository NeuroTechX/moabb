"""BNCI 2025-002 Continuous 2D Trajectory Decoding dataset."""

from datetime import datetime, timezone

import numpy as np
from mne import Annotations
from mne.utils import verbose
from scipy.io import loadmat

from .base import BNCIBaseDataset
from .utils import (
    BNCI_URL,
    bnci_data_path,
    convert_units,
    ensure_data_orientation,
    make_raw,
    validate_subject,
)


EVENT_ID = {"snakerun": 1, "freerun": 2, "eyerun": 3}
_EVENT_ALIASES = {
    "snake": "snakerun",
    "snakerun": "snakerun",
    "free": "freerun",
    "freerun": "freerun",
    "eye": "eyerun",
    "eyerun": "eyerun",
}


def _get_field(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return None


def _normalize_positions(pos, sfreq, n_samples):
    pos = np.asarray(pos).squeeze()
    if pos.size == 0:
        return None
    pos = pos.astype(float)
    if pos.max() <= (n_samples / sfreq + 1):
        pos = np.round(pos * sfreq)
    pos = pos.astype(int)
    if pos.min() >= 1:
        pos = pos - 1
    pos = pos[(pos >= 0) & (pos < n_samples)]
    return pos


def _label_to_desc(label):
    if label is None:
        return None
    if isinstance(label, bytes):
        label = label.decode()
    if isinstance(label, str):
        key = label.strip().lower()
        if key in EVENT_ID:
            return key
        for token, name in _EVENT_ALIASES.items():
            if token in key:
                return name
        return None
    try:
        code = int(label)
    except (TypeError, ValueError):
        return None
    return {v: k for k, v in EVENT_ID.items()}.get(code)


def _annotations_from_candidate(candidate, sfreq, n_samples):
    pos = _get_field(candidate, "pos")
    labels = None
    class_names = _get_field(candidate, "className")

    if pos is not None:
        labels = (
            _get_field(candidate, "y")
            or _get_field(candidate, "label")
            or _get_field(candidate, "type")
        )
    elif isinstance(candidate, np.ndarray) and candidate.ndim == 2:
        if candidate.shape[1] < 2:
            return None
        pos = candidate[:, 0]
        labels = candidate[:, 1]
    else:
        return None

    pos = _normalize_positions(pos, sfreq, n_samples)
    if pos is None or labels is None:
        return None

    labels = np.asarray(labels).squeeze()
    if labels.ndim == 2:
        if labels.shape[0] == len(pos):
            labels = labels.argmax(axis=1)
        elif labels.shape[1] == len(pos):
            labels = labels.argmax(axis=0)
        else:
            return None
    elif labels.ndim != 1 or labels.shape[0] != len(pos):
        return None

    descriptions = []
    if class_names is not None:
        class_names = [str(name) for name in np.atleast_1d(class_names).tolist()]
        labels = labels.astype(int)
        if labels.min() == 1 and labels.max() <= len(class_names):
            labels = labels - 1
        for idx in labels:
            if 0 <= idx < len(class_names):
                descriptions.append(_label_to_desc(class_names[idx]))
            else:
                descriptions.append(None)
    else:
        descriptions = [_label_to_desc(label) for label in labels]

    pairs = [(p, d) for p, d in zip(pos, descriptions) if d is not None]
    if not pairs:
        return None

    onset = [p / sfreq for p, _ in pairs]
    desc = [d for _, d in pairs]
    return Annotations(onset=onset, duration=[0.0] * len(desc), description=desc)


def _extract_annotations(mat_data, sfreq, n_samples):
    containers = [mat_data, _get_field(mat_data, "data")]
    for container in containers:
        if container is None:
            continue
        for key in ("mrk", "markers", "marker", "events", "event"):
            candidate = _get_field(container, key)
            annotations = _annotations_from_candidate(candidate, sfreq, n_samples)
            if annotations is not None:
                return annotations
    return None


@verbose
def _load_data_002_2025(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 002-2025 dataset (Continuous 2D Trajectory Decoding).

    Parameters
    ----------
    subject : int
        Subject number (1-20).
    path : None | str
        Location of where to look for the BNCI data storing location.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI_PATH in mne-python config.
    base_url : str
        Base URL for the dataset.
    only_filenames : bool
        If True, return only the local path of the files.
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    sessions : dict
        Dictionary containing sessions with raw data for each run.
    """
    validate_subject(subject, 20, "BNCI2025-002")

    # Subject IDs in the dataset (as listed on BNCI website)
    # fmt: off
    subject_ids = [
        "fe3", "fg4", "fg6", "fh0", "fh4", "fh5", "fh7", "fi3", "fi6", "fj0",
        "fj4", "fj5", "fj7", "fk3", "fk6", "fl0", "fl3", "fl4", "fl6", "fm0",
    ]
    # fmt: on

    subj_id = subject_ids[subject - 1]

    # 60 EEG channels following the 10-10 system
    # Fp1, Fp2, FT9, FT10 were reallocated as EOG (VEOG1, VEOG2, HEOG1, HEOG2)
    # TP9, TP10 were relocated to PPO1h, PPO2h
    # fmt: off
    ch_names_eeg = [
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POz", "PO4", "PO8",
        "O1", "Oz", "O2",
        "PPO1h", "PPO2h",
    ]
    # fmt: on

    ch_names_eog = ["VEOG1", "VEOG2", "HEOG1", "HEOG2"]
    ch_names = ch_names_eeg + ch_names_eog
    ch_types = ["eeg"] * 60 + ["eog"] * 4

    sessions = {}
    filenames = []

    # 3 sessions, each with 3 perception levels
    perception_levels = ["perc0", "perc50", "perc100"]

    for session_idx in range(1, 4):
        session_runs = {}
        for run_idx, perc in enumerate(perception_levels):
            filename_part = f"{subj_id}_ses{session_idx}_{perc}.mat"
            url = f"{base_url}002-2025/{filename_part}"
            filename = bnci_data_path(url, path, force_update, update_path)[0]
            filenames.append(filename)

            if only_filenames:
                continue

            # Load the MAT file
            raw = _convert_run_002_2025(
                filename,
                ch_names,
                ch_types,
                subject_id=subject,
                session_idx=session_idx,
                perception=perc,
            )
            session_runs[f"{run_idx}{perc}"] = raw

        if not only_filenames:
            sessions[f"{session_idx - 1}ses{session_idx}"] = session_runs

    if only_filenames:
        return filenames

    return sessions


@verbose
def _convert_run_002_2025(
    filename,
    ch_names,
    ch_types,
    subject_id=None,
    session_idx=None,
    perception=None,
    verbose=None,
):
    """Convert one run from 002-2025 dataset to MNE Raw object.

    Parameters
    ----------
    filename : str
        Path to MAT file.
    ch_names : list
        Channel names.
    ch_types : list
        Channel types.
    subject_id : int
        Subject number.
    session_idx : int
        Session index.
    perception : str
        Perception level (perc0, perc50, perc100).
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    raw : mne.io.RawArray
        MNE Raw object containing the EEG data.
    """
    # Load the MAT file
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    # The dataset uses LSL format, with EEG, EOG, and movement data
    # Extract EEG data - try different possible keys
    eeg_data = None
    sfreq = 200.0  # Default sampling rate for this dataset

    if "cnt" in data:
        eeg_data = data["cnt"]
        if "fs" in data:
            sfreq = float(data["fs"])
    elif "X" in data:
        eeg_data = data["X"]
        if "fs" in data:
            sfreq = float(data["fs"])
    elif "data" in data:
        run_data = data["data"]
        if hasattr(run_data, "X"):
            eeg_data = run_data.X
            if hasattr(run_data, "fs"):
                sfreq = float(run_data.fs)
        else:
            eeg_data = run_data
    else:
        # Try to find the data in the structure
        for key in data.keys():
            if not key.startswith("__"):
                val = data[key]
                if hasattr(val, "X"):
                    eeg_data = val.X
                    if hasattr(val, "fs"):
                        sfreq = float(val.fs)
                    break
                elif isinstance(val, np.ndarray) and val.ndim == 2:
                    if val.shape[0] >= 60 or val.shape[1] >= 60:
                        eeg_data = val
                        break

    if eeg_data is None:
        raise ValueError(
            f"Could not find EEG data in MAT file. Keys: {list(data.keys())}"
        )

    # Ensure data is in correct shape (n_channels, n_samples)
    eeg_data = ensure_data_orientation(eeg_data, n_channels=64)

    # Convert to Volts (MNE standard) if data is in microvolts
    if np.abs(eeg_data).max() > 1:  # Likely in microvolts
        eeg_data = convert_units(eeg_data, from_unit="uV", to_unit="V")

    # Check number of channels
    n_channels_data = eeg_data.shape[0]
    n_channels_expected = len(ch_names)

    if n_channels_data != n_channels_expected:
        # If we have more channels, they might include trajectory/marker data
        # Take only the first 64 channels (60 EEG + 4 EOG)
        if n_channels_data > n_channels_expected:
            eeg_data = eeg_data[:n_channels_expected, :]
        else:
            # If we have fewer channels, adjust channel names
            ch_names = ch_names[:n_channels_data]
            ch_types = ch_types[:n_channels_data]

    raw = make_raw(
        eeg_data,
        ch_names,
        ch_types,
        sfreq,
        verbose=verbose,
        montage="standard_1005",
        line_freq=50.0,
        meas_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
        description=f"Session {session_idx}, Perception: {perception}",
    )

    annotations = _extract_annotations(data, sfreq, eeg_data.shape[1])
    if annotations is not None:
        raw.set_annotations(annotations)

    return raw


class BNCI2025_002(BNCIBaseDataset):
    """BNCI 2025-002 Continuous 2D Trajectory Decoding dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 20 able-bodied participants
    performing a continuous 2D trajectory decoding task using attempted
    movement. The study investigates continuous decoding of hand movement
    trajectories from EEG signals, with participants tracking a moving target
    on screen while their dominant arm is strapped to restrict actual motor
    output (simulating attempted movement conditions similar to paralyzed
    individuals).

    The experimental paradigm includes both calibration and online decoding
    phases, with varying levels of EEG feedback (0%, 50%, 100%) to evaluate
    the impact of feedback on decoding performance.

    **Participants**

    - 20 able-bodied subjects (10 male, mean age 24 +/- 5 years)
    - All right-handed
    - 4 had prior EEG experience
    - Location: Institute of Neural Engineering, Graz University of
      Technology, Austria

    **Recording Details**

    - Equipment: 64-channel actiCAP system (Brain Products GmbH)
    - Channels: 60 EEG + 4 EOG electrodes
    - Original sampling rate: 200 Hz
    - Electrode positions: 10-10 system with modifications
      (Fp1, Fp2, FT9, FT10 used as EOG; TP9, TP10 relocated to PPO1h, PPO2h)
    - Reference: Common average
    - Data synchronized using Lab Streaming Layer (LSL)

    **Experimental Procedure**

    Each session consists of:

    - Calibration phase: 2 eye runs (38 trials, 8s each) + 4 snake runs
      (48 trials, 23s each)
    - Online phase with 3 perception conditions:
      - perc0: No EEG feedback (baseline)
      - perc50: 50% EEG feedback
      - perc100: 100% EEG feedback

    Trial types:

    - Snake runs: Tracking a moving white target with decorrelated x/y
      coordinates
    - Free runs: Tracing static shapes (diagonal/circle) at self-paced speed

    **Data Organization**

    - 3 sessions per subject (recorded over 5 days)
    - 3 perception levels per session (perc0, perc50, perc100)
    - Files named: {subject_id}_ses{session}_perc{level}.mat

    References
    ----------
    .. [1] Kobler, R. J., Almeida, I., Sburlea, A. I., & Muller-Putz, G. R.
           (2022). Continuous 2D trajectory decoding from attempted movement:
           across-session performance in able-bodied and feasibility in a
           spinal cord injured participant. Journal of Neural Engineering,
           19(3), 036005. https://doi.org/10.1088/1741-2552/ac689f

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is designed for continuous decoding research, specifically
    for predicting 2D hand movement trajectories from EEG. Unlike
    classification-based motor imagery datasets, this dataset contains
    continuous trajectory labels suitable for regression-based decoders.

    The paradigm "imagery" is used for compatibility with MOABB's motor
    imagery processing pipelines, though the actual task involves attempted
    (rather than imagined) movements.

    See Also
    --------
    BNCI2014_001 : 4-class motor imagery dataset
    BNCI2014_004 : 2-class motor imagery dataset
    """

    _participant_demographics = {
        "n_subjects": 20,
        "gender": {"male": 10, "female": 10},
        "age_mean": 24,
        "age_std": 5,
        "handedness": "all right-handed",
        "health_status": "able-bodied participants",
        "bci_experience": "4 had prior EEG experience",
        "location": "Institute of Neural Engineering, Graz University of Technology, Austria",
    }

    ARTICLE_METADATA = {
        "n_subjects": 20,
        "sessions_per_subject": 3,
        "sampling_rate": 200,
        "n_channels": 64,
        "channel_types": {"eeg": 60, "eog": 4},
        "paradigm": "imagery",
        "events": EVENT_ID,
        "doi": "10.1088/1741-2552/ac689f",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "reference": "common average",
        "filters": "0.18-3 Hz (final processing)",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/002-2025/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 21)),
            sessions_per_subject=3,
            events=EVENT_ID,
            code="BNCI2025-002",
            interval=[0, 8],  # Trial length varies but 8s is a common window
            paradigm="imagery",
            doi="10.1088/1741-2552/ac689f",
            load_fn=_load_data_002_2025,
            base_url=BNCI_URL,
        )
