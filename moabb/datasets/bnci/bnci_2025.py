"""BNCI 2025 datasets."""

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
from mne import Annotations
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FilterDetails,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PerformanceMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)

from .base import BNCIBaseDataset
from .utils import (
    BNCI_URL,
    bnci_data_path,
    convert_units,
    ensure_data_orientation,
    make_raw,
    validate_subject,
)


# =============================================================================
# BNCI 2025-001: Motor Kinematics Reaching
# =============================================================================

# Base URL for the BNCI 2025-001 dataset (hosted at TU Graz)
BNCI_2025_001_URL = "https://lampx.tugraz.at/~bci/database/001-2025/"

# Event code mapping for 001-2025 dataset
# Format: XYZ where X=speed (1=slow, 2=fast), Y=distance (1=near, 2=far), Z=direction (1-4)
# Direction codes: 1=up, 2=down, 3=left, 4=right
_EVENT_CODE_MAPPING_001_2025 = {
    "111": "up_slow_near",
    "112": "down_slow_near",
    "113": "left_slow_near",
    "114": "right_slow_near",
    "121": "up_slow_far",
    "122": "down_slow_far",
    "123": "left_slow_far",
    "124": "right_slow_far",
    "211": "up_fast_near",
    "212": "down_fast_near",
    "213": "left_fast_near",
    "214": "right_fast_near",
    "221": "up_fast_far",
    "222": "down_fast_far",
    "223": "left_fast_far",
    "224": "right_fast_far",
}


@verbose
def _load_data_001_2025(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_2025_001_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2025 dataset (Motor Kinematics Reaching).

    This dataset contains EEG data from 20 subjects performing discrete
    reaching movements with varying speed, distance, and direction parameters.
    The study investigates simultaneous encoding of multiple kinematic
    parameters during movement execution.

    Parameters
    ----------
    subject : int
        Subject number (1-20).
    path : None | str
        Location for data storage.
    force_update : bool
        Force update of the dataset.
    update_path : bool | None
        If True, set the data path in config.
    base_url : str
        Base URL for data download.
    only_filenames : bool
        If True, return only the local path of the files without loading.
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    sessions : dict
        Dictionary of sessions with raw data.

    Notes
    -----
    Dataset details:
    - 20 subjects (12 male, 8 female, mean age 26.1 +/- 4.1 years)
    - 60 EEG + 4 EOG channels = 64 total
    - Sampling rate: 500 Hz
    - 4 directions x 2 speeds x 2 distances = 16 conditions
    - ~60 trials per condition (~960 total per subject)

    The data is stored in EEGLAB format (.set/.fdt files) inside ZIP archives.
    Each subject's ZIP file contains:
    - {subject}v2-trialblocks.set/fdt: Main trial data with reaching movements
    - {subject}v2-eyeblocks.set/fdt: Eye movement calibration data
    """
    validate_subject(subject, 20, "BNCI2025-001")

    # Download the ZIP file for this subject
    zip_filename = f"p{subject:03d}.zip"
    url = f"{base_url}{zip_filename}"

    # Download the ZIP file
    zip_path = dl.data_dl(url, "BNCI", path, force_update, verbose)

    # Determine the extraction directory (same as ZIP location)
    zip_dir = Path(zip_path).parent
    extract_dir = zip_dir / f"p{subject:03d}"

    # Extract if not already extracted or if force_update
    # Try v2 naming first (subject 1 style: p001v2-trialblocks.set)
    set_file = extract_dir / f"p{subject:03d}v2-trialblocks.set"
    set_file_alt = extract_dir / f"p{subject:03d}-trialblocks.set"

    if (not set_file.exists() and not set_file_alt.exists()) or force_update:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Use the file that exists (v2 naming for subject 1, regular for others)
    if not set_file.exists():
        set_file = set_file_alt
    if not set_file.exists():
        raise FileNotFoundError(
            f"Could not find .set file in {extract_dir}. "
            f"Tried: p{subject:03d}v2-trialblocks.set and p{subject:03d}-trialblocks.set"
        )

    if only_filenames:
        return [str(set_file)]

    # Load the EEGLAB file
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=verbose)

    # Remap annotation descriptions from numeric codes to descriptive names
    # The data contains codes like "111", "112", etc. which we map to
    # descriptive names like "up_slow_near", "down_slow_near", etc.
    if raw.annotations is not None and len(raw.annotations) > 0:
        new_descriptions = []
        for desc in raw.annotations.description:
            if desc in _EVENT_CODE_MAPPING_001_2025:
                new_descriptions.append(_EVENT_CODE_MAPPING_001_2025[desc])
            else:
                new_descriptions.append(desc)
        raw.annotations.description = np.array(new_descriptions)

    # Set measurement date for BIDS compliance
    raw.set_meas_date(datetime(2024, 1, 1, tzinfo=timezone.utc))

    # Set line frequency (European dataset)
    raw.info["line_freq"] = 50.0

    # Set montage for standard 10-10 positions
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")

    # Return in MOABB session format
    # Session/run names must match pattern: digit(s) followed by optional letters
    sessions = {"0": {"0": raw}}

    return sessions


class BNCI2025_001(BNCIBaseDataset):
    """BNCI 2025-001 Motor Kinematics Reaching dataset.

    Dataset from Srisrisawang & Muller-Putz (2024) [1]_.

    **Dataset Description**

    This dataset investigates how the brain simultaneously encodes multiple
    kinematic parameters (speed, distance, and direction) during discrete
    reaching movements. Participants performed a four-direction center-out
    reaching task with varying speeds (quick/slow) and distances (near/far).

    The dataset provides insight into movement planning and execution
    processes as measured through EEG, enabling research on brain-computer
    interfaces for motor control and neurorehabilitation applications.

    **Participants**

    - 20 healthy subjects (12 male, 8 female)
    - Age: 26.1 +/- 4.1 years
    - Handedness: 17 right-handed, 3 left-handed (all used right hand)
    - Location: Institute of Neural Engineering, Graz University of
      Technology, Austria

    **Recording Details**

    - Equipment: BrainAmp (Brain Products GmbH)
    - Channels: 60 EEG + 4 EOG = 64 total channels
    - Sampling rate: 500 Hz
    - Reference: Common average reference (CAR) across 55 channels
    - EOG placement: Outer canthi, above/below left eye
    - Electrode positions: Measured with ultrasonic device (ELPOS, Zebris)

    **Experimental Procedure**

    - 4-direction center-out reaching task
    - 2 speed levels: slow, quick
    - 2 distance levels: near, far
    - 16 conditions total (4 directions x 2 speeds x 2 distances)
    - ~60 trials per condition (~960 total per subject)
    - Trial structure:
        - 1 s preparation period
        - Cue movement (0.4-2.4 s depending on condition)
        - >= 1 s waiting period
        - Movement execution
        - 1 s feedback display
        - 2 s intertrial interval

    **Event Codes**

    Events encode the combination of direction, speed, and distance:
    - up_slow_near (1), up_slow_far (2), up_fast_near (3), up_fast_far (4)
    - down_slow_near (5), down_slow_far (6), down_fast_near (7), down_fast_far (8)
    - left_slow_near (9), left_slow_far (10), left_fast_near (11), left_fast_far (12)
    - right_slow_near (13), right_slow_far (14), right_fast_near (15), right_fast_far (16)

    References
    ----------
    .. [1] Srisrisawang, N., & Muller-Putz, G. R. (2024). Simultaneous encoding
           of speed, distance, and direction in discrete reaching: an EEG study.
           Journal of Neural Engineering, 21(6).
           https://doi.org/10.1088/1741-2552/ada0ea

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is notable for its multi-parameter kinematic design,
    enabling study of how multiple movement parameters are represented
    simultaneously in EEG activity. The paradigm uses movement execution
    rather than motor imagery, making it complementary to MI datasets.

    The data is compatible with the MOABB motor imagery paradigm for
    processing purposes, though the underlying task is movement execution.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=60,
            channel_types={"eeg": 60},
            montage="af7 af3 afz af4 af8 f7 f5 f3 f1 fz f2 f4 f6 f8 ft7 fc5 fc3 fc1 fcz fc2 fc4 fc6 ft8 t7 c5 c3 c1 cz c2 c4 c6 t8 tp7 cp5 cp3 cp1 cpz cp2 cp4 cp6 tp8 p7 p5 p3 p1 pz p2 p4 p6 p8 ppo1h ppo2h po7 po3 poz po4 po8 o1 oz o2",
            hardware="BrainAmp",
            reference="right mastoid",
            software="EEGLAB",
            filters="50 Hz notch",
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["horizontal", "vertical"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=20,
            gender={"male": 12, "female": 8},
            age_mean=26.1,
            handedness={"right": 17, "left": 3},
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["rest", "right_hand"],
            trial_duration=5.0,
            study_design="Discrete reaching movements in four directions (up, down, left, right) with varying speeds (quick/slow) and distances (near/far) following visual cue, self-paced execution",
            feedback_type="visual (cue color: green for correct, red for incorrect direction)",
            stimulus_type="avatar",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2552/ada0ea",
            repository="GitHub",
            data_url="https://github.com/rkobler/eyeartifactcorrection",
            license="CC-BY-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw with eye artifact correction model available",
            preprocessing_applied=True,
            preprocessing_steps=["eye artifact correction"],
            filter_details=FilterDetails(
                highpass_hz=0.3,
                lowpass_hz=100.0,
                bandpass={"low_cutoff_hz": 0.3, "high_cutoff_hz": 80.0},
                notch_hz=[50],
                filter_type="Butterworth",
                filter_order=2,
            ),
            artifact_methods=["ICA"],
            re_reference="common average",
            downsampled_to_hz=200,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "Shrinkage LDA"],
            feature_extraction=["Bandpower", "Covariance/Riemannian", "ICA"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="stratified k-fold",
            evaluation_type=["cross_session", "transfer_learning"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=55.9,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["smart_home", "vr_ar"],
        ),
        data_structure=DataStructureMetadata(
            n_trials=30,
            trials_context="per_class",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 21)),
            sessions_per_subject=1,
            events={
                "up_slow_near": 1,
                "up_slow_far": 2,
                "up_fast_near": 3,
                "up_fast_far": 4,
                "down_slow_near": 5,
                "down_slow_far": 6,
                "down_fast_near": 7,
                "down_fast_far": 8,
                "left_slow_near": 9,
                "left_slow_far": 10,
                "left_fast_near": 11,
                "left_fast_far": 12,
                "right_slow_near": 13,
                "right_slow_far": 14,
                "right_fast_near": 15,
                "right_fast_far": 16,
            },
            code="BNCI2025-001",
            interval=[0, 4],  # Movement period
            paradigm="imagery",  # Compatible with motor imagery paradigm
            doi="10.1088/1741-2552/ada0ea",
            load_fn=_load_data_001_2025,
            base_url=BNCI_2025_001_URL,
        )


# =============================================================================
# BNCI 2025-002: Continuous 2D Trajectory Decoding
# =============================================================================

EVENT_ID_002 = {"snakerun": 1, "freerun": 2, "eyerun": 3}
_EVENT_ALIASES_002 = {
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
        if key in EVENT_ID_002:
            return key
        for token, name in _EVENT_ALIASES_002.items():
            if token in key:
                return name
        return None
    try:
        code = int(label)
    except (TypeError, ValueError):
        return None
    return {v: k for k, v in EVENT_ID_002.items()}.get(code)


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


# Marker codes for 002-2025 dataset
# These files contain snake run trials with perception feedback
_MARKER_CODE_002 = {
    1000: "snakerun",  # Trial start - snake tracking task
}


def _extract_annotations_from_markers_array(markers, sfreq):
    """Extract annotations from a continuous MARKERS array.

    The MARKERS array contains event codes at each sample. Events are detected
    by finding rising edges (transitions from 0 or negative to positive values).

    Parameters
    ----------
    markers : ndarray
        1D array of marker codes, one per sample.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    annotations : Annotations | None
        MNE Annotations object with detected events, or None if no events found.
    """
    markers = np.asarray(markers).squeeze()
    if markers.ndim != 1 or markers.size == 0:
        return None

    # Find rising edges: transitions from 0 (or negative) to positive marker codes
    # We're interested in trial starts marked by code 1000
    onsets = []
    descriptions = []

    prev_marker = markers[0]
    for i in range(1, len(markers)):
        curr_marker = markers[i]
        # Detect rising edge to a known marker code
        if curr_marker != prev_marker and curr_marker in _MARKER_CODE_002:
            onsets.append(i / sfreq)
            descriptions.append(_MARKER_CODE_002[curr_marker])
        prev_marker = curr_marker

    if not onsets:
        return None

    return Annotations(
        onset=onsets, duration=[0.0] * len(onsets), description=descriptions
    )


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
        Subject number (1-2). Note: Only 2 subjects are currently available
        on the BNCI server.
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
    validate_subject(subject, 10, "BNCI2025-002")

    # Subject IDs available on the BNCI server
    # Note: 10 of the original 20 subjects' data is currently available
    subject_ids = ["fe3", "fe4", "fe5", "fe6", "fe7", "fe8", "fg1", "fg2", "fg3", "fg4"]

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

    # Try to extract annotations from the MARKERS array (002-2025 format)
    annotations = None
    if "MARKERS" in data:
        annotations = _extract_annotations_from_markers_array(data["MARKERS"], sfreq)

    # Fallback to standard extraction methods
    if annotations is None:
        annotations = _extract_annotations(data, sfreq, eeg_data.shape[1])

    if annotations is not None:
        raw.set_annotations(annotations)

    return raw


class BNCI2025_002(BNCIBaseDataset):
    """BNCI 2025-002 Continuous 2D Trajectory Decoding dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from participants performing a
    continuous 2D trajectory decoding task using attempted movement. The study
    investigates continuous decoding of hand movement trajectories from EEG
    signals, with participants tracking a moving target on screen while their
    dominant arm is strapped to restrict actual motor output (simulating
    attempted movement conditions similar to paralyzed individuals).

    The experimental paradigm includes both calibration and online decoding
    phases, with varying levels of EEG feedback (0%, 50%, 100%) to evaluate
    the impact of feedback on decoding performance.

    Note: Only 2 of the original 20 participants' data is currently available
    on the BNCI server.

    **Participants**

    - 10 able-bodied subjects (5 male, 5 female)
    - Mean age 24 +/- 5 years, all right-handed
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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=200.0,
            n_channels=60,
            channel_types={"eeg": 60},
            montage="channels_fyrxyz",
            hardware="BrainVision",
            sensor_type="active electrodes (actiCAP)",
            reference="Car",
            software="EEGLAB",
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=4,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"male": 5, "female": 5},
            age_mean=24,
            handedness="right-handed (Edinburgh Handedness Inventory)",
            bci_experience="naive BCI users in terms of motor decoding (4 had previous EEG experience)",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_arm", "feet"],
            trial_duration=23.0,
            study_design="Participants attempt movement while viewing trajectory feedback that is a mixture of decoded brain signals and predefined target trajectory",
            feedback_type="none",
            stimulus_type="avatar",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2552/ac689f",
            repository="GitHub",
            data_url="https://github.com/sccn/labstreaminglayer",
            funding=["European Research Council"],
            license="CC-BY-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="continuous signals with synchronized decoded control signals, paradigm targets, visual feedback trajectories, and error metrics",
            preprocessing_applied=True,
            preprocessing_steps=[
                "resampling and alignment to EEG timeline",
                "GND channel removal",
                "channel location assignment",
            ],
            filter_details=FilterDetails(
                highpass_hz=0.18,
                lowpass_hz=3,
                notch_hz=50,
            ),
            artifact_methods=["ICA"],
            re_reference="common average",
            downsampled_to_hz=20,
        ),
        signal_processing=SignalProcessingMetadata(
            feature_extraction=["ERD", "ERS", "Covariance/Riemannian", "ICA"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session", "transfer_learning"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["prosthetic", "robotic_arm", "smart_home", "vr_ar"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials={
                "eyeruns": 38,
                "calibration_snakeruns": 48,
                "50_percent_feedback_snakeruns": 36,
            },
        ),
        file_format="MAT",
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=3,
            events=EVENT_ID_002,
            code="BNCI2025-002",
            interval=[0, 8],  # Trial length varies but 8s is a common window
            paradigm="imagery",
            doi="10.1088/1741-2552/ac689f",
            load_fn=_load_data_002_2025,
            base_url=BNCI_URL,
        )
