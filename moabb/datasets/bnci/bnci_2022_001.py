"""BNCI 2022-001 EEG Correlates of Difficulty Level dataset."""

from datetime import datetime, timezone

import numpy as np
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FilterDetails,
    FrequencyBands,
    ParticipantMetadata,
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


# File naming convention for 001-2022 dataset:
# - Baseline files: s{n}b.mat (e.g., s1b.mat, s2b.mat, ...)
# - Task (wpsize) files: s{n}w.mat (e.g., s1w.mat, s2w.mat, ...)
# The dataset contains 13 subjects (s1-s13)


@verbose
def _load_data_001_2022(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2022 dataset (EEG Correlates of Difficulty Level).

    This dataset contains EEG recordings from subjects piloting a simulated drone
    through waypoints at varying difficulty levels. The study aimed to decode
    subjective perception of task difficulty from EEG signals.

    Parameters
    ----------
    subject : int
        Subject number (1-13).
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

    Notes
    -----
    The dataset provides two types of recordings per subject:
    - Baseline: 1-minute eye close/open recording
    - Task (wpsize): Drone piloting task with varying difficulty levels

    The public release contains only the first session (offline) data,
    downsampled from 2048 Hz to 256 Hz. Online sessions and behavioral
    data are not included.
    """
    validate_subject(subject, 13, "BNCI2022-001")

    # 64 EEG channels using the Biosemi ActiveTwo system (10-20 system positions)
    # fmt: off
    ch_names_eeg = [
        "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
        "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
        "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
        "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
        "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
        "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
    ]
    # fmt: on

    # 3 EOG channels (as described in the dataset documentation)
    # First channel: below the outer canthus of the right eye
    # Second channel: between eyebrows
    # Third channel: below the outer canthus of the left eye
    ch_names_eog = ["EOG1", "EOG2", "EOG3"]

    ch_names = ch_names_eeg + ch_names_eog
    ch_types = ["eeg"] * 64 + ["eog"] * 3

    sessions = {}
    filenames = []

    # Load task (wpsize) data - this is the main task data with difficulty levels
    # File naming pattern: s{n}w.mat (e.g., s1w.mat, s2w.mat, ...)
    task_filename = f"s{subject}w.mat"
    url = f"{base_url}001-2022/{task_filename}"

    filename = bnci_data_path(url, path, force_update, update_path)[0]
    filenames.append(filename)

    if not only_filenames:
        raw = _convert_run_001_2022(
            filename,
            ch_names,
            ch_types,
            subject_id=subject,
            run_type="task",
            verbose=verbose,
        )
        sessions["0task"] = {"0": raw}

    if only_filenames:
        return filenames

    return sessions


@verbose
def _convert_run_001_2022(
    filename, ch_names, ch_types, subject_id=None, run_type="task", verbose=None
):
    """Convert one run from 001-2022 dataset to MNE Raw object.

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
    run_type : str
        Type of recording ('task' or 'baseline').
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    raw : mne.io.RawArray
        MNE Raw object containing the EEG data with annotations for events.
    """
    # Load the MAT file
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    # According to the dataset description, the structure contains:
    # - data.EEG: timepoint-by-channel matrix
    # - data.EOG: timepoint-by-channel matrix
    # - data.Trigger: event triggers
    # - data.Header: sampling rate info
    # - data.Channel_labels: electrode names
    # - data.subjective_report: difficulty ratings (for task files only)

    # Try to extract data from different possible structures
    if "data" in data:
        run_data = data["data"]
    else:
        # Try to find the data structure
        run_data = data

    # Extract EEG and EOG data
    eeg_data = None
    eog_data = None
    trigger = None
    sfreq = 256.0  # Default downsampled rate as per description

    # Handle the data structure
    if hasattr(run_data, "EEG"):
        eeg_data = run_data.EEG
    elif "EEG" in data:
        eeg_data = data["EEG"]

    if hasattr(run_data, "EOG"):
        eog_data = run_data.EOG
    elif "EOG" in data:
        eog_data = data["EOG"]

    if hasattr(run_data, "Trigger"):
        trigger = run_data.Trigger
    elif "Trigger" in data:
        trigger = data["Trigger"]

    # Try to get sampling rate from header
    if hasattr(run_data, "Header"):
        header = run_data.Header
        if hasattr(header, "fs_resample"):
            sfreq = float(header.fs_resample)
        elif hasattr(header, "fs"):
            sfreq = float(header.fs)
    elif "Header" in data:
        header = data["Header"]
        if hasattr(header, "fs_resample"):
            sfreq = float(header.fs_resample)
        elif hasattr(header, "fs"):
            sfreq = float(header.fs)

    if eeg_data is None:
        raise ValueError(
            f"Could not find EEG data in MAT file. Keys: {list(data.keys())}"
        )

    # Ensure data is in correct orientation (n_channels, n_samples)
    if eeg_data.ndim == 2:
        eeg_data = ensure_data_orientation(eeg_data, n_channels=64)

    # Handle EOG data
    if eog_data is not None:
        if eog_data.ndim == 2:
            eog_data = ensure_data_orientation(eog_data, n_channels=3)

        # Combine EEG and EOG
        combined_data = np.vstack([eeg_data, eog_data])
    else:
        # If no separate EOG, the EEG array might contain all 67 channels
        if eeg_data.shape[0] >= 67:
            combined_data = eeg_data[:67, :]
        else:
            combined_data = eeg_data
            # Adjust channel names and types if needed
            ch_names = ch_names[: combined_data.shape[0]]
            ch_types = ch_types[: combined_data.shape[0]]

    # Convert to Volts (MNE standard) if data is in microvolts
    # Biosemi data is typically in microvolts
    if np.abs(combined_data).max() > 1:  # Likely in microvolts
        combined_data = convert_units(combined_data, from_unit="uV", to_unit="V")

    # Create MNE info structure
    n_channels = combined_data.shape[0]
    if n_channels != len(ch_names):
        # Adjust channel names if needed
        if n_channels < len(ch_names):
            ch_names = ch_names[:n_channels]
            ch_types = ch_types[:n_channels]
        else:
            # Add generic channel names for extra channels
            for i in range(len(ch_names), n_channels):
                ch_names.append(f"MISC{i - 67 + 1}")
                ch_types.append("misc")

    raw = make_raw(
        combined_data,
        ch_names,
        ch_types,
        sfreq,
        verbose=verbose,
        montage="biosemi64",
        line_freq=50.0,
        meas_date=datetime(2016, 10, 1, tzinfo=timezone.utc),
    )

    # Add events as annotations if trigger channel exists
    if trigger is not None:
        from mne import Annotations

        # Ensure trigger is 1D
        if trigger.ndim > 1:
            trigger = trigger.flatten()

        # Find event positions and types
        # Event codes according to description:
        # 1: begin of trajectory (countdown before drone moves)
        # 16: waypoint miss (first 16 indicates drone starts moving)
        # 48: waypoint hit
        # 255: end of trajectory (3 seconds after final waypoint)

        event_mapping = {
            1: "trajectory_start",
            16: "waypoint_miss",
            48: "waypoint_hit",
            255: "trajectory_end",
        }

        # Find non-zero trigger positions
        event_indices = np.where(trigger != 0)[0]
        if len(event_indices) > 0:
            event_times = event_indices / sfreq
            event_values = trigger[event_indices].astype(int)

            # Create annotations
            onset = []
            duration = []
            description = []

            for t, v in zip(event_times, event_values):
                if v in event_mapping:
                    onset.append(t)
                    duration.append(0.0)
                    description.append(event_mapping[v])

            if onset:
                annotations = Annotations(
                    onset=onset, duration=duration, description=description
                )
                raw.set_annotations(annotations)

    # Add description
    desc = f"Subject {subject_id}, Run type: {run_type}"
    raw.info["description"] = desc

    return raw


class BNCI2022_001(BNCIBaseDataset):
    """BNCI 2022-001 EEG Correlates of Difficulty Level dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 13 subjects performing a simulated
    drone piloting task through waypoints at varying difficulty levels. The study
    aimed to decode the subjective perception of task difficulty from EEG signals
    to help optimize operator performance by automatically adjusting task difficulty.

    Subjects controlled a simulated drone through circular waypoints using a
    flight joystick. The difficulty was modulated by the size of waypoints -
    smaller waypoints required more precise control and were perceived as more
    difficult. After each trajectory, subjects reported their perceived
    difficulty level.

    **Participants**

    - 13 healthy subjects (8 females, mean age 22.6 years, SD 1.04)
    - All had normal or corrected-to-normal vision
    - No history of motor or neurological disease
    - Location: EPFL, Geneva, Switzerland

    **Recording Details**

    - Equipment: Biosemi ActiveTwo system
    - Channels: 64 EEG + 3 EOG = 67 total
    - Original sampling rate: 2048 Hz (downsampled to 256 Hz in public release)
    - Hardware trigger recorded as 8-bit signal
    - Baseline recording: 1-minute eye close/open

    **Experimental Procedure**

    - Subjects sat in front of a monitor controlling a flight joystick with
      their right hand
    - Task: Pilot simulated drone through circular waypoints
    - 32 trajectories per subject, each with 32 waypoints (~90 seconds each)
    - 16 difficulty levels (waypoint sizes), normalized to each subject's skill
    - Difficulty progression: levels 16->1->16 (decreasing then increasing)
    - After each trajectory, subjects reported:
        - Numeric difficulty level (0-100)
        - Categorical difficulty (easy/hard/extremely hard)

    **Event Codes**

    - trajectory_start (1): Beginning of trajectory (countdown before drone moves)
    - waypoint_miss (16): Drone failed to pass through waypoint
    - waypoint_hit (48): Drone successfully passed through waypoint
    - trajectory_end (255): End of trajectory (3s after final waypoint)

    **Data Organization**

    - 1 session per subject (offline data only, online sessions not included)
    - Two file types per subject:
        - Baseline: eye close/open recording
        - Task (wpsize): main piloting task with difficulty variations

    References
    ----------
    .. [1] Jao, P.-K., Chavarriaga, R., & Millan, J. d. R. (2021). EEG Correlates
           of Difficulty Levels in Dynamical Transitions of Simulated Flying and
           Mapping Tasks. IEEE Transactions on Human-Machine Systems, 51(2), 99-108.
           https://doi.org/10.1109/THMS.2020.3038339

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is designed for cognitive workload assessment and difficulty
    level detection. Unlike motor imagery datasets, the task involves actual
    motor control while the cognitive state (perceived difficulty) varies.

    The public release contains only the first session (offline) data. Additional
    behavioral data and online sessions with closed-loop difficulty adaptation
    are not included. The paradigm "imagery" is used for compatibility, though
    the actual task involves motor execution with cognitive load variations.

    See Also
    --------
    BNCI2015_004 : Multi-class mental task dataset with imagery and cognitive tasks
    BNCI2014_001 : 4-class motor imagery dataset

    Examples
    --------
    >>> from moabb.datasets import BNCI2022_001
    >>> dataset = BNCI2022_001()
    >>> dataset.subject_list
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-10",
            hardware="Biosemi ActiveTwo",
            sensor_type="active",
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
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=3,
                eog_type=["horizontal", "vertical"],
                other_physiological=["ppg"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=13,
            health_status="normal or corrected-to-normal vision, no history of motor or neurological disease (one subject with history of vasovagal syncope)",
            gender={"female": 8, "male": 5},
            age_mean=22.6,
            handedness="12 right-handed, 1 left-handed",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "left_hand", "feet"],
            trial_duration=90,
            study_design="Subjects piloted a simulated drone through circular waypoints using a flight joystick, controlling roll and pitch while the drone maintained constant velocity",
            feedback_type="none",
            stimulus_type="avatar",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TAFFC.2021.3059688",
            associated_paper_doi="10.1109/THMS.2020.3038339",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Other"],
            modality=["Other"],
            type=["Clinical/Intervention"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="downsampled raw",
            preprocessing_applied=True,
            preprocessing_steps=["downsampling from 2048 Hz to 256 Hz"],
            filter_details=FilterDetails(
                highpass_hz=1,
                lowpass_hz=40,
                bandpass=[1, 40],
                filter_type="FIR",
                filter_order=14,
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=256,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["ERS", "PSD", "ICA"],
            frequency_bands=FrequencyBands(
                analyzed_range=[2.0, 28.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_subject", "cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["drone", "gaming", "vr_ar", "communication"],
            environment="outdoor",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 14)),
            sessions_per_subject=1,
            events={
                "trajectory_start": 1,
                "waypoint_miss": 16,
                "waypoint_hit": 48,
                "trajectory_end": 255,
            },
            code="BNCI2022-001",
            interval=[0, 90],  # Approximately 90 seconds per trajectory
            paradigm="imagery",  # For compatibility
            doi="10.1109/THMS.2020.3038339",
            load_fn=_load_data_001_2022,
            base_url=BNCI_URL,
        )
