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
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
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
            channel_types={"eeg": 64, "eog": 3},
            montage="10-10",
            hardware="Biosemi ActiveTwo",
            sensor_type="active",
            reference="car",
            software="EEGLAB",
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "AFz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPz",
                "Cz",
                "EOG1",
                "EOG2",
                "EOG3",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FT7",
                "FT8",
                "Fp1",
                "Fp2",
                "Fpz",
                "Fz",
                "Iz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P10",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP7",
                "TP8",
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
            age_std=1.04,
            handedness={"right": 12, "left": 1},
        ),
        experiment=ExperimentMetadata(
            events={
                "trajectory_start": 1,
                "waypoint_miss": 16,
                "waypoint_hit": 48,
                "trajectory_end": 255,
            },
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "left_hand", "feet"],
            trial_duration=90.0,
            study_design="Subjects piloted a simulated drone through circular waypoints using a flight joystick, controlling roll and pitch while the drone maintained constant velocity. In offline session: 32 trajectories each with constant difficulty level (v-shape design from level 16 to 1 and back to 16), each trajectory had 32 waypoints and lasted ~90 seconds. In online sessions: each condition consisted of 12 trajectories with 33 waypoints and 8 decision points per trajectory.",
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
            synchronicity="cue-based",
            instructions="Subjects piloted a simulated drone through a series of circular waypoints. Subjects controlled the roll and pitch while the drone had a constant velocity of 11.8 arbitrary units per second when flying straight. They were instructed to press a button when the current level was easy as a way to collect ground truth for decoding or to proceed with self-paced learning.",
            stimulus_presentation={
                "screen_size": "twenty-inch screen",
                "screen_resolution": "1680x1050",
                "input_device": "Logitech Extreme 3D Pro joystick",
                "waypoint_colors": "green (current), blue (next), yellow (decision point)",
                "waypoint_distance_pitch": "32 A.U. (at least 2.7 seconds)",
                "waypoint_distance_roll": "24 A.U. (at least 2.0 seconds)",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TAFFC.2021.3059688",
            investigators=["Ping-Keng Jao", "Ricardo Chavarriaga", "Jose del R. Millan"],
            institution="Ecole Polytechnique Federale de Lausanne",
            country="Switzerland",
            publication_year=2021,
            senior_author="Jose del R. Millan",
            contact_info=[
                "ping-keng.jao@alumni.epfl.ch",
                "ricardo.chavarriaga@zhaw.ch",
                "jose.millan@austin.utexas.edu",
            ],
            institution_address="1015 Geneva, Switzerland",
            funding=["Swiss National Centres of Competence in Research (NCCR) Robotics"],
            acknowledgements="The authors would like to thank Alexander Cherpillod for his help in the implementation of the simulator and Ruslan Aydarkhanov for his suggestions in designing the protocol. Some figures were drawn with the Gramm MATLAB toolbox.",
            keywords=[
                "EEG",
                "real-time decoding of difficulty",
                "closed-loop adaptation",
                "(simulated) flying",
                "workload",
                "challenge point",
                "brain-machine interface",
            ],
            associated_paper_doi="10.1109/THMS.2020.3038339",
            license="CC-BY-4.0",
            repository="BNCI Horizon",
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        sessions=["offline", "online_session_2", "online_session_3"],
        data_processed=True,
        file_format="gdf",
        tags=Tags(
            pathology=["Healthy"],
            modality=["EEG"],
            type=["Experimental/Research"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling from 2048 Hz to 256 Hz",
                "casual bandpass filtering between 1 and 40 Hz",
                "SPHARA 20th order spatial low-pass filter for interpolation and artifact reduction",
                "common-average re-referencing",
                "ICA for EOG artifact removal",
                "peripheral electrodes removed (25 central channels kept)",
                "artifact rejection: windows with peak value > 50 µV rejected",
            ],
            highpass_hz=1.0,
            lowpass_hz=40.0,
            bandpass=[1.0, 40.0],
            filter_type="Butterworth",
            filter_order=14,
            artifact_methods=["ICA", "SPHARA", "amplitude thresholding"],
            re_reference="car",
            downsampled_to_hz=256.0,
            notes="Out of 39 recordings, P2 was removed twice from offline or online sessions due to short-circuit with the CMS or DRL electrode. On average, 15.8 ICA components were returned and 1.07 components were removed during construction of online decoders (correlation > 0.7 with EOG).",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "LDA",
                "Generalized Linear Model with elastic net regularization",
            ],
            feature_extraction=["PSD", "ICA", "log-PSD"],
            frequency_bands={
                "analyzed_range": [2.0, 28.0],
                "theta": [4.0, 8.0],
                "alpha": [10.5, 13.0],
            },
            spatial_filters=["SPHARA", "common-average reference"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-pair-out cross-validation (4x or 64x depending on class balance)",
            cv_folds=4,
            evaluation_type=["within_subject", "cross_session"],
        ),
        performance={
            "accuracy_percent": 76.7,
            "offline_validation_accuracy_mean": 76.7,
            "offline_validation_accuracy_std": 5.1,
            "online_session_2_accuracy_mean": 56.2,
            "online_session_2_accuracy_std": 8.6,
            "online_session_3_accuracy_mean": 54.7,
            "online_session_3_accuracy_std": 11.0,
            "online_above_chance_recordings": "16 out of 26 (~62%)",
        },
        bci_application=BCIApplicationMetadata(
            applications=[
                "drone control",
                "adaptive learning",
                "difficulty regulation",
                "visuomotor learning",
            ],
            environment="indoor laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["right_hand", "left_hand", "feet"],
        ),
        data_structure=DataStructureMetadata(
            n_trials={
                "offline_session": "32 trajectories of 32 waypoints each (~90 seconds per trajectory)",
                "online_session_per_condition": "12 trajectories of 33 waypoints each with 8 decision points",
            },
            trials_context="Offline session: v-shape difficulty design (level 16→1→16). Online sessions: each condition had 12 trajectories, starting at level 1 for 1st trajectory, then 4 levels lower than final level of previous trajectory. Average 10.3 seconds per decision group (4 waypoints).",
            n_blocks=2,
        ),
        abstract="Adaptively increasing the difficulty level in learning was shown to be beneficial than increasing the level after some fixed time intervals. To efficiently adapt the level, we aimed at decoding the subjective difficulty level based on Electroencephalography (EEG) signals. We designed a visuomotor learning task that one needed to pilot a simulated drone through a series of waypoints of different sizes, to investigate the effectiveness of the EEG decoder. The EEG decoder was compared with another condition that the subjects decided when to increase the difficulty level. We examined the decoding performance together with behavioral outcomes. The online accuracies were higher than the chance level for 16 out of 26 cases, and the behavioral results, such as task scores, skill curves, and learning patterns, of EEG condition were similar to the condition based on manual regulation of difficulty.",
        methodology="The study compared two conditions for difficulty regulation during a simulated drone piloting task: (1) EEG-based automatic difficulty adjustment using real-time decoding of perceived difficulty, and (2) Manual self-paced adjustment where subjects pressed a button when they found the level easy. Each subject participated in one offline session (for building subject-specific decoders) and two online sessions (each containing both EEG and Manual conditions in counterbalanced order). The task involved piloting a drone through circular waypoints with 16 difficulty levels defined by waypoint radius. Features were extracted using Thomson's multitaper algorithm with 2-second sliding windows, and classification used generalized linear models with elastic net regularization followed by LDA. The study evaluated both decoding accuracy and behavioral outcomes (task scores, skill curves, learning patterns).",
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
