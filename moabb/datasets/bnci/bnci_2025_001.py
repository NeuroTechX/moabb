"""BNCI 2025-001 Motor Kinematics Reaching dataset.

This module implements the BNCI2025_001 dataset for MOABB.
"""

import numpy as np
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


@verbose
def _load_data_001_2025(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
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
    """
    validate_subject(subject, 20, "BNCI2025-001")

    # Download the data file for this subject
    url = "{u}001-2025/sub{s:02d}.mat".format(u=base_url, s=subject)
    filename = bnci_data_path(url, path, force_update, update_path)[0]

    if only_filenames:
        return [filename]

    # Load the MAT file
    mat_data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Expected structure based on BNCI convention:
    # data.X - EEG data (samples x channels)
    # data.y - labels
    # data.trial - trial onset indices
    # data.fs - sampling frequency
    # data.classes - class names

    data = mat_data["data"]

    # Get sampling frequency
    sfreq = float(data.fs) if hasattr(data, "fs") else 500.0

    # Get channel names - 60 EEG + 4 EOG
    if hasattr(data, "channels"):
        ch_names = list(data.channels)
    else:
        # Default channel names based on paper description
        # 60 EEG channels in standard 10-10 montage + 4 EOG
        # fmt: off
        ch_names = [
            "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
            "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
            "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
            "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
            "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
            "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
            "EOG1", "EOG2", "EOG3", "EOG4",
        ]
        # fmt: on

    # Set channel types
    ch_types = ["eeg"] * 60 + ["eog"] * 4

    # Get EEG data and ensure correct orientation
    eeg_data = ensure_data_orientation(data.X, n_channels=64)

    n_channels_data = eeg_data.shape[0]
    if n_channels_data != len(ch_names):
        if n_channels_data > len(ch_names):
            eeg_data = eeg_data[: len(ch_names), :]
        else:
            ch_names = ch_names[:n_channels_data]
            ch_types = ch_types[:n_channels_data]

    # Convert to volts
    eeg_data = convert_units(eeg_data, from_unit="uV", to_unit="V")

    # Get trial information
    trial_onsets = data.trial - 1  # Convert to 0-indexed
    labels = data.y

    # Create trigger channel
    trigger = np.zeros((1, eeg_data.shape[1]))
    for onset, label in zip(trial_onsets, labels):
        if onset < trigger.shape[1]:
            trigger[0, onset] = label

    # Add trigger channel
    all_data = np.vstack([eeg_data, trigger])
    ch_names = ch_names + ["STI"]
    ch_types = ch_types + ["stim"]

    raw = make_raw(
        all_data,
        ch_names,
        ch_types,
        sfreq,
        verbose=verbose,
        montage="standard_1005",
        line_freq=50.0,
    )

    # Return in standard session format
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

    _participant_demographics = {
        "n_subjects": 20,
        "gender": {"male": 12, "female": 8},
        "age_mean": 26.1,
        "age_std": 4.1,
        "handedness": {"right": 17, "left": 3},
        "health_status": "healthy subjects",
        "location": "Institute of Neural Engineering, Graz University of Technology, Austria",
    }

    ARTICLE_METADATA = {
        "n_subjects": 20,
        "sessions_per_subject": 1,
        "sampling_rate": 500,
        "n_channels": 64,
        "channel_types": {"eeg": 60, "eog": 4},
        "reference": "common average",
        "paradigm": "imagery",  # Compatible paradigm for MOABB processing
        "task_type": "movement_execution",
        "doi": "10.1088/1741-2552/ada0ea",
        "license": "Open access (BNCI Horizon 2020)",
        "montage": "standard_1005",
        "filters": "0.3-100 Hz bandpass, 50 Hz notch",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/",
    }

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
            base_url=BNCI_URL,
        )
