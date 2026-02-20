"""BNCI 2024-001 Handwritten Character Classification dataset implementation.

This module provides the implementation for the BNCI 2024-001 dataset,
which contains EEG data from handwritten character classification tasks.

To integrate into bnci.py:
1. Add to dataset_list: "BNCI2024-001": _load_data_001_2024
2. Add to baseurl_list: "BNCI2024-001": BNCI_URL
3. Add to _dataset_years: "BNCI2024-001": 2024
4. Copy _load_data_001_2024 and _convert_run_001_2024 functions
5. Copy BNCI2024_001 class
6. Add BNCI2024_001 to __init__.py exports
"""

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
    FilterDetails,
    FrequencyBands,
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
    make_raw,
    validate_subject,
)


# Mapping of letter markers to event codes
# The MAT file uses marker = 100 + (letter position in alphabet)
# a=0, d=3, e=4, f=5, j=9, n=13, o=14, s=18, t=19, v=21
_LETTER_MARKER_MAP = {
    100: 1,  # letter 'a' (alphabet position 0) -> event 1
    103: 2,  # letter 'd' (alphabet position 3) -> event 2
    104: 3,  # letter 'e' (alphabet position 4) -> event 3
    105: 4,  # letter 'f' (alphabet position 5) -> event 4
    109: 5,  # letter 'j' (alphabet position 9) -> event 5
    113: 6,  # letter 'n' (alphabet position 13) -> event 6
    114: 7,  # letter 'o' (alphabet position 14) -> event 7
    118: 8,  # letter 's' (alphabet position 18) -> event 8
    119: 9,  # letter 't' (alphabet position 19) -> event 9
    121: 10,  # letter 'v' (alphabet position 21) -> event 10
}


@verbose
def _load_data_001_2024(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2024 dataset (Handwritten Character Classification).

    This dataset contains EEG data from 20 healthy subjects performing
    handwritten character (letter) writing tasks. The data was collected
    for research on handwritten character classification from EEG through
    continuous kinematic decoding.

    The data structure contains:
    - round01_paradigm, round02_paradigm: Main experimental runs
    - round01_sgeyesub, round02_sgeyesub: Eye tracking calibration blocks

    Each paradigm run contains:
    - BrainVisionRDA_data: EEG data (n_samples x 64 channels)
    - BrainVisionRDA_time: Timestamps in seconds
    - ParadigmMarker_data: Event markers
    - ParadigmMarker_time: Event timestamps
    - MoCap_data: Motion capture data (pen position)
    - MoCap_time: Motion capture timestamps

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
    """
    validate_subject(subject, 20, "BNCI2024-001")

    # Download the MAT file for this subject
    url = "{u}001-2024/S{s:02d}.mat".format(u=base_url, s=subject)
    filename = bnci_data_path(url, path, force_update, update_path)[0]

    if only_filenames:
        return [filename]

    # Load the MAT file
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Process the paradigm runs (round01_paradigm, round02_paradigm)
    runs = []
    for round_name in ["round01_paradigm", "round02_paradigm"]:
        if round_name in data:
            raw = _convert_run_001_2024(data[round_name], verbose)
            if raw is not None:
                runs.append(raw)

    # Return in sessions format
    sessions = {"0": {str(ii): run for ii, run in enumerate(runs)}}
    return sessions


@verbose
def _convert_run_001_2024(run, verbose=None):
    """Convert one run from 001-2024 dataset to raw.

    Parameters
    ----------
    run : mat_struct
        Run data from MAT file containing BrainVisionRDA_data,
        BrainVisionRDA_time, ParadigmMarker_data, and ParadigmMarker_time.
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    raw : instance of RawArray
        Raw MNE object.
    """
    # Parse EEG data - shape is (n_samples, n_channels)
    eeg_data = np.asarray(run.BrainVisionRDA_data)
    eeg_time = np.asarray(run.BrainVisionRDA_time)

    # Transpose to (n_channels, n_samples) as expected by MNE
    eeg_data = eeg_data.T

    n_chan, n_samples = eeg_data.shape

    # Calculate sampling rate from timestamps
    # Time is in seconds (absolute timestamps)
    duration = eeg_time[-1] - eeg_time[0]
    sfreq = (n_samples - 1) / duration
    # Round to nearest integer (should be ~500 Hz)
    sfreq = round(sfreq)

    # Channel names: 60 EEG + 4 EOG channels
    # Based on standard 10-20 extended montage for EEG channels
    # fmt: off
    ch_names_eeg = [
        "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
        "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
        "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2",
        "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
        "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5", "C1",
        "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5",
        "P1", "P2", "P6", "PO3",
    ]
    ch_names_eog = ["EOG1", "EOG2", "EOG3", "EOG4"]
    # fmt: on
    ch_names = ch_names_eeg + ch_names_eog
    ch_types = ["eeg"] * 60 + ["eog"] * 4

    # Adjust channel names/types if necessary
    if n_chan != len(ch_names):
        # Fall back to generic channel names if mismatch
        ch_names = ["EEG%d" % ch for ch in range(1, n_chan + 1)]
        ch_types = ["eeg"] * n_chan
        montage = None
    else:
        montage = "standard_1005"

    # Convert from microvolts to volts
    eeg_data = convert_units(eeg_data, from_unit="uV", to_unit="V")

    # Create trigger channel from ParadigmMarker data
    trigger = np.zeros((1, n_samples))

    # Get markers and their timestamps
    markers = np.asarray(run.ParadigmMarker_data)
    marker_times = np.asarray(run.ParadigmMarker_time)

    # Convert marker timestamps to sample indices
    # marker_times are absolute timestamps, eeg_time[0] is start time
    start_time = eeg_time[0]
    for i, (marker, mtime) in enumerate(zip(markers, marker_times)):
        # Only process letter markers (>= 100)
        if marker in _LETTER_MARKER_MAP:
            # Convert time to sample index
            sample_idx = int(round((mtime - start_time) * sfreq))
            if 0 <= sample_idx < n_samples:
                trigger[0, sample_idx] = _LETTER_MARKER_MAP[marker]

    # Stack EEG data with trigger channel
    eeg_data = np.vstack([eeg_data, trigger])
    ch_names = list(ch_names) + ["STI"]
    ch_types = list(ch_types) + ["stim"]

    raw = make_raw(
        eeg_data,
        ch_names,
        ch_types,
        sfreq,
        verbose=verbose,
        montage=montage,
        line_freq=50.0,
    )

    return raw


class BNCI2024_001(BNCIBaseDataset):
    """BNCI 2024-001 Handwritten Character Classification dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 20 healthy subjects performing
    handwritten character (letter) writing tasks. Participants wrote 10
    different letters (a, d, e, f, j, n, o, s, t, v) while EEG was recorded.
    The study investigates the classification of handwritten characters from
    non-invasive EEG through continuous kinematic decoding.

    **Participants**

    - 20 healthy subjects
    - Location: Institute of Neural Engineering, Graz University of Technology,
      Austria

    **Recording Details**

    - Equipment: BrainVision EEG system with 60 EEG + 4 EOG channels
    - Channels: 60 EEG electrodes + 4 EOG electrodes = 64 total
    - Electrode montage: Extended 10-20 system
    - Sampling rate: 500 Hz

    **Experimental Procedure**

    - 10 letter classes: a, d, e, f, j, n, o, s, t, v
    - Participants wrote letters inside a box while fixating on the screen
    - No visual feedback of the writing was provided during the task
    - 2 experimental rounds per subject, each containing ~32 trials per letter
    - Additional motion capture data was recorded (pen position)

    **Event Codes**

    The events correspond to the 10 different letters written by participants:

    - letter_a (1): Letter 'a'
    - letter_d (2): Letter 'd'
    - letter_e (3): Letter 'e'
    - letter_f (4): Letter 'f'
    - letter_j (5): Letter 'j'
    - letter_n (6): Letter 'n'
    - letter_o (7): Letter 'o'
    - letter_s (8): Letter 's'
    - letter_t (9): Letter 't'
    - letter_v (10): Letter 'v'

    References
    ----------
    .. [1] Crell, M. R., & Muller-Putz, G. R. (2024). Handwritten character
           classification from EEG through continuous kinematic decoding.
           Computers in Biology and Medicine, 182, 109132.
           https://doi.org/10.1016/j.compbiomed.2024.109132

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is notable for exploring non-invasive EEG-based handwritten
    character classification, which could enable communication for individuals
    with limited movement capacity. The study demonstrated that handwritten
    characters can be classified from non-invasive EEG and that decoding
    movement kinematics prior to classification improves performance.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=64,
            channel_types={"eeg": 60, "eog": 4},
            montage="eogl1 eogl2 eogl3 eogr1 af7 af3 afz af4 af8 f7 f5 f3 f1 fz f2 f4 f6 f8 ft7 fc5 fc3 fc1 fcz fc2 fc4 fc6 ft8 t7 c5 c3 c1 cz c2 c4 c6 t8 tp7 cp5 cp3 cp1 cpz cp2 cp4 cp6 tp8 p7 p5 p3 p1 pz p2 p4 p6 p8 ppo1h ppo2h po7 po3 poz po4 po8 o1 oz o2",
            hardware="BrainVision",
            sensor_type="active electrodes",
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
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["horizontal", "vertical"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=20,
            health_status="healthy",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=10,
            class_labels=[
                "letter_a",
                "letter_d",
                "letter_e",
                "letter_f",
                "letter_j",
                "letter_n",
                "letter_o",
                "letter_s",
                "letter_t",
                "letter_v",
            ],
            trial_duration=8,
            study_design="Handwritten character task with 10 letters (a,d,e,f,j,n,o,s,t,v) using right index finger",
            feedback_type="none during main paradigm; training included visual guidance",
            stimulus_type="letter cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.compbiomed.2024.109132",
            repository="BNCI Horizon 2020",
            data_url="https://bnci-horizon-2020.eu/database/data-sets",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            preprocessing_steps=[
                "resampling",
                "notch filtering",
                "high-pass filtering",
                "bad channel interpolation",
                "EOG derivative computation",
                "low-pass filtering of EOG",
                "epoching",
                "visual artifact rejection",
            ],
            filter_details=FilterDetails(
                highpass_hz=0.4,
                lowpass_hz=5,
                bandpass={"low_cutoff_hz": 0.3, "high_cutoff_hz": 70.0},
                notch_hz=[49, 51],
                filter_type="Butterworth",
                filter_order=2,
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=128,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Logistic Regression"],
            feature_extraction=["PSD", "Covariance/Riemannian", "ICA"],
            frequency_bands=FrequencyBands(
                analyzed_range=[20.0, 90.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold",
            cv_folds=5,
            evaluation_type=["cross_session"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=94.1,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["robotic_arm", "vr_ar", "neurofeedback"],
        ),
        data_structure=DataStructureMetadata(
            n_trials=60,
            trials_context="per_class",
        ),
        file_format="MAT",
        data_processed=False,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 21)),
            sessions_per_subject=1,
            events={
                "letter_a": 1,
                "letter_d": 2,
                "letter_e": 3,
                "letter_f": 4,
                "letter_j": 5,
                "letter_n": 6,
                "letter_o": 7,
                "letter_s": 8,
                "letter_t": 9,
                "letter_v": 10,
            },
            code="BNCI2024-001",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.1016/j.compbiomed.2024.109132",
            load_fn=_load_data_001_2024,
            base_url=BNCI_URL,
        )
