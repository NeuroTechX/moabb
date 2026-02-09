"""BNCI 2016-002 Emergency Braking during Simulated Driving dataset."""

from datetime import datetime, timezone

import h5py
from mne import Annotations
from mne.utils import verbose

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FrequencyBands,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)

from .base import BNCIBaseDataset
from .utils import bnci_data_path, convert_units, make_raw


# BBCI URL where the data is hosted
BBCI_URL = "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-EmergencyBraking/"

# Subject VP codes for all 18 subjects
# Format: subject_number -> VP code
_SUBJECT_VP_CODES = {
    1: "ae",
    2: "bad",
    3: "bba",
    4: "dx",
    5: "gaa",
    6: "gab",
    7: "gac",
    8: "gae",
    9: "gag",
    10: "gah",
    11: "gal",
    12: "gam",
    13: "ih",
    14: "ii",
    15: "ja",
}


def _read_hdf5_string(f, ref):
    """Read a string from HDF5 object reference."""
    data = f[ref][:]
    return "".join(chr(c) for c in data.flatten())


@verbose
def _load_data_002_2016(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 002-2016 dataset (Emergency Braking during Simulated Driving).

    This dataset contains EEG and physiological signals from 18 subjects
    performing emergency braking maneuvers in a driving simulator.

    Parameters
    ----------
    subject : int
        Subject number (1-15, currently available subjects).
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
        Dictionary of sessions with raw data. Each subject has 3 blocks
        stored as a single session with all data concatenated.
    """
    if subject not in _SUBJECT_VP_CODES:
        raise ValueError(
            f"Subject must be one of {list(_SUBJECT_VP_CODES.keys())}. Got {subject}."
        )

    vp_code = _SUBJECT_VP_CODES[subject]
    url = f"{base_url}VP{vp_code}.mat"
    filename = bnci_data_path(url, path, force_update, update_path, verbose)[0]

    if only_filenames:
        return [filename]

    # Load HDF5 file (MATLAB v7.3 format)
    with h5py.File(filename, "r") as f:
        # Get sampling rate
        sfreq = float(f["cnt"]["fs"][0, 0])

        # Get channel labels
        clab_refs = f["cnt"]["clab"][:]
        ch_names = []
        for ref in clab_refs.flatten():
            ch_names.append(_read_hdf5_string(f, ref))

        # Get continuous data (channels x samples)
        data = f["cnt"]["x"][:]

        # Get class names
        className_refs = f["mrk"]["className"][:]
        class_names = []
        for ref in className_refs.flatten():
            class_names.append(_read_hdf5_string(f, ref))

        # Get marker times (in samples) and labels
        marker_times = f["mrk"]["time"][:].flatten()
        marker_labels = f["mrk"]["y"][:]  # shape: (n_events, n_classes)

    # Determine channel types based on channel names
    # 59 EEG + 2 EOG + 1 EMG + 7 other (gas, brake, wheel, distance, etc.)
    ch_types = []
    eog_channels = ["EOGv", "EOGh"]
    emg_channels = ["EMGf"]
    misc_channels = [
        "lead_gas",
        "lead_brake",
        "dist_to_lead",
        "wheel_X",
        "wheel_Y",
        "gas",
        "brake",
    ]

    for ch_name in ch_names:
        if ch_name in eog_channels:
            ch_types.append("eog")
        elif ch_name in emg_channels:
            ch_types.append("emg")
        elif ch_name in misc_channels:
            ch_types.append("misc")
        else:
            ch_types.append("eeg")

    # Convert EEG/EOG/EMG data to volts (data appears to be in microvolts based on typical BBCI format)
    eeg_eog_emg_mask = [i for i, ct in enumerate(ch_types) if ct in ["eeg", "eog", "emg"]]
    data_scaled = convert_units(
        data.copy(), from_unit="uV", to_unit="V", channel_mask=eeg_eog_emg_mask
    )

    raw = make_raw(
        data_scaled,
        ch_names,
        ch_types,
        sfreq,
        verbose=verbose,
        montage="standard_1005",
        line_freq=50.0,
        meas_date=datetime(2011, 1, 1, tzinfo=timezone.utc),
    )

    # Create annotations from markers
    # Event mapping for P300 paradigm compatibility:
    # car_normal (index 0): lead car normal driving -> NonTarget
    # car_brake (index 1): lead car starts braking (emergency situation onset) -> Target
    # car_hold (index 2): lead car holding/stopped (not used for classification)
    # car_collision (index 3): collision occurred (not used for classification)
    # react_emg (index 4): subject's EMG reaction detected (not used for classification)
    event_mapping = {
        0: "NonTarget",
        1: "Target",
        2: "car_hold",
        3: "car_collision",
        4: "react_emg",
    }

    onset_times = []
    descriptions = []

    for i, time_ms in enumerate(marker_times):
        # Find which class this event belongs to
        event_row = marker_labels[i, :]
        for class_idx, value in enumerate(event_row):
            if value > 0:
                # Marker times are in milliseconds, convert to seconds
                onset_times.append(time_ms / 1000.0)
                descriptions.append(event_mapping[class_idx])
                break

    if onset_times:
        annotations = Annotations(
            onset=onset_times,
            duration=[0.0] * len(onset_times),
            description=descriptions,
        )
        raw.set_annotations(annotations)

    # Return as single session with single run
    sessions = {"0": {"0": raw}}

    return sessions


class BNCI2016_002(BNCIBaseDataset):
    """BNCI 2016-002 Emergency Braking during Simulated Driving dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2016_002  15      69      ~200 brake events   1.0s            200Hz           1
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG and physiological signals recorded during
    emergency braking maneuvers in a driving simulator. The study demonstrated
    that drivers' intentions to perform emergency braking can be detected from
    brain and muscle activity prior to the behavioral response, enabling
    predictive braking assistance systems.

    Participants drove in a realistic driving simulator, maintaining distance
    from a lead vehicle while navigating curves and traffic. When the lead
    vehicle unexpectedly braked (emergency situation), subjects had to brake
    as quickly as possible. The dataset captures the neural and physiological
    signatures preceding emergency braking actions.

    **Participants**

    - 18 subjects (14 males, 4 females) - currently 15 subjects available
    - Age: 30.6 +/- 5.4 years
    - All healthy with valid driver's licenses
    - Location: Berlin Institute of Technology (TU Berlin), Germany

    **Recording Details**

    - Equipment: BrainProducts actiCap system with BrainAmp amplifiers
    - Channels: 59 EEG + 2 EOG + 1 EMG + 7 driving-related signals = 69 total
    - Sampling rate: 200 Hz (downsampled from 1000 Hz)
    - Reference: Common average reference
    - EEG electrode montage: Extended 10-20 system

    **Additional Channels**

    - EOGv, EOGh: Vertical and horizontal electrooculogram
    - EMGf: Electromyogram (right foot, tibialis anterior muscle)
    - lead_gas, lead_brake: Lead vehicle gas/brake pedal positions
    - dist_to_lead: Distance to lead vehicle
    - wheel_X, wheel_Y: Steering wheel position
    - gas, brake: Subject's gas/brake pedal positions

    **Experimental Procedure**

    - Three 45-minute driving blocks per subject (135 minutes total)
    - Driving task: Follow a lead vehicle, maintain safe distance
    - Emergency situations: Lead vehicle brakes unexpectedly
    - Subject response: Emergency braking required
    - Inter-trial interval: Variable (realistic driving conditions)

    **Event Codes**

    For P300 paradigm compatibility, events are mapped to Target/NonTarget:

    - Target: Lead car starts braking (emergency situation onset, originally car_brake)
    - NonTarget: Lead car driving normally (originally car_normal)

    Additional events (not used for P300 classification):

    - car_hold: Lead car holding/stopped
    - car_collision: Collision occurred (subject failed to brake in time)
    - react_emg: Subject's EMG reaction detected (braking initiated)

    **Key Findings**

    The study found that combining EEG and EMG signals enables detection of
    emergency braking intention 130 ms earlier than pedal-based systems alone.
    At 100 km/h, this corresponds to a 3.66 m reduction in braking distance.

    The EEG analysis revealed a characteristic event-related potential signature
    comprising three components:

    1. Sensory registration of critical traffic situations
    2. Mental evaluation of the sensory information
    3. Motor preparation

    References
    ----------
    .. [1] Haufe, S., Treder, M. S., Gugler, M. F., Sagebaum, M., Curio, G., &
           Blankertz, B. (2011). EEG potentials predict upcoming emergency
           brakings during simulated driving. Journal of Neural Engineering,
           8(5), 056001. https://doi.org/10.1088/1741-2560/8/5/056001

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is valuable for research on:

    - Predictive braking assistance systems
    - Neuroergonomics and driving safety
    - Real-time detection of emergency intentions
    - Multimodal biosignal integration (EEG + EMG + vehicle dynamics)

    The paradigm represents a unique blend of ERP (event-related potential)
    analysis with ecological validity in a naturalistic driving context.

    **Data Availability**: Currently 15 of 18 subjects are available.
    Files are hosted at the BBCI (Berlin Brain-Computer Interface) archive.

    License: Creative Commons Attribution Non-Commercial No Derivatives
    (CC BY-NC-ND 4.0)
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=200.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-20",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="Car",
            software="Matlab",
            filters={"highpass_hz": 0.1, "lowpass_hz": 250},
            impedance_threshold_kohm={"eeg": 20, "emg": 50},
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
                has_emg=True,
                emg_channels=2,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            health_status="healthy",
            gender={"male": 14, "female": 4},
            age_mean=30.6,
            handedness="right-handed",
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=2,
            class_labels=["rest", "feet"],
            trial_duration=5.0,
            study_design="to drive a virtual racing car\nusing the steering wheel and gas/brake pedals (automatic\nclutch), and to tightly follow a computer-controlled lead\nvehicle at a driving speed of 100 km h-1.",
            feedback_type="visual (colored circle indicating distance: green <20m, yellow otherwise; brakelight flashing)",
            stimulus_type="oddball",
            stimulus_modalities=["visual", "multisensory"],
            primary_modality="multisensory",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/8/5/056001",
            funding=["DFG grant", "grant nos s", "BMBF grant", "grant no MU MU"],
        ),
        tags=Tags(
            pathology=["Other"],
            modality=["Visual"],
            type=["Clinical/Intervention"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "lowpass filtering",
                "bandpass filtering",
                "notch filtering",
                "rectification",
                "downsampling/upsampling",
                "baseline correction",
            ],
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "Shrinkage LDA"],
            feature_extraction=["Bandpower", "Covariance/Riemannian", "ICA"],
            frequency_bands=FrequencyBands(
                theta=[4, 8],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["vr_ar", "communication"],
            environment="laboratory",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_trials=225,
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(_SUBJECT_VP_CODES.keys()),
            sessions_per_subject=1,
            events={
                "Target": 1,  # Emergency braking onset (lead car brakes)
                "NonTarget": 2,  # Normal driving (lead car driving normally)
            },
            code="BNCI2016-002",
            interval=[-0.5, 1.0],  # 500ms before to 1s after emergency onset
            paradigm="p300",  # ERP-based paradigm
            doi="10.1088/1741-2560/8/5/056001",
            load_fn=_load_data_002_2016,
            base_url=BBCI_URL,
        )
