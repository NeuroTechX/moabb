"""BNCI 2016-002 Emergency Braking during Simulated Driving dataset."""

from datetime import datetime, timezone

import numpy as np
from mne import Annotations
from mne.utils import verbose
from pymatreader import read_mat

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

    # Load HDF5 file (MATLAB v7.3 format) via pymatreader
    mat = read_mat(filename)

    sfreq = float(np.asarray(mat["cnt"]["fs"]).flat[0])
    ch_names = mat["cnt"]["clab"]
    if isinstance(ch_names, str):
        ch_names = [ch_names]
    # pymatreader returns MATLAB's (n_samples, n_channels); transpose to
    # (n_channels, n_samples) expected by MNE RawArray.
    data = np.asarray(mat["cnt"]["x"]).T
    class_names = mat["mrk"]["className"]
    if isinstance(class_names, str):
        class_names = [class_names]
    marker_times = np.asarray(mat["mrk"]["time"]).flatten()
    marker_labels = np.asarray(mat["mrk"]["y"])

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

    # marker_labels has shape (n_classes, n_events); each column is an event
    # and the row index with a positive value indicates the class.
    for i, time_ms in enumerate(marker_times):
        # Find which class this event belongs to
        event_col = marker_labels[:, i]
        for class_idx, value in enumerate(event_col):
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
            n_channels=59,
            channel_types={"eeg": 59, "emg": 1, "eog": 2, "misc": 7},
            montage="extended 10-20",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nose",
            software="TORCS",
            filters={"highpass_hz": 0.1, "lowpass_hz": 250},
            impedance_threshold_kohm={"eeg": 20, "emg": 50},
            sensors=[
                "AF3",
                "AF4",
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
                "EMGf",
                "EOGh",
                "EOGv",
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
                "Fz",
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
                "brake",
                "dist_to_lead",
                "gas",
                "lead_brake",
                "lead_gas",
                "wheel_X",
                "wheel_Y",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["vertical", "horizontal"],
                has_emg=True,
                emg_channels=1,
                other_physiological=["technical_markers"],
            ),
            cap_manufacturer="Easycap",
            cap_model="Easycap",
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            health_status="healthy",
            gender={"male": 14, "female": 4},
            age_mean=30.6,
            age_std=5.4,
            handedness="right-handed",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="emergency_braking",
            task_type="driving_simulation",
            events={
                "car_brake": 1,
                "car_hold": 2,
                "react_emg": 3,
                "car_accelerate": 4,
                "car_collision": 5,
            },
            n_classes=2,
            class_labels=["normal_driving", "emergency_braking"],
            trial_duration=3.0,
            study_design="Participants drove a virtual racing car using steering wheel and gas/brake pedals, tightly following a computer-controlled lead vehicle at 100 km/h. The lead vehicle occasionally decelerated abruptly (20-40s inter-stimulus-interval) to 60-80 km/h, requiring immediate emergency braking. Three blocks of 45 min each with 10-15 min rest between blocks.",
            feedback_type="visual (colored circle indicating distance: green <20m, yellow otherwise; brakelight flashing)",
            stimulus_type="emergency_braking_scenario",
            stimulus_modalities=["visual", "multisensory"],
            primary_modality="visual",
            synchronicity="asynchronous",
            mode="online",
            has_training_test_split=True,
            instructions="Drive a virtual racing car using steering wheel and gas/brake pedals, tightly follow the lead vehicle within 20m at 100 km/h. Perform immediate emergency braking when the lead vehicle decelerates abruptly to avoid a crash.",
            stimulus_presentation={
                "isi_range": "20-40 seconds",
                "deceleration_range": "60-80 km/h",
                "brakelight": "flashing",
                "oncoming_traffic": "present",
                "sharp_curves": "present",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/8/5/056001",
            description="Emergency braking detection during simulated driving using EEG and EMG to predict driver's braking intention before behavioral response.",
            investigators=[
                "Stefan Haufe",
                "Matthias S Treder",
                "Manfred F Gugler",
                "Max Sagebaum",
                "Gabriel Curio",
                "Benjamin Blankertz",
            ],
            institution="Berlin Institute of Technology",
            country="Germany",
            publication_year=2011,
            senior_author="Benjamin Blankertz",
            contact_info=["stefan.haufe@tu-berlin.de"],
            associated_paper_doi="10.1088/1741-2560/8/5/056001",
            funding=[
                "DFG grant",
                "BMBF grant",
                "Bernstein Focus Neurotechnology, Berlin",
            ],
            institution_address="Franklinstraße 28/29, D-10587 Berlin, Germany",
            institution_department="Machine Learning Group, Department of Computer Science",
            ethics_approval=[
                "IRB of Charité University Medicine, Berlin",
                "Declaration of Helsinki",
                "Written informed consent from all participants",
            ],
            keywords=[
                "emergency braking",
                "driving simulation",
                "EEG",
                "EMG",
                "brain-computer interface",
                "neuroergonomics",
                "event-related potentials",
                "machine learning",
                "driver assistance",
            ],
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        contributing_labs=[
            "Machine Learning Group, Berlin Institute of Technology",
            "Bernstein Focus Neurotechnology, Berlin",
            "Neurophysics Group, Charité University Medicine Berlin",
            "Intelligent Data Analysis Group, Fraunhofer Institute FIRST",
        ],
        n_contributing_labs=4,
        data_processed=True,
        file_format=".mat",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual", "Multisensory"],
            type=["Driving", "Neuroergonomics"],
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
                "synchronization",
            ],
            highpass_hz=0.1,
            lowpass_hz=45.0,
            bandpass=[15.0, 90.0],
            notch_hz=50.0,
            filter_type="Chebychev type II (EEG lowpass), Elliptic (EMG bandpass), digital (notch)",
            filter_order="tenth-order (EEG), sixth-order (EMG), second-order (notch)",
            re_reference="nose",
            downsampled_to_hz=200.0,
            epoch_window=[-0.3, 1.2],
            notes="EEG lowpass filtered at 45 Hz (causal). EMG bandpass filtered 15-90 Hz with 50 Hz notch and rectified. All signals synchronized and resampled to 200 Hz. Baseline correction using first 100 ms.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "RLDA",
                "Regularized Linear Discriminant Analysis",
                "Shrinkage LDA",
            ],
            feature_extraction=[
                "Event-Related Potentials",
                "Spatio-temporal features",
                "Bi-serial correlation",
                "Area Under Curve",
            ],
            spatial_filters=["Artifact rejection based on spectral power"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="sequential temporal split",
            evaluation_type=["temporal_validation"],
        ),
        performance={
            "auc": 0.5,
            "braking_time_reduction_ms": 130,
            "braking_distance_reduction_m": 3.66,
        },
        bci_application=BCIApplicationMetadata(
            applications=[
                "driving_assistance",
                "emergency_braking_detection",
                "neuroergonomics",
            ],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="emergency_braking_erp",
        ),
        data_structure=DataStructureMetadata(
            n_trials="~99 emergency braking events per subject (test set)",
            n_blocks=3,
            block_duration_s=2700.0,
            trials_context="Emergency braking events with 20-40s inter-stimulus-interval, total ~225 events across 3 blocks per subject",
        ),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
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
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )
