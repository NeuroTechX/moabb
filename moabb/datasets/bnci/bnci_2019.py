"""BNCI 2019 datasets."""

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_gdf

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
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


BNCI_URL_001_2019 = "http://bnci-horizon-2020.eu/database/data-sets/001-2019/"


class BNCI2019_001(BaseDataset):
    """BNCI 2019-001 Motor Imagery dataset for Spinal Cord Injury patients.

    .. admonition:: Dataset summary

        ============= ======= ======= ================= =============== =============== ============
        Name          #Subj   #Chan   #Trials/class     Trials length   Sampling Rate   #Sessions
        ============= ======= ======= ================= =============== =============== ============
        BNCI2019_001  10      61+3EOG 72 per class      3s              256Hz           1
        ============= ======= ======= ================= =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset consists of EEG recordings from 10 participants with cervical
    spinal cord injury (SCI) performing attempted hand and arm movements.

    Participants attempted five movement types: supination, pronation, hand open,
    palmar grasp, and lateral grasp.

    **Participants**

    - 10 participants with cervical spinal cord injury
    - Age range: 20-78 years (mean 49.8, SD 17.6)
    - Gender: 9 male, 1 female
    - Handedness: All right-handed

    **Recording Details**

    - Channels: 61 EEG + 3 EOG electrodes
    - Sampling rate: 256 Hz
    - Reference: Left earlobe

    **Motor Imagery Classes**

    - supination (776): Forearm supination
    - pronation (777): Forearm pronation
    - hand_open (779): Hand opening movement
    - palmar_grasp (925): Palmar (power) grasp
    - lateral_grasp (926): Lateral (key) grasp

    References
    ----------
    .. [1] Ofner, P. et al. (2019). Attempted arm and hand movements can be
           decoded from low-frequency EEG from persons with spinal cord injury.
           Scientific Reports, 9(1), 7134.
           https://doi.org/10.1038/s41598-019-43594-9

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=61,
            channel_types={"eeg": 61, "eog": 3},
            montage="10-5",
            hardware="g.tec",
            sensor_type="active electrode",
            reference="left earlobe",
            ground="AFF2h",
            software="EEGlab 14.1.1b",
            filters="50 Hz notch, 0.01-100 Hz bandpass",
            sensors=[
                "AFz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CCP1h",
                "CCP2h",
                "CCP3h",
                "CCP4h",
                "CCP5h",
                "CCP6h",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPP1h",
                "CPP2h",
                "CPP3h",
                "CPP4h",
                "CPP5h",
                "CPP6h",
                "CPz",
                "Cz",
                "F1",
                "F2",
                "F3",
                "F4",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCC1h",
                "FCC2h",
                "FCC3h",
                "FCC4h",
                "FCC5h",
                "FCC6h",
                "FCz",
                "FFC1h",
                "FFC2h",
                "FFC3h",
                "FFC4h",
                "FFC5h",
                "FFC6h",
                "Fz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "POz",
                "PPO1h",
                "PPO2h",
                "Pz",
                "eog-l",
                "eog-m",
                "eog-r",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=3,
                eog_type=[
                    "above nasion",
                    "below outer canthi left",
                    "below outer canthi right",
                ],
                has_emg=False,
            ),
            cap_manufacturer="g.tec medical engineering GmbH",
            cap_model="g.GAMMAsys/g.LADYbird",
            electrode_type="active electrode",
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="patients",
            gender={"male": 9, "female": 1},
            age_mean=49.8,
            age_min=20,
            age_max=78,
            ages=[35, 42, 62, 20, 57, 78, 27, 69, 53, 55],
            handedness="right-handed (all participants originally)",
            clinical_population="spinal cord injury",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="attempted movement",
            n_classes=5,
            class_labels=[
                "hand_open",
                "palmar_grasp",
                "lateral_grasp",
                "pronation",
                "supination",
            ],
            trial_duration=5.0,
            tasks=[
                "hand_open",
                "palmar_grasp",
                "lateral_grasp",
                "pronation",
                "supination",
            ],
            study_design="motor imagery and attempted movements",
            feedback_type="visual feedback (online paradigm only - movement icon displayed when movement detected)",
            stimulus_type="visual cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="both",
            has_training_test_split=True,
            instructions="Participants were instructed to attempt or execute movements based on class cue displayed on screen. They were asked to focus gaze on fixation cross, avoid eye movements, swallowing, and blinking during trial period.",
            events={
                "hand_open": 1,
                "palmar_grasp": 2,
                "lateral_grasp": 3,
                "pronation": 4,
                "supination": 5,
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41598-019-43594-9",
            description="This dataset investigates whether attempted arm and hand movements in persons with spinal cord injury can be decoded from low-frequency EEG signals (MRCPs). The study includes offline 5-class classification and online proof-of-concept for self-paced movement detection.",
            investigators=[
                "Patrick Ofner",
                "Andreas Schwarz",
                "Joana Pereira",
                "Daniela Wyss",
                "Renate Wildburger",
                "Gernot R. Müller-Putz",
            ],
            institution="Graz University of Technology",
            institution_department="Institute of Neural Engineering, BCI-Lab",
            institution_address="Graz, Austria",
            country="Austria",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.2222268",
            license="CC-BY-4.0",
            publication_year=2019,
            senior_author="Gernot R. Müller-Putz",
            contact_info=["gernot.mueller@tugraz.at"],
            associated_paper_doi="10.1038/s41598-019-43594-9",
            funding=["European ICT Programme Project H2020-643955 'MoreGrasp'"],
            ethics_approval=[
                "Ethics committee for the hospitals of the Austrian general accident insurance institution AUVA (approval number 3/2017)"
            ],
            acknowledgements="This work is supported by the European ICT Programme Project H2020-643955 'MoreGrasp'.",
        ),
        tags=Tags(
            pathology=["Spinal Cord Injury"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw (GDF format)",
            preprocessing_applied=True,
            preprocessing_steps=[
                "bandpass filter",
                "notch filter",
                "ICA",
                "artifact rejection",
            ],
            highpass_hz=0.01,
            lowpass_hz=100,
            bandpass={"low_cutoff_hz": 0.01, "high_cutoff_hz": 100.0},
            notch_hz=[50],
            filter_type="Chebyshev",
            filter_order=8,
            artifact_methods=[
                "ICA",
                "visual inspection",
                "abnormal joint probability",
                "abnormal kurtosis",
            ],
            re_reference="CAR",
            notes="Noisy channels were visually inspected and removed. AFz was removed by default as it is sensitive to eye blinks and eye movements. ICA was performed on 0.3-70 Hz filtered signals using extended infomax. PCA dimensionality reduction retained 99% variance. Artifact-contaminated ICs (muscle and eye-related) were removed. Trials with values above/below ±100 μV, abnormal joint probabilities, or abnormal kurtosis (5x SD threshold) were rejected. Final analysis used 0.3-3 Hz bandpass filter.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Shrinkage LDA", "sLDA"],
            feature_extraction=["time-domain low-frequency signals", "MRCPs", "ICA"],
            frequency_bands={
                "analyzed_range": [0.3, 3.0],
            },
            spatial_filters=["CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10x10-fold",
            cv_folds=10,
            evaluation_type=["within_subject", "cross_validation"],
        ),
        performance={
            "accuracy_percent": 45.3,
            "peak_accuracy_5class": 45.3,
            "peak_latency_5class_s": 1.1,
            "confidence_interval_lower": 40.3,
            "confidence_interval_upper": 50.3,
            "chance_level_5class": 20.0,
            "significance_level_5class": 22.3,
            "peak_accuracy_3class_subset": 53.0,
            "peak_latency_3class_subset_s": 1.0,
            "online_accuracy_2class": 68.4,
            "online_TPR_percent": 31.75,
            "online_FP_per_min": 3.4,
        },
        bci_application=BCIApplicationMetadata(
            applications=[
                "neuroprosthetic",
                "upper_limb_control",
                "hand_grasp_control",
            ],
            environment="indoor",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "hand_open",
                "palmar_grasp",
                "lateral_grasp",
                "pronation",
                "supination",
            ],
            cue_duration_s=3.0,
            imagery_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=360,
            trials_context="total per subject (72 trials per class)",
            n_trials_per_class={
                "hand_open": 72,
                "palmar_grasp": 72,
                "lateral_grasp": 72,
                "pronation": 72,
                "supination": 72,
            },
            n_blocks=9,
        ),
        sessions_per_subject=1,
        runs_per_session=9,
        file_format="GDF",
        data_processed=False,
        contributing_labs=[
            "Graz University of Technology Institute of Neural Engineering BCI-Lab",
            "AUVA rehabilitation clinic Tobelbad",
        ],
        n_contributing_labs=2,
        abstract="We show that persons with spinal cord injury (SCI) retain decodable neural correlates of attempted arm and hand movements. We investigated hand open, palmar grasp, lateral grasp, pronation, and supination in 10 persons with cervical SCI. Discriminative movement information was provided by the time-domain of low-frequency electroencephalography (EEG) signals. Based on these signals, we obtained a maximum average classification accuracy of 45% (chance level was 20%) with respect to the five investigated classes. Pattern analysis indicates central motor areas as the origin of the discriminative signals. Furthermore, we introduce a proof-of-concept to classify movement attempts online in a closed loop, and tested it on a person with cervical SCI. We achieved here a modest classification performance of 68.4% with respect to palmar grasp vs hand open (chance level 50%).",
        methodology="10 participants with cervical SCI were recruited from a rehabilitation center (AUVA rehabilitation clinic, Tobelbad, Austria). Participants were aged 20-78 years with neurological level of injury C1-C7 and AIS scores A-D. They sat in wheelchairs and attempted/executed movements based on visual cues shown on screen. Each trial lasted 5 seconds with a fixation cross and beep at start, class cue displayed at 2 seconds. 9 runs with 40 trials per run were recorded (360 trials total, 72 per class). EEG was recorded from 61 electrodes using g.tec g.USBamps and g.GAMMAsys/g.LADYbird active electrode system at 256 Hz with 0.01-100 Hz bandpass and 50 Hz notch filter. Preprocessing included visual inspection, ICA artifact removal, trial rejection, and 0.3-3 Hz bandpass filtering. Classification used shrinkage LDA with 10x10 cross-validation. Online proof-of-concept used modified training paradigm with ready/go cues and 3-class classifier (hand open, palmar grasp, rest) with pre/post class detection logic.",
    )

    _EVENTS = {
        "supination": 776,
        "pronation": 777,
        "hand_open": 779,
        "palmar_grasp": 925,
        "lateral_grasp": 926,
    }

    _MOVEMENT_RUNS = [3, 4, 5, 6, 7, 10, 11, 12, 13]

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=self._EVENTS,
            code="BNCI2019-001",
            interval=[2, 5],
            paradigm="imagery",
            doi="10.1038/s41598-019-43594-9",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)
        montage = make_standard_montage("standard_1005")
        eog_channels = ["eog-l", "eog-m", "eog-r"]
        data = {}
        for run_idx, path in enumerate(file_paths):
            raw = read_raw_gdf(path, eog=eog_channels, preload=True, verbose="ERROR")
            raw.set_montage(montage, on_missing="ignore")
            raw._data[np.isnan(raw._data)] = 0
            # Convert EEG channels (excluding last 3 EOG channels) from uV to V
            raw._data[:-3] *= 1e-6
            stim = raw.annotations.description.astype(np.dtype("<21U"))
            stim[stim == "776"] = "supination"
            stim[stim == "777"] = "pronation"
            stim[stim == "779"] = "hand_open"
            stim[stim == "925"] = "palmar_grasp"
            stim[stim == "926"] = "lateral_grasp"
            raw.annotations.description = stim
            raw.info["line_freq"] = 50.0
            raw.set_meas_date(datetime(2019, 1, 1, tzinfo=timezone.utc))
            data[str(run_idx)] = raw
        return {"0": data}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return paths to data files for a given subject."""
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject number {subject}. Valid: {self.subject_list}"
            )
        url = f"{BNCI_URL_001_2019}P{subject:02d}.zip"
        zip_path = dl.data_dl(url, "BNCI", path, force_update, verbose)
        zip_dir = Path(zip_path).parent
        if not (zip_dir / f"P{subject:02d} Run 3.gdf").exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(zip_dir)
        paths = []
        for run in self._MOVEMENT_RUNS:
            gdf_path = zip_dir / f"P{subject:02d} Run {run}.gdf"
            if gdf_path.exists():
                paths.append(str(gdf_path))
        if not paths:
            raise FileNotFoundError(f"No GDF files found for subject {subject}")
        return paths
