"""SSVEP Fatigue Dataset with Dynamic Stopping Strategy.

Han et al. (2024), IEEE TNSRE.
DOI: 10.1109/TNSRE.2024.3380635
"""

import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
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
from .utils import TSINGHUA_64CH_NAMES, build_raw_from_epochs, safe_extract_zip


ZENODO_URL = "https://zenodo.org/records/10507229/files/"

# The 4 data conditions and their directory names after zip extraction
_CONDITIONS = [
    ("low_frequency_train_data", "low", "train"),
    ("low_frequency_fatigue_data", "low", "fatigue"),
    ("high_frequency_train_data", "high", "train"),
    ("high_frequency_fatigue_data", "high", "fatigue"),
]

# fmt: off
# Low-frequency events (16): 8.0-15.5 Hz, 0.5 Hz step
# High-frequency events (16): 25.5-33.0 Hz, 0.5 Hz step
_EVENTS = {
    "8": 1, "8.5": 2, "9": 3, "9.5": 4,
    "10": 5, "10.5": 6, "11": 7, "11.5": 8,
    "12": 9, "12.5": 10, "13": 11, "13.5": 12,
    "14": 13, "14.5": 14, "15": 15, "15.5": 16,
    "25.5": 17, "26": 18, "26.5": 19, "27": 20,
    "27.5": 21, "28": 22, "28.5": 23, "29": 24,
    "29.5": 25, "30": 26, "30.5": 27, "31": 28,
    "31.5": 29, "32": 30, "32.5": 31, "33": 32,
}
# fmt: on


class Han2024Fatigue(BaseDataset):
    """SSVEP fatigue dataset with two frequency paradigms.

    Dataset from [1]_.

    This dataset contains 64-channel EEG recordings from 24 healthy subjects
    (12 males, 12 females, aged 18-26) performing two SSVEP-BCI tasks:

    - **Low-frequency paradigm:** 16 targets (8.0-15.5 Hz, 0.5 Hz step)
    - **High-frequency paradigm:** 16 targets (25.5-33.0 Hz, 0.5 Hz step)

    Both paradigms used JFPM encoding with phases cycling through
    0, 0.5*pi, pi, 1.5*pi in a 4x4 matrix layout.

    The experiment consisted of two phases: training (6 blocks per frequency
    condition) and fatigue (24 blocks per condition). Each block contained
    16 trials (2 s stimulation per trial).

    EEG was recorded at 1000 Hz with a Synamps2 system (Neuroscan) and 64
    channels. Each epoch spans 3000 samples (3 s at 1000 Hz).

    .. note::

       Channel selection is critical for this dataset. Using all 64 channels
       with CCA-based methods yields near-chance accuracy because the high
       channel-to-sample ratio causes overfitting. The paper uses 9 occipital
       channels (PO7, PO3, POz, PO4, PO8, O1, Oz, O2, and one additional)
       and achieves >90% with TRCA. Users should pick occipital channels
       before classification.

       Additionally, the cross-session evaluation (training on alert session
       '0', testing on fatigued session '1') is a challenging domain-shift
       problem that standard CCA/TRCA may not handle well without
       fatigue-aware strategies.

    Data is stored as [16, 64, 3000, N_blocks] matrices (targets, channels,
    timepoints, blocks) in per-subject zip files on Zenodo. Each subject has
    4 separate files: low_frequency_train, low_frequency_fatigue,
    high_frequency_train, high_frequency_fatigue.

    In MOABB, this is mapped as:
    - Session '0': Training blocks (6 blocks per condition, 12 total)
    - Session '1': Fatigue blocks (24 blocks per condition, 48 total)

    References
    ----------
    .. [1] Y. Han, Y. Ke, R. Wang, T. Wang, and D. Ming, "Enhancing
       SSVEP-BCI Performance Under Fatigue State Using Dynamic Stopping
       Strategy," IEEE Trans. Neural Syst. Rehab. Eng., vol. 32,
       pp. 1407-1415, 2024. DOI: 10.1109/TNSRE.2024.3380635
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="standard_1005",
            hardware="Synamps2 (Neuroscan)",
            sensors=TSINGHUA_64CH_NAMES,
            line_freq=50.0,
            reference="Cz",
            ground="midway between Fz and FPz",
            impedance_threshold_kohm=10,
            filters={"bandpass_hz": [0.15, 200.0]},
        ),
        participants=ParticipantMetadata(
            n_subjects=24,
            health_status="healthy",
            gender={"male": 12, "female": 12},
            age_min=18,
            age_max=26,
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            events=dict(_EVENTS),
            n_classes=32,
            trial_duration=2.0,
            stimulus_type="JFPM visual flicker",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=True,
            task_type="gaze-shifting",
            feedback_type="none",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TNSRE.2024.3380635",
            investigators=[
                "Yuheng Han",
                "Yufeng Ke",
                "Ruiyan Wang",
                "Tao Wang",
                "Dong Ming",
            ],
            senior_author="Dong Ming",
            institution="Tianjin University",
            country="CN",
            repository="Zenodo",
            data_url="https://zenodo.org/records/10507229",
            license="CC BY 4.0",
            publication_year=2024,
            institution_department="Academy of Medical Engineering and Translational Medicine, Tianjin University",
            ethics_approval=["Research Ethics Committee of Tianjin University"],
            funding=[
                "National Key Research and Development Program of China (Grant 2021YFF1200603)",
                "National Natural Science Foundation of China (Grants 62276184, 61806141)",
            ],
            keywords=["SSVEP", "BCI", "fatigue", "dynamic stopping", "EEG"],
        ),
        preprocessing=PreprocessingMetadata(data_state="epoched"),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=(
                [8.0 + i * 0.5 for i in range(16)] + [25.5 + i * 0.5 for i in range(16)]
            ),
            frequency_resolution_hz=0.5,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=60,
            n_trials="960 per frequency band (16 targets x 60 blocks)",
            trials_context="6 training + 24 fatigue blocks per frequency condition",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["TRCA"],
            feature_extraction=None,
            frequency_bands=None,
            spatial_filters=["TRCA"],
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=None),
        bci_application=BCIApplicationMetadata(environment="lab", online_feedback=False),
        tags=Tags(pathology=["healthy"], modality=["visual"], type=["perception"]),
        sessions_per_subject=2,
        file_format="MAT",
    )

    _events = _EVENTS

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 25)),
            sessions_per_subject=2,
            events=self._events,
            code="Han2024Fatigue",
            interval=[0.14, 2.14],
            paradigm="ssvep",
            doi="10.1109/TNSRE.2024.3380635",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across training and fatigue sessions.

        Loads 4 separate .mat files per subject (low/high x train/fatigue).
        Each file has shape [16, 64, 3000, N_blocks] = (targets, channels,
        timepoints, blocks). Training files have 6 blocks, fatigue files
        have 24 blocks.
        """
        n_targets = 16
        sfreq = 1000

        file_paths = self.data_path(subject)

        # Group conditions by session: train->session '0', fatigue->session '1'
        session_epochs = {"0": [], "1": []}
        session_events = {"0": [], "1": []}
        for dir_name, freq_band, phase in _CONDITIONS:
            mat_path = file_paths[dir_name]
            mat = loadmat(mat_path, squeeze_me=True)
            data = mat["data"]  # (16, 64, 3000, N_blocks)
            n_blocks = data.shape[3]

            # Event offset: low-freq classes are 1-16, high-freq are 17-32
            event_offset = 16 if freq_band == "high" else 0
            sess_key = "0" if phase == "train" else "1"

            # Reshape all blocks at once: (N_blocks*16, 64, 3000)
            reshaped = data.transpose(3, 0, 1, 2).reshape(-1, 64, 3000)
            event_ids = np.tile(np.arange(1, n_targets + 1) + event_offset, n_blocks)

            session_epochs[sess_key].append(reshaped)
            session_events[sess_key].append(event_ids)

        sessions = {}
        for sess_name in session_epochs:
            if not session_epochs[sess_name]:
                continue
            all_data = np.concatenate(session_epochs[sess_name], axis=0)
            all_events = np.concatenate(session_events[sess_name])
            raw = build_raw_from_epochs(
                all_data, TSINGHUA_64CH_NAMES, sfreq, all_events, "standard_1005"
            )
            sessions[sess_name] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if all 4 condition files exist
        all_paths = {}
        all_found = True
        for dir_name, _, _ in _CONDITIONS:
            mat_file = data_dir / dir_name / f"S{subject}.mat"
            if mat_file.exists() and not force_update:
                all_paths[dir_name] = str(mat_file)
            else:
                all_found = False

        if all_found:
            return all_paths

        # Download the zip file
        url = f"{ZENODO_URL}S{subject}.zip"
        zip_path = dl.data_dl(url, sign, path, force_update, verbose)

        # Extract
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            safe_extract_zip(zf, data_dir)

        # Find the .mat files in subdirectories
        all_paths = {}
        for dir_name, _, _ in _CONDITIONS:
            mat_file = data_dir / dir_name / f"S{subject}.mat"
            if mat_file.exists():
                all_paths[dir_name] = str(mat_file)
            else:
                # Search recursively
                found = list(data_dir.rglob(f"{dir_name}/S{subject}.mat"))
                if found:
                    all_paths[dir_name] = str(found[0])
                else:
                    raise FileNotFoundError(
                        f"Could not find {dir_name}/S{subject}.mat "
                        f"after extracting S{subject}.zip"
                    )

        return all_paths
