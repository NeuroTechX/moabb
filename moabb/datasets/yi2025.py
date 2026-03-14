"""Multi-joint upper-limb motor imagery EEG dataset.

Yi, Chen, Wang, et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-05286-0
Data DOI: 10.6084/m9.figshare.24123303.v3
"""

import logging
from pathlib import Path

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
from .utils import build_raw_from_epochs


log = logging.getLogger(__name__)

_SFREQ = 250.0

# Event codes for 8 multi-joint MI classes.
_EVENTS = {
    "hand_open_close": 1,
    "wrist_flex_ext": 2,
    "wrist_abd_add": 3,
    "elbow_pron_sup": 4,
    "elbow_flex_ext": 5,
    "shoulder_pron_sup": 6,
    "shoulder_abd_add": 7,
    "shoulder_flex_ext": 8,
}

# 62 EEG channels (64-ch cap minus M1, M2, HEO, VEO, EKG, EMG).
# fmt: off
_CH_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]
# fmt: on

# Per-subject Figshare file IDs (subjectN_data_label.mat, ~159 MB each).
_FILE_IDS = {
    1: 52690937,
    2: 52690916,
    3: 52690898,
    4: 52690892,
    5: 52690907,
    6: 52690931,
    7: 52690925,
    8: 52690919,
    9: 52690943,
    10: 52690913,
    11: 52690901,
    12: 52690922,
    13: 52690940,
    14: 52690934,
    15: 52690928,
    16: 52690895,
    17: 52690904,
    18: 52690910,
}


class Yi2025(BaseDataset):
    """Multi-joint upper-limb MI dataset from Yi et al. 2025.

    Dataset from [1]_.

    This dataset contains EEG recordings from 18 healthy subjects
    performing 8-class motor imagery of different upper-limb joints.
    Recorded with 64-channel Neuroscan SynAmps2 at 1000 Hz,
    downsampled to 250 Hz. The pre-processed epoched data contains
    62 EEG channels (excluding M1, M2, HEO, VEO, EKG, EMG).

    The 8 MI classes correspond to multi-joint movements:

    1. Hand open/close
    2. Wrist flexion/extension
    3. Wrist abduction/adduction
    4. Elbow pronation/supination
    5. Elbow flexion/extension
    6. Shoulder pronation/supination
    7. Shoulder abduction/adduction
    8. Shoulder flexion/extension

    Each subject performed 1 session with 8 blocks of 40 trials
    (5 per class), for 320 total trials. Trial structure: 2 s fixation,
    2 s cue, 4 s MI, 10-12 s rest.

    The shared data is pre-processed (bandpass 4-40 Hz, CAR,
    downsampled to 250 Hz) and epoched (0-4 s post-cue, 1000 samples).

    .. note::

       Each subject file is ~159 MB (total ~2.9 GB for 18 subjects).

    References
    ----------
    .. [1] Yi, W., Chen, J., Wang, D., et al. (2025). A multi-modal
           dataset of EEG and fNIRS for motor imagery of multi-types of
           joints from unilateral upper limb. Scientific Data, 12, 953.
           https://doi.org/10.1038/s41597-025-05286-0

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=62,
            channel_types={"eeg": 62},
            hardware="Neuroscan SynAmps2",
            reference="left mastoid (M1)",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
            montage="standard_1005",
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            gender={"female": 10, "male": 8},
            age_min=22,
            age_max=27,
            health_status="healthy",
            species="human",
            handedness="right",
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=8,
            class_labels=list(_EVENTS.keys()),
            events=dict(_EVENTS),
            trial_duration=4.0,
            study_design=(
                "8-class multi-joint upper-limb MI. 8 blocks of 40 "
                "trials (5 per class), 320 total trials per subject."
            ),
            stimulus_type="video + text",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            synchronicity="cue-based",
            feedback_type="none",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-05286-0",
            investigators=[
                "Weibo Yi",
                "Jiaming Chen",
                "Dan Wang",
                "Xinkang Hu",
                "Meng Xu",
                "Fangda Li",
                "Shuhan Wu",
                "Jin Qian",
            ],
            institution="China University of Mining and Technology",
            country="CN",
            data_url="https://figshare.com/articles/dataset/Data/24123303",
            publication_year=2025,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor Imagery"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            highpass_hz=4.0,
            lowpass_hz=40.0,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=list(_EVENTS.keys()),
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=320,
            n_trials_per_class={k: 40 for k in _EVENTS},
            n_blocks=8,
            trials_context="8 blocks x 40 trials (5 per class x 8 classes)",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["ShallowConvNet"],
            feature_extraction=["ERSP"],
            frequency_bands={
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "bandpass": [4.0, 40.0],
            },
            spatial_filters=["CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold",
            cv_folds=5,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=True,
        file_format="MAT (pre-epoched)",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 19)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Yi2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-05286-0",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")

        path = dl.get_dataset_path("Yi2025", path)
        basepath = Path(path) / "MNE-yi2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        mat_file = basepath / f"subject{subject}_data_label.mat"
        if not mat_file.exists() or force_update:
            file_id = _FILE_IDS[subject]
            url = f"https://ndownloader.figshare.com/files/{file_id}"
            dl_path = dl.data_dl(url, "Yi2025", path, force_update, verbose)
            dl_path = Path(dl_path)
            if dl_path != mat_file:
                dl_path.rename(mat_file)

        return [str(mat_file)]

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)
        mat_path = file_paths[0]

        mat = loadmat(mat_path, squeeze_me=True)
        data = mat["data"]  # (320, 62, 1000): trials x channels x samples
        labels = mat["label"]  # (320,): class labels 1-8

        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {data.shape}")

        n_ch = data.shape[1]
        ch_names = (
            list(_CH_NAMES[:n_ch])
            if n_ch <= len(_CH_NAMES)
            else [f"EEG{i + 1}" for i in range(n_ch)]
        )

        # Data is pre-processed (bandpass 4-40 Hz, CAR, downsampled) and
        # already in volts (max ~0.87 V), so no additional scaling needed.
        raw = build_raw_from_epochs(
            data,
            ch_names,
            _SFREQ,
            labels.astype(int),
            "standard_1005",
            scale=1.0,
            buffer_samples=int(0.5 * _SFREQ),
            onset_sample=0,
        )

        return {"0": {"0": raw}}
