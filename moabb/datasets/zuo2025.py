"""Lower-limb motor imagery EEG dataset for knee pain patients.

Zuo, Yin, Wang, et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-05767-2
Data DOI: 10.6084/m9.figshare.28740260.v3
"""

import logging
from pathlib import Path

import mne
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
    SignalProcessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

_SFREQ = 500.0

_EVENTS = {
    "left_leg": 1,
    "right_leg": 2,
}

# 30 EEG channels (from channels.tsv, excluding 2 EOG channels).
# fmt: off
_CH_NAMES = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
    "FCz", "FC3", "FC4", "FT7", "FT8",
    "Cz", "C3", "C4", "T3", "T4",
    "CPz", "CP3", "CP4", "TP7", "TP8",
    "Pz", "P3", "P4", "T5", "T6",
    "Oz", "O1", "O2",
]
# fmt: on

# Per-subject Figshare file IDs for raw session files.
# fmt: off
_RAW_FILES = {
    1: [
        (53698397, "sub-01_ses-01_task-MI.mat"),
        (53698340, "sub-01_ses-02_task-MI.mat"),
        (53698343, "sub-01_ses-03_task-MI.mat"),
        (53698346, "sub-01_ses-04_task-MI.mat"),
        (53698349, "sub-01_ses-05_task-MI.mat"),
    ],
    2: [
        (53698406, "sub-02_ses-01_task-MI.mat"),
        (53698409, "sub-02_ses-02_task-MI.mat"),
        (53698352, "sub-02_ses-03_task-MI.mat"),
        (53698355, "sub-02_ses-04_task-MI.mat"),
        (53698358, "sub-02_ses-05_task-MI.mat"),
    ],
    3: [
        (53698364, "sub-03_ses-01_task-MI.mat"),
        (53698367, "sub-03_ses-02_task-MI.mat"),
        (53698328, "sub-03_ses-03_task-MI.mat"),
        (53698319, "sub-03_ses-04_task-MI.mat"),
        (53698370, "sub-03_ses-05_task-MI.mat"),
    ],
    4: [
        (53698433, "sub-04_ses-01_task-MI.mat"),
        (53698373, "sub-04_ses-02_task-MI.mat"),
        (53698442, "sub-04_ses-03_task-MI.mat"),
        (53698322, "sub-04_ses-04_task-MI.mat"),
        (53698376, "sub-04_ses-05_task-MI.mat"),
    ],
    5: [
        (53698451, "sub-05_ses-01_task-MI.mat"),
        (53698379, "sub-05_ses-02_task-MI.mat"),
        (53698331, "sub-05_ses-03_task-MI.mat"),
        (53698334, "sub-05_ses-04_task-MI.mat"),
        (53698337, "sub-05_ses-05_task-MI.mat"),
    ],
    6: [
        (53698463, "sub-06_ses-01_task-MI.mat"),
        (53698466, "sub-06_ses-02_task-MI.mat"),
        (53698469, "sub-06_ses-03_task-MI.mat"),
        (53698475, "sub-06_ses-04_task-MI.mat"),
        (53698532, "sub-06_ses-05_task-MI.mat"),
    ],
    7: [
        (53698400, "sub-07_ses-01_task-MI.mat"),
        (53698325, "sub-07_ses-02_task-MI.mat"),
        (53698403, "sub-07_ses-03_task-MI.mat"),
        (53698484, "sub-07_ses-04_task-MI.mat"),
        (53698487, "sub-07_ses-05_task-MI.mat"),
    ],
    8: [
        (53698412, "sub-08_ses-01_task-MI.mat"),
        (53698499, "sub-08_ses-02_task-MI.mat"),
        (53698502, "sub-08_ses-03_task-MI.mat"),
        (53698571, "sub-08_ses-04_task-MI.mat"),
        (53698511, "sub-08_ses-05_task-MI.mat"),
    ],
    9: [
        (53698424, "sub-09_ses-01_task-MI.mat"),
        (53698361, "sub-09_ses-02_task-MI.mat"),
        (53698520, "sub-09_ses-03_task-MI.mat"),
        (53698430, "sub-09_ses-04_task-MI.mat"),
        (53698436, "sub-09_ses-05_task-MI.mat"),
    ],
    10: [
        (53698535, "sub-10_ses-01_task-MI.mat"),
        (53698538, "sub-10_ses-02_task-MI.mat"),
        (53698445, "sub-10_ses-03_task-MI.mat"),
        (53698457, "sub-10_ses-04_task-MI.mat"),
        (53698460, "sub-10_ses-05_task-MI.mat"),
    ],
    11: [
        (53698553, "sub-11_ses-01_task-MI.mat"),
        (53698556, "sub-11_ses-02_task-MI.mat"),
        (53698559, "sub-11_ses-03_task-MI.mat"),
        (53698562, "sub-11_ses-04_task-MI.mat"),
        (53698472, "sub-11_ses-05_task-MI.mat"),
    ],
    12: [
        (53698568, "sub-12_ses-01_task-MI.mat"),
        (53698478, "sub-12_ses-02_task-MI.mat"),
        (53698382, "sub-12_ses-03_task-MI.mat"),
        (53698385, "sub-12_ses-04_task-MI.mat"),
        (53698388, "sub-12_ses-05_task-MI.mat"),
    ],
    13: [
        (53698643, "sub-13_ses-01_task-MI.mat"),
        (53698586, "sub-13_ses-02_task-MI.mat"),
        (53698646, "sub-13_ses-03_task-MI.mat"),
        (53698652, "sub-13_ses-04_task-MI.mat"),
        (53698490, "sub-13_ses-05_task-MI.mat"),
    ],
    14: [
        (53698493, "sub-14_ses-01_task-MI.mat"),
        (53698391, "sub-14_ses-02_task-MI.mat"),
    ],
    15: [
        (53698505, "sub-15_ses-01_task-MI.mat"),
        (53698508, "sub-15_ses-02_task-MI.mat"),
        (53698598, "sub-15_ses-03_task-MI.mat"),
        (53698601, "sub-15_ses-04_task-MI.mat"),
        (53698514, "sub-15_ses-05_task-MI.mat"),
    ],
    16: [
        (53698517, "sub-16_ses-01_task-MI.mat"),
        (53698610, "sub-16_ses-02_task-MI.mat"),
        (53698523, "sub-16_ses-03_task-MI.mat"),
        (53698529, "sub-16_ses-04_task-MI.mat"),
        (53698619, "sub-16_ses-05_task-MI.mat"),
    ],
    17: [
        (53698622, "sub-17_ses-01_task-MI.mat"),
        (53698628, "sub-17_ses-02_task-MI.mat"),
        (53698634, "sub-17_ses-03_task-MI.mat"),
        (53698541, "sub-17_ses-04_task-MI.mat"),
        (53698418, "sub-17_ses-05_task-MI.mat"),
    ],
    18: [
        (53698421, "sub-18_ses-01_task-MI.mat"),
        (53698427, "sub-18_ses-03_task-MI.mat"),
        (53698544, "sub-18_ses-04_task-MI.mat"),
        (53698439, "sub-18_ses-05_task-MI.mat"),
    ],
    19: [
        (53698649, "sub-19_ses-01_task-MI.mat"),
        (53698655, "sub-19_ses-02_task-MI.mat"),
    ],
    20: [
        (53698565, "sub-20_ses-01_task-MI.mat"),
        (54029906, "sub-20_ses-02_task-MI.mat"),
        (54029909, "sub-20_ses-03_task-MI.mat"),
        (53698574, "sub-20_ses-04_task-MI.mat"),
        (53698580, "sub-20_ses-05_task-MI.mat"),
    ],
    21: [
        (53698583, "sub-21_ses-01_task-MI.mat"),
        (53698589, "sub-21_ses-02_task-MI.mat"),
        (53698592, "sub-21_ses-05_task-MI.mat"),
    ],
    22: [
        (53698595, "sub-22_ses-03_task-MI.mat"),
        (53698481, "sub-22_ses-05_task-MI.mat"),
    ],
    23: [
        (53698604, "sub-23_ses-03_task-MI.mat"),
        (53698613, "sub-23_ses-05_task-MI.mat"),
    ],
    24: [
        (53698616, "sub-24_ses-01_task-MI.mat"),
        (53698496, "sub-24_ses-02_task-MI.mat"),
        (53698625, "sub-24_ses-03_task-MI.mat"),
        (53698631, "sub-24_ses-04_task-MI.mat"),
        (53698394, "sub-24_ses-05_task-MI.mat"),
    ],
    25: [
        (53698637, "sub-25_ses-01_task-MI.mat"),
        (53698640, "sub-25_ses-02_task-MI.mat"),
    ],
    26: [
        (54029900, "sub-26_ses-04_task-MI.mat"),
    ],
    27: [
        (54029915, "sub-27_ses-03_task-MI.mat"),
    ],
    28: [
        (54029912, "sub-28_ses-02_task-MI.mat"),
        (53698547, "sub-28_ses-03_task-MI.mat"),
        (53698550, "sub-28_ses-04_task-MI.mat"),
    ],
    29: [
        (54029903, "sub-29_ses-02_task-MI.mat"),
        (54029891, "sub-29_ses-03_task-MI.mat"),
        (54029897, "sub-29_ses-04_task-MI.mat"),
        (54029894, "sub-29_ses-05_task-MI.mat"),
    ],
    30: [
        (53698577, "sub-30_ses-03_task-MI.mat"),
        (53698448, "sub-30_ses-04_task-MI.mat"),
    ],
}
# fmt: on

# Event labels in raw data row 33 (from events.json, label -> MOABB code).
# README and events.json disagree on left/right mapping.
# events.json: 2=right_leg_MI_start, 5=left_leg_MI_start
# README: 2=left_leg_MI_start, 5=right_leg_MI_start
# We follow events.json (generated programmatically) and note the ambiguity.
_RAW_EVENT_MAP = {
    5: 1,  # left_leg_MI_start -> left_leg (1)
    2: 2,  # right_leg_MI_start -> right_leg (2)
}


class Zuo2025(BaseDataset):
    """Lower-limb MI dataset for knee pain patients from Zuo et al. 2025.

    Dataset from [1]_.

    This dataset contains 30-channel EEG recordings from 30 knee pain
    patients performing left and right leg motor imagery. Recorded with
    ZhenTec EEG system at 500 Hz using the 10-20 montage.

    Each subject completed 5 sessions (every 2 days), with 100 trials
    per session (50 left leg, 50 right leg). Trial structure: 4 s MI,
    4 s rest, ~2 s gap.

    .. note::

       12 of 30 subjects have incomplete raw session files on Figshare
       (32 of 150 sessions missing). The adapter loads all available
       sessions per subject. Each session file is ~100 MB.

    .. note::

       There is a known inconsistency between the dataset README and
       events.json regarding the left/right leg label mapping. This
       adapter follows the events.json mapping (generated
       programmatically).

    References
    ----------
    .. [1] Zuo, C., Yin, Y., Wang, H., et al. (2025). Enhancing
           classification of a large lower-limb motor imagery EEG
           dataset for BCI in knee pain patients. Scientific Data,
           12, 1451. https://doi.org/10.1038/s41597-025-05767-2

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=30,
            channel_types={"eeg": 30},
            hardware="ZhenTec EEG system",
            reference="CPz",
            ground="FPz",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
            montage="standard_1005",
        ),
        participants=ParticipantMetadata(
            n_subjects=30,
            gender={"female": 12, "male": 18},
            age_mean=33.5,
            age_min=24,
            age_max=45,
            health_status="knee pain patients",
            clinical_population="knee_pain",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_leg", "right_leg"],
            events=dict(_EVENTS),
            trial_duration=4.0,
            study_design=(
                "2-class lower-limb MI (left/right leg flexion/"
                "extension). 5 sessions, 100 trials per session."
            ),
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            synchronicity="cue-based",
            feedback_type="none",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-05767-2",
            investigators=[
                "Chongwen Zuo",
                "Yi Yin",
                "Haochong Wang",
                "Zhiyang Zheng",
                "Xiaoyan Ma",
                "Yuan Yang",
                "Jue Wang",
                "Shan Wang",
                "Zi-gang Huang",
                "Chaoqun Ye",
            ],
            institution="Air Force Medical Center, Beijing",
            country="CN",
            data_url=("https://figshare.com/articles/dataset/" "28740260"),
            publication_year=2025,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=5,
        runs_per_session=1,
        tags=Tags(
            pathology=["Knee Pain"],
            modality=["Motor"],
            type=["Clinical", "Motor Imagery"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["left_leg", "right_leg"],
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=500,
            n_trials_per_class={"left_leg": 250, "right_leg": 250},
            trials_context="5 sessions x 100 trials (50 left + 50 right)",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+LDA", "FBCSP+SVM", "EEGNet", "OTFWRGD"],
            feature_extraction=["CSP", "FBCSP", "deep_learning", "Riemannian_geometry"],
            frequency_bands={
                "alpha_mu": [8.0, 15.0],
                "beta": [15.0, 30.0],
            },
            spatial_filters=["CSP", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"],
            environment="clinical",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 31)),
            sessions_per_subject=5,
            events=dict(_EVENTS),
            code="Zuo2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-05767-2",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")

        path = dl.get_dataset_path("Zuo2025", path)
        basepath = Path(path) / "MNE-zuo2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        file_list = _RAW_FILES.get(subject, [])
        paths = []
        for file_id, fname in file_list:
            local_path = basepath / fname
            if not local_path.exists() or force_update:
                url = f"https://ndownloader.figshare.com/files/{file_id}"
                dl_path = dl.data_dl(url, "Zuo2025", str(basepath), force_update, verbose)
                dl_path = Path(dl_path)
                if dl_path != local_path:
                    dl_path.rename(local_path)
            paths.append(str(local_path))

        return paths if paths else [str(basepath)]

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        path = dl.get_dataset_path("Zuo2025", None)
        basepath = Path(path) / "MNE-zuo2025-data"

        file_list = _RAW_FILES.get(subject, [])
        if not file_list:
            raise FileNotFoundError(
                f"No raw session files available for subject {subject}"
            )

        # Ensure files are downloaded.
        self.data_path(subject)

        sessions = {}
        for file_id, fname in file_list:
            local_path = basepath / fname
            if not local_path.exists():
                log.warning("Missing %s for subject %d", fname, subject)
                continue

            # Extract session number from filename.
            # Format: sub-XX_ses-YY_task-MI.mat
            ses_str = fname.split("ses-")[1].split("_")[0]

            try:
                raw = self._load_raw_mat(local_path)
                sessions[ses_str] = {"0": raw}
            except Exception as e:
                log.warning("Failed to load %s: %s", fname, e)

        if not sessions:
            raise FileNotFoundError(f"No loadable data for subject {subject}")
        return sessions

    def _load_raw_mat(self, mat_path):
        """Load a raw session MAT file and return MNE Raw."""
        mat = loadmat(str(mat_path), squeeze_me=True)

        # The raw data is stored as 'EEG' (or 'data' for some subjects)
        # with shape [33 x N].
        # Rows 1-32: EEG channels (30 EEG + 2 EOG)
        # Row 33: Event labels
        if "EEG" in mat:
            eeg_raw = mat["EEG"]
        elif "data" in mat:
            eeg_raw = mat["data"]
        else:
            raise KeyError(
                f"Expected 'EEG' or 'data' key in {mat_path}, "
                f"got {[k for k in mat.keys() if not k.startswith('__')]}"
            )
        if eeg_raw.shape[0] > eeg_raw.shape[1]:
            eeg_raw = eeg_raw.T  # Ensure shape is [channels x samples]

        n_rows, n_samples = eeg_raw.shape

        # Extract EEG data (first 30 channels, skip 2 EOG).
        eeg_data = eeg_raw[:30, :]

        # Extract event channel (last row).
        event_ch = eeg_raw[n_rows - 1, :]

        # Build stim channel from event labels.
        stim = np.zeros((1, n_samples))
        # Find transitions where MI starts (labels 2 or 5).
        for i in range(1, n_samples):
            label = int(event_ch[i])
            if label in _RAW_EVENT_MAP and int(event_ch[i - 1]) != label:
                stim[0, i] = _RAW_EVENT_MAP[label]

        # Apply Common Average Reference (CAR) to match the paper's
        # preprocessing. The shared extraction files have CAR applied;
        # without it, central/posterior channels lose spatial
        # discriminability for lower-limb MI.
        eeg_data = eeg_data.astype(np.float64)
        eeg_data = eeg_data - eeg_data.mean(axis=0, keepdims=True)

        # Scale to volts (data is in microvolts).
        if np.abs(eeg_data).max() > 1e-3:
            eeg_data = eeg_data * 1e-6

        ch_types = ["eeg"] * 30 + ["stim"]
        info = mne.create_info(
            ch_names=list(_CH_NAMES) + ["STI"],
            ch_types=ch_types,
            sfreq=_SFREQ,
        )
        full_data = np.concatenate([eeg_data, stim], axis=0)
        raw = mne.io.RawArray(data=full_data, info=info, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")

        return raw
