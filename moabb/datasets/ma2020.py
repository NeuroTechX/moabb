"""Multi-channel EEG recording during motor imagery of different joints.

Ma et al. (2020), Scientific Data.
DOI: 10.1038/s41597-020-0535-2
"""

from pathlib import Path

from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


_DATAVERSE_API = "https://dataverse.harvard.edu/api/access/datafile/"

# Harvard Dataverse file IDs for sub-001..sub-025 derivative .mat files.
# Extracted from the Dataverse API metadata for doi:10.7910/DVN/RBN3XG.
_FILE_IDS = {
    1: 3658080,
    2: 3658081,
    3: 3658082,
    4: 3658090,
    5: 3658086,
    6: 3658089,
    7: 3658091,
    8: 3658087,
    9: 3658085,
    10: 3658088,
    11: 3658111,
    12: 3658103,
    13: 3658100,
    14: 3658107,
    15: 3658104,
    16: 3658101,
    17: 3658105,
    18: 3658108,
    19: 3658102,
    20: 3658109,
    21: 3658110,
    22: 3658106,
    23: 3658099,
    24: 3658078,
    25: 3658079,
}

# 62 EEG channels after removing HEO, M2, VEO, EMG1, EMG2 from the original
# 67-channel recording.  Order matches the .mat derivatives.
# fmt: off
MA2020_CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz",
    "O2", "CB2",
]
# fmt: on

# Per-subject demographics from participants.tsv (Harvard Dataverse BIDS layout).
# (sex, age) tuples indexed by 1-based subject number.
_DEMOGRAPHICS = {
    1: ("male", 26),
    2: ("male", 27),
    3: ("male", 24),
    4: ("female", 25),
    5: ("male", 25),
    6: ("male", 24),
    7: ("female", 26),
    8: ("male", 24),
    9: ("female", 24),
    10: ("male", 25),
    11: ("male", 27),
    12: ("male", 26),
    13: ("male", 29),
    14: ("male", 26),
    15: ("female", 25),
    16: ("female", 24),
    17: ("female", 26),
    18: ("male", 26),
    19: ("female", 25),
    20: ("male", 28),
    21: ("male", 25),
    22: ("male", 27),
    23: ("male", 26),
    24: ("male", 23),
    25: ("male", 25),
}


class Ma2020(BaseDataset):
    """Motor imagery dataset from Ma et al. 2020.

    Dataset from [1]_.

    This dataset contains 62-channel EEG recordings from 25 healthy subjects
    (18 males, 7 females, aged 23-29 years, mean age 25.56) performing motor
    imagery of two different joints of the right upper limb: hand and elbow.
    All subjects were right-handed and had no prior BCI experience.

    Data were collected over 3 separate days (Monday, Wednesday, Friday of the
    same week). Each day consisted of 5 MI sessions and 2 resting-state
    sessions, yielding 15 MI sessions per subject. Each MI session contains
    40 trials (20 hand, 20 elbow) with randomized order.

    Trial structure (8 s total):
        - 0-2 s: fixation (white circle)
        - 2-3 s: attention cue (red circle)
        - 3-7 s: motor imagery period (4 s, "Hand" or "Elbow" displayed)
        - 7-8 s: break

    The derivative .mat files used here contain preprocessed and epoched data
    at 200 Hz (62 EEG channels). Preprocessing included: removal of
    non-EEG channels (HEO, M2, VEO, EMG1, EMG2), common average
    re-referencing, 0.1-100 Hz bandpass filtering, baseline removal,
    EOG artifact correction (SOBI), and downsampling to 200 Hz.

    Warnings
    --------
    This dataset includes channels 'CB1' and 'CB2' which are not part of
    the standard 10-20 montage. They are treated as standard EEG channels
    with ``on_missing="ignore"`` for montage setting.

    The data is downloaded from Harvard Dataverse. Each subject's .mat file
    is approximately 190 MB (4.8 GB total for all 25 subjects).

    References
    ----------
    .. [1] X. Ma, S. Qiu, C. Du, J. Xing, and H. He, "Multi-channel EEG
       recording during motor imagery of different joints from the same limb,"
       Scientific Data, vol. 7, no. 1, p. 191, 2020.
       DOI: 10.1038/s41597-020-0535-2
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=200.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="standard_1005",
            hardware="Neuroscan SynAmps2",
            sensors=MA2020_CH_NAMES,
            line_freq=50.0,
            reference="common average",
            ground="AFz",
            impedance_threshold_kohm=5,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=False,
                eog_channels=0,
                has_emg=False,
                emg_channels=0,
                other_physiological=None,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender={"male": 18, "female": 7},
            age_mean=25.56,
            age_min=23,
            age_max=29,
            handedness={"right": 25},
            bci_experience="naive",
            ages=[d[1] for d in [_DEMOGRAPHICS[i] for i in range(1, 26)]],
            sexes=[d[0] for d in [_DEMOGRAPHICS[i] for i in range(1, 26)]],
            handedness_list=["right"] * 25,
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "right_elbow"],
            trial_duration=4.0,
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            task_type="motor_imagery_same_limb",
            feedback_type="none",
            has_training_test_split=False,
            instructions=(
                "Subjects were asked to concentrate on performing the "
                "indicated motor imagery task (right hand or right elbow) "
                "using kinesthetic, not visual, motor imagery while "
                "avoiding any motion during imagination."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-020-0535-2",
            investigators=[
                "Xuelin Ma",
                "Shuang Qiu",
                "Changde Du",
                "Jiezhen Xing",
                "Huiguang He",
            ],
            senior_author="Huiguang He",
            institution="Chinese Academy of Sciences",
            country="CN",
            repository="Harvard Dataverse",
            data_url="https://doi.org/10.7910/DVN/RBN3XG",
            license="CC0 1.0",
            publication_year=2020,
            institution_department="Institute of Automation",
            ethics_approval=[
                "Ethics Committee of the Institute of Automation, Chinese Academy of Sciences"
            ],
            funding=[
                "National Key Research and Development Plan of China (No. 2017YFB1002502)",
                "National Natural Science Foundation of China (No. 61976209)",
                "National Natural Science Foundation of China (No. 61906188)",
            ],
            keywords=[
                "motor imagery",
                "EEG",
                "BCI",
                "same limb",
                "hand",
                "elbow",
            ],
        ),
        sessions_per_subject=15,
        runs_per_session=1,
        data_processed=True,
        file_format="MAT",
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            preprocessing_steps=[
                "channel removal (HEO, M2, VEO, EMG1, EMG2)",
                "common average re-reference",
                "bandpass filter 0.1-100 Hz",
                "baseline removal",
                "EOG correction (SOBI)",
                "downsampling to 200 Hz",
            ],
            highpass_hz=0.1,
            lowpass_hz=100.0,
            re_reference="common average",
            downsampled_to_hz=200.0,
            artifact_methods=["SOBI (EOG correction)"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["right_hand", "right_elbow"],
            cue_duration_s=1.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=600,
            n_trials_per_class={"right_hand": 300, "right_elbow": 300},
            n_blocks=15,
            trials_context=(
                "3 days x 5 MI sessions/day = 15 sessions, "
                "40 trials/session (20 hand + 20 elbow)"
            ),
        ),
        bci_application=BCIApplicationMetadata(
            environment="laboratory",
            online_feedback=False,
            applications=["motor rehabilitation"],
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["motor"],
            type=["imagery"],
        ),
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=15,
            events=dict(right_hand=1, right_elbow=2),
            code="Ma2020",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-020-0535-2",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Loads the derivative .mat file containing preprocessed epoched data
        and reconstructs continuous Raw objects for each of the 15 MI sessions.
        """
        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True)

        # task_data: (15, 40, 62, 800) = [sessions, trials, channels, samples]
        # task_label: (15, 40) = event codes (1=hand, 2=elbow)
        task_data = mat["task_data"]
        task_label = mat["task_label"]

        # Ensure shapes are correct (squeeze_me may remove singletons)
        if task_label.ndim == 3:
            task_label = task_label.squeeze(-1)

        n_sessions = task_data.shape[0]

        # Set subject demographics
        sex, age = _DEMOGRAPHICS.get(subject, (None, None))
        _sex_map = {"male": 1, "female": 2}

        sessions = {}
        for sess_idx in range(n_sessions):
            # (40, 62, 800)
            epoch_data = task_data[sess_idx]
            labels = task_label[sess_idx].astype(int)

            raw = build_raw_from_epochs(
                epoch_data,
                MA2020_CH_NAMES,
                200,
                labels,
                "standard_1005",
            )

            # Attach per-subject demographics
            if sex is not None:
                raw.info["subject_info"] = {
                    "sex": _sex_map.get(sex, 0),
                    "his_id": str(subject),
                }
            if age is not None:
                raw._moabb_subject_age = age

            sessions[str(sess_idx)] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        data_dir.mkdir(parents=True, exist_ok=True)

        mat_file = data_dir / f"sub-{subject:03d}_task-motorimagery_eeg.mat"
        if mat_file.exists() and not force_update:
            return str(mat_file)

        # Download from Harvard Dataverse using the file ID
        file_id = _FILE_IDS[subject]
        url = f"{_DATAVERSE_API}{file_id}"
        dl_path = Path(dl.data_dl(url, sign, path, force_update, verbose))

        # Move the downloaded file to the canonical location
        if dl_path.resolve() != mat_file.resolve():
            import shutil

            shutil.move(str(dl_path), str(mat_file))

        return str(mat_file)
