"""Multi-channel EEG recording during motor imagery of different joints.

Ma et al. (2020), Scientific Data.
DOI: 10.1038/s41597-020-0535-2
"""

import logging
from pathlib import Path

import mne

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
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
    SignalProcessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

_DATAVERSE_API = "https://dataverse.harvard.edu/api/access/datafile/"

# Harvard Dataverse file IDs for raw .cnt files in sourcedata/.
# Each subject has 15 MI sessions (ses-01..ses-15).
# Keyed by subject number; values are lists of 15 file IDs in session order.
# fmt: off
_CNT_FILE_IDS = {
    1: [3658114, 3658115, 3658116, 3658118, 3658117, 3658123, 3658122, 3658126, 3658125, 3658124, 3658131, 3658130, 3658127, 3658128, 3658129],
    2: [3658148, 3658151, 3658146, 3658156, 3658143, 3658144, 3658145, 3658157, 3658159, 3658147, 3658153, 3658150, 3658152, 3658154, 3658141],
    3: [3658801, 3658804, 3658803, 3658805, 3658802, 3658807, 3658809, 3658808, 3658811, 3658810, 3658813, 3658812, 3658816, 3658814, 3658815],
    4: [3658837, 3658833, 3658830, 3658836, 3658832, 3658835, 3658831, 3658834, 3658839, 3658841, 3658844, 3658838, 3658843, 3658840, 3658842],
    5: [3659118, 3659119, 3659120, 3659115, 3659117, 3659114, 3659116, 3659113, 3659121, 3659125, 3659126, 3659123, 3659124, 3659127, 3659122],
    6: [3659144, 3659138, 3659141, 3659139, 3659140, 3659142, 3659143, 3659137, 3659147, 3659149, 3659151, 3659150, 3659145, 3659146, 3659148],
    7: [3659159, 3659158, 3659165, 3659161, 3659160, 3659163, 3659164, 3659162, 3659156, 3659157, 3659166, 3659172, 3659167, 3659171, 3659170],
    8: [3659176, 3659175, 3659183, 3659181, 3659184, 3659180, 3659179, 3659182, 3659177, 3659178, 3659201, 3659195, 3659193, 3659194, 3659196],
    9: [3659207, 3659208, 3659205, 3659202, 3659206, 3659210, 3659204, 3659211, 3659209, 3659203, 3659221, 3659219, 3659226, 3659222, 3659224],
    10: [3659230, 3659227, 3659236, 3659231, 3659229, 3659235, 3659233, 3659234, 3659232, 3659228, 3659243, 3659239, 3659241, 3659238, 3659244],
    11: [3659255, 3659251, 3659248, 3659252, 3659250, 3659254, 3659253, 3659247, 3659246, 3659249, 3659263, 3659260, 3659264, 3659265, 3659261],
    12: [3659274, 3659268, 3659272, 3659270, 3659267, 3659271, 3659275, 3659269, 3659266, 3659273, 3659278, 3659279, 3659282, 3659284, 3659280],
    13: [3659294, 3659291, 3659289, 3659295, 3659297, 3659288, 3659292, 3659290, 3659296, 3659293, 3659303, 3659300, 3659299, 3659305, 3659306],
    14: [3659310, 3659314, 3659311, 3659317, 3659315, 3659313, 3659312, 3659309, 3659318, 3659316, 3659327, 3659322, 3659319, 3659320, 3659326],
    15: [3659331, 3659340, 3659338, 3659336, 3659334, 3659333, 3659332, 3659335, 3659337, 3659339, 3659347, 3659344, 3659348, 3659342, 3659346],
    16: [3659357, 3659354, 3659356, 3659359, 3659363, 3659361, 3659358, 3659360, 3659355, 3659362, 3659367, 3659370, 3659368, 3659364, 3659365],
    17: [3659390, 3659389, 3659385, 3659387, 3659388, 3659386, 3659392, 3659393, 3659391, 3659394, 3659407, 3659409, 3659403, 3659410, 3659405],
    18: [3659412, 3659415, 3659419, 3659416, 3659420, 3659414, 3659418, 3659411, 3659413, 3659417, 3659426, 3659423, 3659422, 3659428, 3659425],
    19: [3659442, 3659436, 3659439, 3659435, 3659443, 3659437, 3659438, 3659441, 3659440, 3659444, 3659461, 3659457, 3659454, 3659460, 3659456],
    20: [3659472, 3659468, 3659475, 3659469, 3659474, 3659473, 3659467, 3659470, 3659476, 3659471, 3659485, 3659484, 3659483, 3659478, 3659486],
    21: [3659493, 3659487, 3659494, 3659491, 3659492, 3659488, 3659496, 3659495, 3659490, 3659489, 3659499, 3659621, 3659500, 3659507, 3659503],
    22: [3659514, 3659517, 3659515, 3659511, 3659509, 3659518, 3659513, 3659510, 3659516, 3659512, 3659525, 3659528, 3659532, 3659530, 3659526],
    23: [3659536, 3659539, 3659541, 3659542, 3659543, 3659540, 3659537, 3659538, 3659544, 3659535, 3659553, 3659550, 3659552, 3659548, 3659549],
    24: [3659566, 3659561, 3659562, 3659569, 3659560, 3659567, 3659565, 3659563, 3659564, 3659568, 3659577, 3659573, 3659578, 3659576, 3659570],
    25: [3659584, 3659580, 3659582, 3659588, 3659587, 3659583, 3659581, 3659586, 3659585, 3659579, 3659596, 3659592, 3659593, 3659595, 3659590],
}
# fmt: on

# 62 EEG channels (after the adapter drops HEO, M2, VEO from the raw 65-ch).
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

# Raw .cnt files have uppercase names; 10 need case correction for standard_1005.
_CH_RENAME = {
    "FP1": "Fp1",
    "FPZ": "Fpz",
    "FP2": "Fp2",
    "FZ": "Fz",
    "FCZ": "FCz",
    "CZ": "Cz",
    "CPZ": "CPz",
    "PZ": "Pz",
    "POZ": "POz",
    "OZ": "Oz",
}

# Non-EEG channels to set type on (then drop).
_AUX_TYPES = {"HEO": "eog", "VEO": "eog", "M2": "misc"}

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

    Raw Neuroscan ``.cnt`` files are loaded from the ``sourcedata/`` of the
    BIDS archive on Harvard Dataverse (one file per session, ~87 MB each,
    1000 Hz, 65 channels). Auxiliary channels (HEO, VEO, M2) are dropped.

    .. note::

       Channels CB1 and CB2 (cerebellum) are not part of the standard
       10-05 montage and are set with ``on_missing="ignore"``.

    References
    ----------
    .. [1] X. Ma, S. Qiu, C. Du, J. Xing, and H. He, "Multi-channel EEG
       recording during motor imagery of different joints from the same limb,"
       Scientific Data, vol. 7, no. 1, p. 191, 2020.
       DOI: 10.1038/s41597-020-0535-2
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="standard_1005",
            hardware="Neuroscan SynAmps2",
            sensors=MA2020_CH_NAMES,
            line_freq=50.0,
            ground="AFz",
            impedance_threshold_kohm=5,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["horizontal", "vertical"],
                has_emg=False,
                emg_channels=0,
                other_physiological=["M2"],
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
            events={"right_hand": 1, "right_elbow": 2},
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
                "Junfeng Xing",
                "Huiguang He",
            ],
            senior_author="Huiguang He",
            institution="Chinese Academy of Sciences",
            country="CN",
            repository="Harvard Dataverse",
            data_url="https://doi.org/10.7910/DVN/RBN3XG",
            license="CC-BY-4.0",
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
            keywords=["motor imagery", "EEG", "BCI", "same limb", "hand", "elbow"],
        ),
        sessions_per_subject=15,
        runs_per_session=1,
        data_processed=False,
        file_format="CNT",
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
        signal_processing=SignalProcessingMetadata(
            classifiers=["FBCSP+SVM"],
            feature_extraction=["FBCSP"],
            frequency_bands={"alpha": [8.0, 13.0], "beta": [20.0, 25.0]},
            spatial_filters=["CAR", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold", cv_folds=5, evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            environment="laboratory",
            online_feedback=False,
            applications=["motor_rehabilitation", "prosthetic_control"],
        ),
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["imagery"]),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=15,
            events={"right_hand": 1, "right_elbow": 2},
            code="Ma2020",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-020-0535-2",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    # Annotation codes in raw .cnt -> MOABB event names.
    _ANNOT_MAP = {"1": "right_hand", "2": "right_elbow"}

    def _get_single_subject_data(self, subject):
        """Return data for a single subject from raw .cnt files.

        Each of the 15 MI sessions is a separate Neuroscan .cnt file with
        40 embedded event annotations (20 hand + 20 elbow).
        """
        subj_dir = Path(self.data_path(subject))

        sex, age = _DEMOGRAPHICS.get(subject, (None, None))
        _sex_map = {"male": 1, "female": 2}

        sessions = {}
        for sess_idx in range(15):
            cnt_name = (
                f"sub-{subject:03d}_ses-{sess_idx + 1:02d}_task-motorimagery_eeg.cnt"
            )
            cnt_path = subj_dir / cnt_name
            if not cnt_path.exists():
                log.warning("Missing %s", cnt_path)
                continue

            raw = mne.io.read_raw_cnt(str(cnt_path), preload=True, verbose=False)

            # Fix channel name case for standard_1005 montage
            raw.rename_channels(
                {ch: _CH_RENAME[ch] for ch in raw.ch_names if ch in _CH_RENAME}
            )

            # Set non-EEG channel types
            raw.set_channel_types(
                {ch: t for ch, t in _AUX_TYPES.items() if ch in raw.ch_names}
            )
            # Drop aux channels only when return_all_modalities is False
            if not self.return_all_modalities:
                raw.drop_channels([ch for ch in _AUX_TYPES if ch in raw.ch_names])

            # Rename event annotations
            raw.annotations.rename(self._ANNOT_MAP)

            # Attach demographics
            if sex is not None:
                raw.info["subject_info"] = {
                    "sex": _sex_map.get(sex, 0),
                    "his_id": str(subject),
                }

            sessions[str(sess_idx)] = {"0": raw}

        if not sessions:
            raise FileNotFoundError(
                f"No .cnt files found for subject {subject} in {subj_dir}"
            )
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"sub-{subject:03d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Download each session's .cnt if not already present
        file_ids = _CNT_FILE_IDS[subject]
        for sess_idx, file_id in enumerate(file_ids):
            cnt_name = (
                f"sub-{subject:03d}_ses-{sess_idx + 1:02d}_task-motorimagery_eeg.cnt"
            )
            cnt_path = subj_dir / cnt_name
            if cnt_path.exists() and not force_update:
                continue

            url = f"{_DATAVERSE_API}{file_id}"
            dl_path = Path(dl.data_dl(url, sign, path, force_update, verbose))
            if dl_path.resolve() != cnt_path.resolve():
                import shutil

                shutil.move(str(dl_path), str(cnt_path))

        return str(subj_dir)
