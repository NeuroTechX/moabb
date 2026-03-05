"""40-Class Beta-Range SSVEP Speller Dataset.

Kim et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-06032-2
"""

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
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
from .utils import FIGSHARE_DL_URL, build_raw_from_epochs


# Figshare file IDs for raw_eeg_ssvep_subj_NN.mat
# fmt: off
_SSVEP_FILE_IDS = {
    1: 53705183, 2: 53705180, 3: 53705123, 4: 53705168, 5: 53705171,
    6: 53705108, 7: 53705132, 8: 53705153, 9: 53705126, 10: 53705156,
    11: 53705129, 12: 53705099, 13: 53705177, 14: 53705150, 15: 53707388,
    16: 53705135, 17: 53705162, 18: 53705087, 19: 53705114, 20: 53705174,
    21: 53705075, 22: 53705117, 23: 53707391, 24: 53705078, 25: 53705165,
    26: 53705090, 27: 53705144, 28: 53705120, 29: 53707394, 30: 53705141,
    31: 53705207, 32: 53705159, 33: 53705138, 34: 53705111, 35: 53705189,
    36: 53705195, 37: 53705186, 38: 53705198, 39: 53705192, 40: 53705201,
}
# fmt: on

_EVENTS = {
    "14": 1,
    "15": 2,
    "16": 3,
    "17": 4,
    "18": 5,
    "19": 6,
    "20": 7,
    "21": 8,
    "14.2": 9,
    "15.2": 10,
    "16.2": 11,
    "17.2": 12,
    "18.2": 13,
    "19.2": 14,
    "20.2": 15,
    "21.2": 16,
    "14.4": 17,
    "15.4": 18,
    "16.4": 19,
    "17.4": 20,
    "18.4": 21,
    "19.4": 22,
    "20.4": 23,
    "21.4": 24,
    "14.6": 25,
    "15.6": 26,
    "16.6": 27,
    "17.6": 28,
    "18.6": 29,
    "19.6": 30,
    "20.6": 31,
    "21.6": 32,
    "14.8": 33,
    "15.8": 34,
    "16.8": 35,
    "17.8": 36,
    "18.8": 37,
    "19.8": 38,
    "20.8": 39,
    "21.8": 40,
}
# fmt: on


class Kim2025BetaRange(BaseDataset):
    """40-class beta-range SSVEP speller dataset.

    Dataset from [1]_.

    This dataset contains 33-channel EEG (31 scalp + 2 mastoid references)
    recorded from 40 healthy subjects (25 males, 15 females, aged 20-35)
    performing a 40-target SSVEP-BCI speller task using beta-range frequencies
    (14.0-21.8 Hz, 0.2 Hz step). The JFPM approach was used with phase
    differences of 0.5*pi between adjacent stimuli.

    Each subject completed 6 blocks of 40 trials. Trial structure was 1.5 s
    rest, 0.5 s cue, and 5.0 s SSVEP stimulation. EEG was recorded at 1024 Hz
    with a BioSemi ActiveTwo system. Stored epochs span [-2000, 5000] ms
    relative to stimulus onset (7168 samples at 1024 Hz). The event marker is
    placed at stimulus onset (sample 2048), and interval=[0.0, 5.0] extracts
    the 5 s stimulation window.

    The stimuli were presented in a 5x8 matrix on a 120 Hz monitor.

    References
    ----------
    .. [1] H. Kim, K. Won, M. Ahn, and S. C. Jun, "A 40-class SSVEP speller
       dataset: beta range stimulation for low-fatigue BCI applications,"
       Scientific Data, vol. 12, p. 1751, 2025.
       DOI: 10.1038/s41597-025-06032-2
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1024.0,
            n_channels=33,
            channel_types={"eeg": 31, "misc": 2},
            montage="standard_1005",
            hardware="BioSemi ActiveTwo",
            reference="CMS/DRL",
            line_freq=60.0,
            sensor_type="active",
            electrode_type="wet",
            electrode_material="Ag/AgCl",
            ground="CMS/DRL near Pz",
            impedance_threshold_kohm=5,
            software="OpenViBE",
            cap_manufacturer="BioSemi",
        ),
        participants=ParticipantMetadata(
            n_subjects=40,
            health_status="healthy",
            gender={"male": 25, "female": 15},
            age_mean=22.8,
            age_min=20,
            age_max=35,
            age_std=3.34,
            bci_experience="3 of 40 had prior SSVEP-BCI experience",
            # Per-subject sex from questionnaire_en.xlsx; S24 inferred as
            # female from paper total (25M/15F).
            # fmt: off
            sexes=[
                "male", "male", "male", "female", "male", "female", "male",
                "male", "male", "male", "male", "male", "male", "female",
                "male", "female", "female", "male", "male", "female", "male",
                "male", "female", "female", "female", "female", "female",
                "male", "female", "male", "female", "male", "male", "male",
                "male", "male", "male", "female", "female", "male",
            ],
            # Per-subject handedness from questionnaire_en.xlsx; S24 unknown.
            handedness_list=[
                "right", "right", "right", "right", "right", "right", "right",
                "right", "right", "right", "right", "right", "right", "right",
                "right", "right", "right", "right", "left", "right", "right",
                "right", "right", None, "right", "right", "right", "right",
                "right", "right", "right", "right", "left", "right", "right",
                "left", "right", "right", "right", "left",
            ],
            # fmt: on
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            n_classes=40,
            trial_duration=5.0,
            stimulus_type="JFPM visual flicker",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            task_type="speller",
            feedback_type="none",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-06032-2",
            investigators=["Heegyu Kim", "Kyungho Won", "Minkyu Ahn", "Sung Chan Jun"],
            senior_author="Sung Chan Jun",
            institution="Gwangju Institute of Science and Technology",
            country="KR",
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.28806815.v2",
            license="CC BY 4.0",
            publication_year=2025,
            institution_department="School of Electrical Engineering and Computer Science, GIST",
            ethics_approval=["GIST IRB, No. 20211201-HR-64-02-04"],
            keywords=[
                "SSVEP",
                "BCI",
                "beta range",
                "visual fatigue",
                "40-class speller",
                "JFPM",
                "EEG",
            ],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[14.0 + i * 0.2 for i in range(40)],
            frequency_resolution_hz=0.2,
        ),
        data_structure=DataStructureMetadata(
            n_blocks=6,
            n_trials=240,
        ),
        bci_application=BCIApplicationMetadata(
            environment="lab",
            online_feedback=False,
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["visual"],
            type=["perception"],
        ),
        sessions_per_subject=6,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 41)),
            sessions_per_subject=6,
            events=_EVENTS,
            code="Kim2025BetaRange",
            interval=[0.0, 5.0],
            paradigm="ssvep",
            doi="10.1038/s41597-025-06032-2",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across all 6 blocks."""
        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True, simplify_cells=True)
        eeg_struct = mat["eeg"]

        data = eeg_struct["data"]  # shape: (33, 7168, 40, 6)
        ch_names_raw = list(eeg_struct["chan_locs"])
        srate = int(eeg_struct["srate"])  # 1024
        n_classes = data.shape[2]  # 40
        n_blocks = data.shape[3]  # 6

        # Epoch window is [-2000, 5000] ms; stimulus onset is at +2000 ms
        onset_sample = int(round(2.0 * srate))  # sample 2048

        # Normalize channel names to match MNE standard_1005
        ch_names = _normalize_ch_names(ch_names_raw)
        ch_types = _infer_ch_types(ch_names)
        event_ids = np.arange(1, n_classes + 1)

        sessions = {}
        for block_idx in range(n_blocks):
            block_data = data[:, :, :, block_idx]  # (33, 7168, 40)
            block_data = np.transpose(block_data, (2, 0, 1))  # (40, 33, 7168)
            raw = build_raw_from_epochs(
                block_data,
                ch_names,
                srate,
                event_ids,
                "standard_1005",
                ch_types=ch_types,
                onset_sample=onset_sample,
            )
            sessions[str(block_idx)] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")
        file_id = _SSVEP_FILE_IDS[subject]
        url = f"{FIGSHARE_DL_URL}{file_id}"
        return dl.data_dl(url, self.code, path, force_update, verbose)


def _normalize_ch_names(ch_names):
    """Normalize channel names from .mat file to match MNE conventions.

    The .mat files use uppercase midline names (e.g. 'CZ', 'POZ', 'PZ',
    'CPZ', 'OZ') which must be converted to mixed case ('Cz', 'POz',
    'Pz', 'CPz', 'Oz') for MNE standard_1005 montage compatibility.
    """
    # Map uppercase midline channels to MNE mixed-case convention
    mapping = {
        "CZ": "Cz",
        "PZ": "Pz",
        "OZ": "Oz",
        "CPZ": "CPz",
        "POZ": "POz",
    }
    return [mapping.get(ch, ch) for ch in ch_names]


def _infer_ch_types(ch_names):
    """Infer channel types (31 EEG + 2 mastoid misc channels)."""
    mastoid_aliases = {"M1", "M2", "A1", "A2", "TP9", "TP10"}
    ch_types = ["misc" if ch.upper() in mastoid_aliases else "eeg" for ch in ch_names]
    if ch_types.count("misc") != 2 and len(ch_types) >= 2:
        # Fall back to the dataset convention where the last two channels are mastoids.
        ch_types = ["eeg"] * len(ch_names)
        ch_types[-2:] = ["misc", "misc"]
    return ch_types
