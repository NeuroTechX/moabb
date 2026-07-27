"""XP1 simultaneous EEG-fMRI motor imagery / neurofeedback dataset (Lioi et al.).

Right-hand kinaesthetic motor imagery vs rest, 20 s block design, EEG recorded
inside an MR scanner. Only the EEG modality is exposed here (the fMRI NIfTI
volumes are ignored).

OpenNeuro: ds002336 (XP1). Sibling dataset XP2 is ds002338.

Data descriptor:
    Lioi, G., Cury, C., Perronnet, L., Mano, M., Bannier, E., Lecuyer, A.,
    & Barillot, C. (2019). Simultaneous MRI-EEG during a motor imagery
    neurofeedback task: an open access brain imaging dataset for multi-modal
    data integration. bioRxiv 862375. https://doi.org/10.1101/862375
"""

import logging

import mne
from mne.channels import make_standard_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# OpenNeuro dataset id and public S3 mirror (no auth required).
_OPENNEURO_ID = "ds002336"
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# Motor-imagery-vs-rest task runs, in acquisition order. Each run alternates
# 20 s rest / 20 s right-hand kinaesthetic MI blocks. The ``task-motorloc``
# run is a motor-execution localizer (marker ``S  1``) and is intentionally
# excluded to keep the paradigm pure imagery. MIpre/MIpost are absent for a
# few subjects; runs are added only when the files actually exist.
_TASKS = ["MIpre", "eegNF", "fmriNF", "eegfmriNF", "MIpost"]

# BrainVision annotation markers relevant to the MI paradigm:
#   "Stimulus/S 99" -> rest block onset
#   "Stimulus/S  2" -> task (right-hand MI) block onset
# (R128 = fMRI gradient volume marker, dropped by not matching event_id.)
_MARKER_TO_LABEL = {"Stimulus/S 99": "rest", "Stimulus/S  2": "right_hand"}

_EVENTS = {"rest": 1, "right_hand": 2}

# 63 EEG channels (channel 32 in the recording is ECG). FCz is the reference
# and AFz the ground, so neither appears among the recorded EEG channels.
# fmt: off
_EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "TP9", "TP10",
    "POz", "F1", "F2", "C1", "C2", "P1", "P2", "AF3", "AF4", "FC3",
    "FC4", "CP3", "CP4", "PO3", "PO4", "F5", "F6", "C5", "C6", "P5",
    "P6", "AF7", "AF8", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8", "FT9",
    "FT10", "Fpz", "CPz",
]
# fmt: on


class LioiXP1(BaseDataset):
    """XP1 simultaneous EEG-fMRI motor imagery / neurofeedback dataset [1]_ [2]_.

    Ten healthy subjects performed right-hand kinaesthetic motor imagery inside
    an MR scanner during a 20 s block design alternating rest and task. Data
    were acquired with a 64-channel MR-compatible Brain Products system sampled
    at 5 kHz (channel 32 is ECG; the 63 remaining channels are EEG), reference
    FCz, ground AFz.

    The protocol comprised six EEG-fMRI runs. This loader exposes the five
    motor-imagery runs that share the rest / right-hand-MI marker structure:

    - ``MIpre``    : MI without neurofeedback, 5 blocks (absent for sub 1)
    - ``eegNF``    : unimodal EEG neurofeedback, 10 blocks
    - ``fmriNF``   : unimodal fMRI neurofeedback, 10 blocks
    - ``eegfmriNF``: bimodal EEG-fMRI neurofeedback, 10 blocks
    - ``MIpost``   : MI without neurofeedback, 5 blocks (absent for sub 1)

    Each is returned as a separate run within a single session. The
    motor-execution localizer run (``task-motorloc``) is excluded. Only the EEG
    modality is loaded; the simultaneously acquired fMRI is ignored. The EEG
    still contains MR gradient and ballistocardiogram artifacts (it is delivered
    raw, unprocessed).

    Two classes are exposed: ``right_hand`` (MI task block) and ``rest``.

    References
    ----------
    .. [1] Lioi, G., Cury, C., Perronnet, L., Mano, M., Bannier, E.,
           Lecuyer, A., & Barillot, C. (2019). Simultaneous MRI-EEG during a
           motor imagery neurofeedback task: an open access brain imaging
           dataset for multi-modal data integration. bioRxiv 862375.
           https://doi.org/10.1101/862375
    .. [2] Perronnet, L., Lecuyer, A., Mano, M., Bannier, E., Lotte, F.,
           Clerc, M., & Barillot, C. (2017). Unimodal versus bimodal EEG-fMRI
           neurofeedback of a motor imagery task. Frontiers in Human
           Neuroscience, 11, 193. https://doi.org/10.3389/fnhum.2017.00193
    """

    nemar_id = "ds002336"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=5000.0,
            n_channels=63,
            channel_types={"eeg": 63, "ecg": 1},
            montage="standard_1005",
            hardware="Brain Products MR-compatible 64-channel EEG",
            reference="FCz",
            ground="AFz",
            line_freq=50.0,
            sensors=list(_EEG_CHANNELS),
            auxiliary_channels=AuxiliaryChannelsMetadata(other_physiological=["ecg"]),
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"male": 8, "female": 2},
            age_min=19.0,
            age_max=39.0,
            species="human",
            ages=[25, 27, 25, 31, 39, 36, 19, 29, 27, 26],
            sexes=[
                "male",
                "male",
                "male",
                "male",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
            ],
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            trial_duration=20.0,
            study_design=(
                "20 s block design alternating rest and right-hand kinaesthetic "
                "motor imagery, across MI-without-feedback and three "
                "neurofeedback (EEG, fMRI, bimodal) runs, acquired simultaneously "
                "with fMRI."
            ),
            feedback_type="neurofeedback",
            stimulus_type="visual (ball-to-target)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1101/862375",
            related_paper_dois=["10.3389/fnhum.2017.00193"],
            investigators=[
                "Giulia Lioi",
                "Claire Cury",
                "Lorraine Perronnet",
                "Marsel Mano",
                "Elise Bannier",
                "Anatole Lecuyer",
                "Christian Barillot",
            ],
            institution="Inria Rennes / University of Rennes",
            country="FR",
            data_url="https://openneuro.org/datasets/ds002336",
            publication_year=2019,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=5,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand"],
            imagery_duration_s=20.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=780,
            trials_context=(
                "Per subject: 3 NF runs x 10 MI blocks + 2 MI runs x 5 blocks "
                "= 40 right-hand MI blocks (30 for sub 1, missing MIpre/MIpost), "
                "each paired with a 20 s rest block. Total right-hand blocks 390, "
                "matched by 390 rest blocks -> 780 blocks."
            ),
        ),
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="LioiXP1",
            interval=[0, 20],
            paradigm="imagery",
            doi="10.1101/862375",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _subject_id(self, subject):
        """Map integer subject 1..10 to the BIDS label ``xp10N``/``xp110``."""
        return f"xp{100 + subject}"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sub = self._subject_id(subject)
        paths = []
        for task in _TASKS:
            base = f"{_S3_BASE}/sub-{sub}/eeg/sub-{sub}_task-{task}_eeg"
            # The .vhdr references its .eeg/.vmrk siblings by basename, so all
            # three must be fetched into the same local directory.
            for ext in (".eeg", ".vmrk", ".vhdr"):
                try:
                    local = dl.data_dl(
                        f"{base}{ext}", self.code, path=path, force_update=force_update
                    )
                except Exception as exc:  # noqa: BLE001
                    # MIpre/MIpost are missing for some subjects (HTTP 404).
                    if ext == ".eeg":
                        log.info("Skipping absent run %s for sub-%s (%s)", task, sub, exc)
                        break
                    raise
                if ext == ".vhdr":
                    paths.append(local)
        return paths

    def _get_single_subject_data(self, subject):
        vhdr_paths = self.data_path(subject)

        runs = {}
        for idx, task in enumerate(_TASKS):
            match = [p for p in vhdr_paths if f"task-{task}_eeg.vhdr" in p]
            if not match:
                continue
            raw = mne.io.read_raw_brainvision(match[0], preload=True, verbose=False)

            # Channel 32 is ECG; label it so it is not treated as EEG.
            if "ECG" in raw.ch_names:
                raw.set_channel_types({"ECG": "ecg"})

            # Remap BrainVision markers to class labels; unmatched markers
            # (R128 gradient, New Segment, S 1 localizer) are left untouched
            # and simply do not match event_id.
            desc = raw.annotations.description.astype("<U16")
            for marker, label in _MARKER_TO_LABEL.items():
                desc[desc == marker] = label
            raw.annotations.description = desc

            with mne.utils.use_log_level("error"):
                raw.set_montage(
                    make_standard_montage("standard_1005"), on_missing="ignore"
                )

            runs[f"{idx}{task}"] = stim_channels_with_selected_ids(raw, self.event_id)

        return {"0": runs}
