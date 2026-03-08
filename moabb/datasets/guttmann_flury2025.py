"""Eye-BCI multimodal dataset (MI paradigm).

Guttmann-Flury, Sheng, and Zhu (2025), Scientific Data.
DOI: 10.1038/s41597-025-04861-9
Data DOI: 10.7303/syn64005218
"""

import logging
from pathlib import Path

import mne
import numpy as np

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
    Tags,
)
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# Synapse project ID for programmatic download.
_SYNAPSE_PROJECT = "syn64005218"

# Event mapping for the MI paradigm.
_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
}

# 62 EEG channel names (Neuroscan 64-ch Quik-Cap minus M1/M2 mastoids).
# fmt: off
_CH_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "O1", "OZ", "O2", "CB1", "CB2",
]
# fmt: on

# Sessions per subject (variable: 1-3 sessions).
# Subjects with <3 sessions: S08,S10,S16,S17,S18,S21,S22,S23,S24,S25,
# S26,S27,S28,S29 have 1 session; S19,S30 have 2 sessions.
_SESSIONS_PER_SUBJECT = {
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 1,
    9: 3,
    10: 1,
    11: 3,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 1,
    17: 1,
    18: 1,
    19: 2,
    20: 3,
    21: 1,
    22: 1,
    23: 1,
    24: 1,
    25: 1,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 2,
    31: 3,
}


class GuttmannFlury2025(BaseDataset):
    """Eye-BCI multimodal MI dataset from Guttmann-Flury et al 2025.

    Dataset from the article *Dataset combining EEG, eye-tracking,
    and high-speed video for ocular activity analysis across BCI
    paradigms* [1]_.

    It contains EEG data from 31 subjects recorded with a 62-channel
    Neuroscan Quik-Cap + SynAmps2 at 1000 Hz. Four paradigms were
    tested (MI, ME, SSVEP, P300). This adapter loads only the
    **Motor Imagery** paradigm (2-class: left/right hand grasping).

    Each MI session has 40 trials (20 left, 20 right). Trial
    structure: 2 s fixation + 4 s imagery + 1-1.5 s rest.

    **Note**: This dataset is hosted on Synapse and requires a free
    Synapse account with an auth token. Set the environment variable
    ``SYNAPSE_AUTH_TOKEN`` before downloading. Create a token at
    https://www.synapse.org/#!PersonalAccessTokens:

    References
    ----------
    .. [1] Guttmann-Flury, E., Sheng, X., & Zhu, X. (2025). Dataset
           combining EEG, eye-tracking, and high-speed video for
           ocular activity analysis across BCI paradigms. Scientific
           Data, 12, 587. https://doi.org/10.1038/s41597-025-04861-9
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62, "eog": 1, "emg": 2},
            montage="standard_1005",
            hardware="Neuroscan Quik-Cap 65-ch, SynAmps2",
            sensor_type="Ag/AgCl",
            reference="right mastoid (M1)",
            ground="forehead",
            filters={"highpass_time_constant_s": 10},
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=31,
            health_status="healthy",
            gender={"female": 11, "male": 20},
            age_mean=29.0,
            age_min=20.0,
            age_max=57.0,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=7.5,
            study_design=(
                "Multi-paradigm BCI (MI/ME/SSVEP/P300). "
                "MI: 2-class hand grasping imagery, 40 trials/session, "
                "up to 3 sessions per subject."
            ),
            feedback_type="none",
            stimulus_type="visual rectangle cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-04861-9",
            investigators=[
                "Eva Guttmann-Flury",
                "Xinjun Sheng",
                "Xiangyang Zhu",
            ],
            institution="Shanghai Jiao Tong University",
            country="CN",
            data_url="https://www.synapse.org/Synapse:syn64005218",
            publication_year=2025,
            license="CC0",
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=2520,
            trials_context="63 sessions x 40 trials = 2520",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
        ),
        data_processed=False,
        file_format="CNT (Neuroscan)",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 32)),
            sessions_per_subject=3,
            events=dict(_EVENTS),
            code="GuttmannFlury2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-04861-9",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))

        n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)
        sessions = {}

        for sess_idx in range(1, n_sess + 1):
            cnt_name = f"MI-S-{subject:02d}-Sess-{sess_idx}.cnt"
            cnt_path = base / cnt_name

            if not cnt_path.exists():
                # Try alternative locations.
                candidates = list(base.rglob(f"*MI*S*{subject:02d}*Sess*{sess_idx}*.cnt"))
                if candidates:
                    cnt_path = candidates[0]
                else:
                    log.warning("Missing %s", cnt_name)
                    continue

            try:
                raw = mne.io.read_raw_cnt(str(cnt_path), preload=True, verbose="ERROR")

                # Map annotation descriptions to MOABB event names.
                desc = raw.annotations.description.astype(np.dtype("<15U"))
                desc[desc == "Left"] = "left_hand"
                desc[desc == "Right"] = "right_hand"
                raw.annotations.description = desc

                raw = stim_channels_with_selected_ids(raw, self.event_id)
                sessions[str(sess_idx - 1)] = {"0": raw}
            except Exception as e:
                log.warning("Failed to load %s: %s", cnt_path.name, e)

        if not sessions:
            raise FileNotFoundError(f"No MI data for subject {subject} in {base}")
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("GuttmannFlury2025", path)
        basepath = Path(path) / "MNE-guttmannflury2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        # Check if MI CNT files already exist.
        cnt_pattern = f"MI-S-{subject:02d}-Sess-*.cnt"
        existing = list(basepath.rglob(cnt_pattern))
        if existing and not force_update:
            return str(basepath)

        # Download from Synapse (requires synapseclient + auth token).
        try:
            import synapseclient  # noqa: F401
        except ImportError:
            raise ImportError(
                "The synapseclient package is required to download "
                "the GuttmannFlury2025 dataset. Install it with:\n"
                "  pip install synapseclient\n"
                "Then set SYNAPSE_AUTH_TOKEN environment variable."
            )

        import os

        token = os.environ.get("SYNAPSE_AUTH_TOKEN")
        if not token:
            raise RuntimeError(
                "SYNAPSE_AUTH_TOKEN environment variable not set. "
                "Create a Personal Access Token at "
                "https://www.synapse.org/#!PersonalAccessTokens: "
                "and set it as SYNAPSE_AUTH_TOKEN."
            )

        syn = synapseclient.login(authToken=token, silent=True)

        # Get project children to find MI files for this subject.
        log.info(
            "Downloading GuttmannFlury2025 MI data for subject %d from Synapse...",
            subject,
        )

        # Walk the Synapse project to find MI CNT files.
        # The exact folder structure on Synapse is TBD; search by filename.
        n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)
        for sess_idx in range(1, n_sess + 1):
            cnt_name = f"MI-S-{subject:02d}-Sess-{sess_idx}.cnt"
            target = basepath / cnt_name
            if target.exists() and not force_update:
                continue

            try:
                # Query Synapse for the file by name.
                results = syn.findEntityId(cnt_name, parent=_SYNAPSE_PROJECT)
                if results:
                    syn.get(results, downloadLocation=str(basepath))
                    log.info("Downloaded %s", cnt_name)
            except Exception as e:
                log.warning("Could not download %s: %s", cnt_name, e)

        return str(basepath)
