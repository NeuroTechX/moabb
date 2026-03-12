"""Eye-BCI multimodal dataset (MI paradigm).

Guttmann-Flury, Sheng, and Zhu (2025), Scientific Data.
DOI: 10.1038/s41597-025-04861-9
Data DOI: 10.7303/syn64005218
"""

import logging
import zipfile
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
from .utils import safe_extract_zip, stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# Zenodo record for the MI paradigm.
_ZENODO_RECORD = "PLACEHOLDER"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

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

# Corrected sessions per subject (verified from Order Paradigms.csv).
_SESSIONS_PER_SUBJECT = {
    1: 1,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 1,
    7: 3,
    8: 3,
    9: 3,
    10: 2,
    11: 1,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 1,
    20: 1,
    21: 1,
    22: 1,
    23: 3,
    24: 1,
    25: 2,
    26: 1,
    27: 1,
    28: 1,
    29: 1,
    30: 1,
    31: 1,
}

# MI recordings with "bis" suffix (repeated due to technical issues).
_MI_BIS = {(8, 1), (9, 1), (17, 1)}


def _mi_bdf_name(subject, session):
    """Return the BDF filename for a given MI recording."""
    code = f"MI{subject:02d}{session}"
    if (subject, session) in _MI_BIS:
        code += "bis"
    return f"{code}.bdf"


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

    The data is hosted on Zenodo (re-hosted from Synapse with EEG
    converted from CSV to BDF format).

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
            n_channels=66,
            channel_types={"eeg": 64, "eog": 1, "stim": 1},
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
            age_mean=28.3,
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
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
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
        file_format="BDF",
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
            bdf_name = _mi_bdf_name(subject, sess_idx)
            sess_dir = base / f"Sess{sess_idx:02d}"
            bdf_path = sess_dir / bdf_name

            if not bdf_path.exists():
                # Try finding BDF in base directory or alternative paths.
                candidates = list(base.rglob(f"MI{subject:02d}{sess_idx}*.bdf"))
                if candidates:
                    bdf_path = candidates[0]
                else:
                    log.warning("Missing %s", bdf_name)
                    continue

            try:
                raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")

                # Find events from the STIM channel (Trig).
                stim_ch = "Trig"
                if stim_ch not in raw.ch_names:
                    # Fall back to last channel
                    stim_ch = raw.ch_names[-1]

                events = mne.find_events(raw, stim_channel=stim_ch, verbose="ERROR")

                # Create annotations from events.
                event_id_inv = {v: k for k, v in _EVENTS.items()}
                annot_onset = []
                annot_dur = []
                annot_desc = []
                for ev in events:
                    code = int(ev[2])
                    if code in event_id_inv:
                        annot_onset.append(ev[0] / raw.info["sfreq"])
                        annot_dur.append(0.0)
                        annot_desc.append(event_id_inv[code])

                if annot_onset:
                    annotations = mne.Annotations(
                        onset=np.array(annot_onset),
                        duration=np.array(annot_dur),
                        description=annot_desc,
                    )
                    raw.set_annotations(annotations)
                else:
                    log.warning("No MI events (codes 1/2) in %s", bdf_path.name)

                raw = stim_channels_with_selected_ids(raw, self.event_id)
                sessions[str(sess_idx - 1)] = {"0": raw}

            except Exception as e:
                log.warning("Failed to load %s: %s", bdf_path.name, e)

        if not sessions:
            raise FileNotFoundError(f"No MI data for subject {subject} in {base}")
        return sessions

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        if _ZENODO_RECORD == "PLACEHOLDER":
            raise NotImplementedError(
                "GuttmannFlury2025 Zenodo record ID not yet set. "
                "Data must be uploaded to Zenodo first."
            )

        sign = "GuttmannFlury2025"
        path = dl.get_dataset_path(sign, path)
        basepath = Path(path) / "MNE-guttmannflury2025-data"
        subj_dir = basepath / f"S{subject:02d}"

        # Check if BDF files already exist for this subject.
        n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)
        all_exist = True
        for sess_idx in range(1, n_sess + 1):
            bdf_name = _mi_bdf_name(subject, sess_idx)
            sess_dir = subj_dir / f"Sess{sess_idx:02d}"
            if not (sess_dir / bdf_name).exists():
                all_exist = False
                break

        if all_exist and not force_update:
            return str(subj_dir)

        # Download per-subject ZIP from Zenodo.
        zip_name = f"S{subject:02d}.zip"
        url = f"{_ZENODO_BASE}/{zip_name}"
        dl_path = Path(dl.data_dl(url, sign, path, force_update, verbose))

        # The downloaded file might be in a nested path; find it.
        if dl_path.is_dir():
            zip_candidates = list(dl_path.rglob(zip_name))
            if zip_candidates:
                dl_path = zip_candidates[0]
            else:
                raise FileNotFoundError(
                    f"Downloaded {zip_name} but could not locate ZIP in {dl_path}"
                )

        # Extract ZIP to subject directory.
        subj_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s to %s", zip_name, subj_dir)
        with zipfile.ZipFile(str(dl_path)) as zf:
            safe_extract_zip(zf, subj_dir)

        return str(subj_dir)
