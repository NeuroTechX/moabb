"""Eye-BCI multimodal dataset (MI + ME paradigms).

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

# Zenodo records for MI and ME paradigms.
_ZENODO_RECORDS = {
    "MI": "PLACEHOLDER",
    "ME": "PLACEHOLDER",
}

# Event mapping shared by MI and ME paradigms.
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

# Recordings with "bis" suffix (repeated due to technical issues).
_BIS_MAP = {
    "MI": {(8, 1), (9, 1), (17, 1)},
    "ME": {
        (4, 1),
        (5, 2),
        (6, 1),
        (7, 2),
        (8, 1),
        (8, 2),
        (10, 2),
        (13, 1),
        (14, 2),
        (22, 1),
    },
}


def _bdf_name(paradigm, subject, session):
    """Return the BDF filename for a given recording.

    Parameters
    ----------
    paradigm : str
        "MI" or "ME".
    subject : int
        Subject number (1-31).
    session : int
        Session number (1-based).
    """
    code = f"{paradigm}{subject:02d}{session}"
    if (subject, session) in _BIS_MAP.get(paradigm, set()):
        code += "bis"
    return f"{code}.bdf"


class GuttmannFlury2025(BaseDataset):
    """Eye-BCI multimodal MI/ME dataset from Guttmann-Flury et al 2025.

    Dataset from the article *Dataset combining EEG, eye-tracking,
    and high-speed video for ocular activity analysis across BCI
    paradigms* [1]_.

    It contains EEG data from 31 subjects recorded with a 62-channel
    Neuroscan Quik-Cap + SynAmps2 at 1000 Hz. Four paradigms were
    tested (MI, ME, SSVEP, P300). This adapter loads the **Motor
    Imagery** and/or **Motor Execution** paradigms (2-class: left/right
    hand grasping), following the same pattern as
    :class:`moabb.datasets.PhysionetMI`.

    Each MI/ME session has 40 trials (20 left, 20 right). Trial
    structure: 2 s fixation + 4 s imagery/execution + 1-1.5 s rest.

    The data is hosted on Zenodo (re-hosted from Synapse with EEG
    converted from CSV to BDF format).

    Parameters
    ----------
    imagined : bool (default True)
        If True, load motor imagery (MI) runs.
    executed : bool (default False)
        If True, load motor execution (ME) runs.
    subjects : list of int | None
        List of subject numbers to load. Default loads all 31.
    sessions : list of int | None
        List of session numbers to load. Default loads all.

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
                "MI and ME: 2-class hand grasping, 40 trials/session, "
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
            trials_context="63 sessions x 40 trials = 2520 (MI only, default)",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
        ),
        data_processed=False,
        file_format="BDF",
    )

    def __init__(self, imagined=True, executed=False, subjects=None, sessions=None):
        if not imagined and not executed:
            raise ValueError("At least one of `imagined` or `executed` must be True.")

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
        self.imagined = imagined
        self.executed = executed

    @property
    def _paradigms(self):
        """Return list of paradigm codes to load."""
        paradigms = []
        if self.imagined:
            paradigms.append("MI")
        if self.executed:
            paradigms.append("ME")
        return paradigms

    def _load_raw(self, bdf_path):
        """Load a BDF file and add event annotations."""
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")

        # Find events from the STIM channel (Trig).
        stim_ch = "Trig"
        if stim_ch not in raw.ch_names:
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
            log.warning("No events (codes 1/2) in %s", bdf_path.name)

        return stim_channels_with_selected_ids(raw, self.event_id)

    def _get_single_subject_data(self, subject):
        """Return data for a single subject.

        When both ``imagined`` and ``executed`` are True, MI and ME
        recordings are returned as separate runs within each session
        (run "0" for MI, run "1" for ME).
        """
        paradigms = self._paradigms
        sessions = {}

        for paradigm in paradigms:
            base = Path(self._data_path_for_paradigm(subject, paradigm))
            n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)

            for sess_idx in range(1, n_sess + 1):
                name = _bdf_name(paradigm, subject, sess_idx)
                sess_dir = base / f"Sess{sess_idx:02d}"
                bdf_path = sess_dir / name

                if not bdf_path.exists():
                    candidates = list(
                        base.rglob(f"{paradigm}{subject:02d}{sess_idx}*.bdf")
                    )
                    if candidates:
                        bdf_path = candidates[0]
                    else:
                        log.warning("Missing %s", name)
                        continue

                try:
                    raw = self._load_raw(bdf_path)
                except Exception as e:
                    log.warning("Failed to load %s: %s", bdf_path.name, e)
                    continue

                sess_key = str(sess_idx - 1)
                if sess_key not in sessions:
                    sessions[sess_key] = {}

                # MI → run "0", ME → run "1" (or "0" if MI not loaded).
                run_idx = len(sessions[sess_key])
                sessions[sess_key][str(run_idx)] = raw

        if not sessions:
            raise FileNotFoundError(f"No MI/ME data for subject {subject}")
        return sessions

    def _data_path_for_paradigm(
        self,
        subject,
        paradigm,
        path=None,
        force_update=False,
        verbose=None,
    ):
        """Download and return the subject directory for one paradigm."""
        record_id = _ZENODO_RECORDS[paradigm]
        if record_id == "PLACEHOLDER":
            raise NotImplementedError(
                f"GuttmannFlury2025 {paradigm} Zenodo record ID not yet set. "
                "Data must be uploaded to Zenodo first."
            )

        sign = "GuttmannFlury2025"
        path = dl.get_dataset_path(sign, path)
        basepath = Path(path) / "MNE-guttmannflury2025-data"
        subj_dir = basepath / paradigm / f"S{subject:02d}"

        # Check if BDF files already exist for this subject.
        n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)
        all_exist = True
        for sess_idx in range(1, n_sess + 1):
            name = _bdf_name(paradigm, subject, sess_idx)
            sess_dir = subj_dir / f"Sess{sess_idx:02d}"
            if not (sess_dir / name).exists():
                all_exist = False
                break

        if all_exist and not force_update:
            return str(subj_dir)

        # Download per-subject ZIP from Zenodo.
        zenodo_base = f"https://zenodo.org/records/{record_id}/files"
        zip_name = f"S{subject:02d}.zip"
        url = f"{zenodo_base}/{zip_name}"
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
        log.info("Extracting %s (%s) to %s", zip_name, paradigm, subj_dir)
        with zipfile.ZipFile(str(dl_path)) as zf:
            safe_extract_zip(zf, subj_dir)

        return str(subj_dir)

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

        # Download all needed paradigms, return the first one's path.
        paths = []
        for paradigm in self._paradigms:
            p = self._data_path_for_paradigm(
                subject, paradigm, path, force_update, verbose
            )
            paths.append(p)
        return paths[0]
