"""Eye-BCI multimodal dataset (MI, ME, SSVEP, P300 paradigms).

Guttmann-Flury, Sheng, and Zhu (2025), Scientific Data.
DOI: 10.1038/s41597-025-04861-9
Data DOI: 10.7303/syn64005218
"""

import csv
import json
import logging
import re
import zipfile
from pathlib import Path

import mne
import numpy as np

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
from .utils import safe_extract_zip, stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# Zenodo records per paradigm.
_ZENODO_RECORDS = {
    "MI": "18970793",
    "ME": "18971758",
    "SSVEP": "18978288",
    "P3004L": "18980192",
    "P3005L": "18982867",
}

_DOI = "10.1038/s41597-025-04861-9"

# Event mappings per paradigm family.
_MI_ME_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
}

# SSVEP: 4 frequencies at 4 spatial positions.
# Actual frequencies from the E-Prime sync CSV: 10, 11, 12, 13 Hz.
_SSVEP_EVENTS = {
    "10.0": 1,
    "11.0": 2,
    "12.0": 3,
    "13.0": 4,
}
# Direction → frequency mapping (retained for future annotation support).
_SSVEP_DIRECTION_TO_FREQ = {
    "Up": "13.0",
    "Down": "10.0",
    "Left": "12.0",
    "Right": "11.0",
}

# P300 speller: Target vs NonTarget flash events.
_P300_EVENTS = {
    "Target": 1,
    "NonTarget": 2,
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

# Recordings with "bis"/"tri" suffix (repeated due to technical issues).
_BIS_MAP = {
    "MI": {(8, 1): "bis", (9, 1): "bis", (17, 1): "bis"},
    "ME": {
        (4, 1): "bis",
        (5, 2): "bis",
        (6, 1): "bis",
        (7, 2): "bis",
        (8, 1): "bis",
        (8, 2): "bis",
        (10, 2): "bis",
        (13, 1): "bis",
        (14, 2): "bis",
        (22, 1): "bis",
    },
    "SSVEP": {
        (3, 1): "bis",
        (14, 2): "bis",
        (19, 1): "bis",
        (23, 1): "bis",
        (23, 2): "tri",
        (28, 1): "bis",
    },
    "P3004L": {(3, 1): "bis", (10, 1): "bis", (31, 1): "bis"},
    "P3005L": {
        (9, 1): "bis",
        (10, 2): "bis",
        (14, 3): "bis",
        (19, 1): "bis",
        (23, 1): "bis",
    },
}

# Shared documentation metadata.
_DOCUMENTATION = DocumentationMetadata(
    doi=_DOI,
    investigators=[
        "Eva Guttmann-Flury",
        "Xinjun Sheng",
        "Xiangyang Zhu",
    ],
    institution="Shanghai Jiao Tong University",
    country="CN",
    publication_year=2025,
    license="CC0",
)

# Shared participant metadata.
_PARTICIPANTS = ParticipantMetadata(
    n_subjects=31,
    health_status="healthy",
    gender={"female": 11, "male": 20},
    age_mean=28.3,
    age_min=20.0,
    age_max=57.0,
    species="human",
)

# Shared acquisition metadata.
_ACQUISITION = AcquisitionMetadata(
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
)


def _bdf_name(paradigm, subject, session):
    """Return the BDF filename for a given recording.

    Parameters
    ----------
    paradigm : str
        "MI", "ME", "SSVEP", "P3004L", or "P3005L".
    subject : int
        Subject number (1-31).
    session : int
        Session number (1-based).
    """
    code = f"{paradigm}{subject:02d}{session}"
    suffix = _BIS_MAP.get(paradigm, {}).get((subject, session), "")
    return f"{code}{suffix}.bdf"


def _data_path_for_paradigm(
    paradigm,
    subject,
    sign="GuttmannFlury2025",
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


def _find_bdf(base, paradigm, subject, sess_idx):
    """Locate a BDF file, trying expected path then fallback glob."""
    name = _bdf_name(paradigm, subject, sess_idx)
    sess_dir = base / f"Sess{sess_idx:02d}"
    bdf_path = sess_dir / name

    if bdf_path.exists():
        return bdf_path

    candidates = list(base.rglob(f"{paradigm}{subject:02d}{sess_idx}*.bdf"))
    if candidates:
        return candidates[0]

    log.warning("Missing %s", name)
    return None


def _load_annotations_json(bdf_path):
    """Load the _annotations.json sidecar next to a BDF file."""
    stem = bdf_path.stem  # e.g., "SSVEP011" or "SSVEP031bis"
    json_path = bdf_path.parent / f"{stem}_annotations.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def _decode_ssvep_from_sync_csv(bdf_path):
    """Decode SSVEP frequency events from the sync CSV.

    The sync CSV has cues like "10 Hz", "11 Hz", "12 Hz", "13 Hz"
    indicating which frequency was stimulated on each trial.
    """
    stem = bdf_path.stem
    csv_path = bdf_path.parent / f"{stem}_sync.csv"
    if not csv_path.exists():
        return None

    _hz_re = re.compile(r"^(\d+)\s*Hz$")
    records = []
    prev_cue = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cue = row.get("Cues", "").strip()
            if not cue or cue == prev_cue:
                continue
            prev_cue = cue
            m = _hz_re.match(cue)
            if m:
                freq = f"{int(m.group(1))}.0"
                if freq in _SSVEP_EVENTS:
                    records.append({"onset": float(row["Time"]), "description": freq})

    return records if records else None


def _decode_p300_from_sync_csv(bdf_path):
    """Decode P300 Target/NonTarget flash events from the sync CSV.

    The sync CSV records frame-level cues from the E-Prime P300 speller:

    - ``"Letter X in WORD"`` identifies the target letter for each trial.
    - ``"SeqNN: A, B, C, ..."`` shows which 6 characters are highlighted
      in each flash.  A flash is Target if the target letter is in the
      group, NonTarget otherwise.

    Returns a list of dicts ``{"onset": float, "description": str}``
    suitable for writing as an annotations JSON, or *None* if the sync
    CSV is missing.
    """
    stem = bdf_path.stem
    csv_path = bdf_path.parent / f"{stem}_sync.csv"
    if not csv_path.exists():
        return None

    # Parse sync CSV — only need Time and Cues columns.
    times = []
    cues = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cue = row.get("Cues", "").strip()
            if cue:
                times.append(float(row["Time"]))
                cues.append(cue)

    # Walk through cue transitions to extract flash events.
    target_letter = None
    records = []
    _letter_re = re.compile(r"^Letter (.+) in (.+)$")
    _flash_re = re.compile(r"^Seq\d+: (.+)$")

    prev_cue = None
    for t, cue in zip(times, cues):
        if cue == prev_cue:
            continue  # skip repeated frames
        prev_cue = cue

        m = _letter_re.match(cue)
        if m:
            target_letter = m.group(1)
            continue

        m = _flash_re.match(cue)
        if m and target_letter is not None:
            chars = [c.strip() for c in m.group(1).split(",")]
            desc = "Target" if target_letter in chars else "NonTarget"
            records.append({"onset": t, "description": desc})

    return records if records else None


def _load_raw_with_stim_events(bdf_path, event_id):
    """Load BDF file, decode Trig channel events, and set annotations.

    Used by MI/ME where Trig channel codes directly map to event types.
    """
    raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")

    stim_ch = "Trig"
    if stim_ch not in raw.ch_names:
        stim_ch = raw.ch_names[-1]

    events = mne.find_events(raw, stim_channel=stim_ch, verbose="ERROR")

    event_id_inv = {v: k for k, v in event_id.items()}
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
        log.warning("No events in %s", bdf_path.name)

    return stim_channels_with_selected_ids(raw, event_id)


# ---------------------------------------------------------------------------
# MI + ME adapter
# ---------------------------------------------------------------------------


class GuttmannFlury2025_MI(BaseDataset):
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
        acquisition=_ACQUISITION,
        participants=_PARTICIPANTS,
        experiment=ExperimentMetadata(
            events=dict(_MI_ME_EVENTS),
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
        documentation=_DOCUMENTATION,
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
        signal_processing=SignalProcessingMetadata(
            classifiers=None,
            feature_extraction=None,
            frequency_bands=None,
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method=None,
            evaluation_type=None,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="BDF",
    )

    def __init__(
        self,
        imagined=True,
        executed=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        if not imagined and not executed:
            raise ValueError("At least one of `imagined` or `executed` must be True.")

        super().__init__(
            subjects=list(range(1, 32)),
            sessions_per_subject=3,
            events=dict(_MI_ME_EVENTS),
            code="GuttmannFlury2025-MI",
            interval=[0, 4],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
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

    def _get_single_subject_data(self, subject):
        """Return data for a single subject.

        When both ``imagined`` and ``executed`` are True, MI and ME
        recordings are returned as separate runs within each session
        (run "0" for MI, run "1" for ME).
        """
        sessions = {}

        for paradigm in self._paradigms:
            base = Path(_data_path_for_paradigm(paradigm, subject, self.code))
            n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)

            for sess_idx in range(1, n_sess + 1):
                bdf_path = _find_bdf(base, paradigm, subject, sess_idx)
                if bdf_path is None:
                    continue

                try:
                    raw = _load_raw_with_stim_events(bdf_path, self.event_id)
                except Exception as e:
                    log.warning("Failed to load %s: %s", bdf_path.name, e)
                    continue

                sess_key = str(sess_idx - 1)
                if sess_key not in sessions:
                    sessions[sess_key] = {}

                # MI -> run "0", ME -> run "1" (or "0" if MI not loaded).
                run_idx = len(sessions[sess_key])
                sessions[sess_key][str(run_idx)] = raw

        if not sessions:
            raise FileNotFoundError(f"No MI/ME data for subject {subject}")
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

        paths = []
        for paradigm in self._paradigms:
            p = _data_path_for_paradigm(
                paradigm, subject, self.code, path, force_update, verbose
            )
            paths.append(p)
        return paths[0]


class GuttmannFlury2025_ME(BaseDataset):
    """Eye-BCI Motor Execution dataset from Guttmann-Flury et al 2025.

    Same paradigm as :class:`GuttmannFlury2025_MI` but loads Motor
    Execution recordings (real hand grasping) instead of Motor Imagery.

    See :class:`GuttmannFlury2025_MI` for full documentation.
    """

    METADATA = GuttmannFlury2025_MI.METADATA

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 32)),
            sessions_per_subject=3,
            events=dict(_MI_ME_EVENTS),
            code="GuttmannFlury2025-ME",
            interval=[0, 4],
            paradigm="imagery",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )
        self.imagined = False
        self.executed = True

    @property
    def _paradigms(self):
        return ["ME"]

    _get_single_subject_data = GuttmannFlury2025_MI._get_single_subject_data

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        return _data_path_for_paradigm(
            "ME", subject, self.code, path, force_update, verbose
        )


# ---------------------------------------------------------------------------
# SSVEP adapter
# ---------------------------------------------------------------------------


class GuttmannFlury2025_SSVEP(BaseDataset):
    """Eye-BCI multimodal SSVEP dataset from Guttmann-Flury et al 2025.

    Dataset from the article *Dataset combining EEG, eye-tracking,
    and high-speed video for ocular activity analysis across BCI
    paradigms* [1]_.

    This adapter loads the **SSVEP** paradigm (4-class: 10, 11, 12,
    13 Hz flickering stimuli). Each SSVEP session has 40 trials
    (4 frequencies x 10 repetitions). Trial structure:
    fixation + stimulus + rest.

    Event types are decoded from the E-Prime sync CSV that records
    the stimulation frequency for each trial.

    Parameters
    ----------
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
        acquisition=_ACQUISITION,
        participants=_PARTICIPANTS,
        experiment=ExperimentMetadata(
            events=dict(_SSVEP_EVENTS),
            paradigm="ssvep",
            n_classes=4,
            class_labels=["10.0", "11.0", "12.0", "13.0"],
            trial_duration=7.0,
            study_design=(
                "Multi-paradigm BCI (MI/ME/SSVEP/P300). "
                "SSVEP: 4-class frequency flickering, 48 trials/session, "
                "up to 3 sessions per subject."
            ),
            feedback_type="none",
            stimulus_type="flickering LED",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=_DOCUMENTATION,
        sessions_per_subject=3,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[8.0, 10.0, 12.0, 15.0],
        ),
        data_structure=DataStructureMetadata(
            n_trials=3024,
            trials_context="63 sessions x 48 trials = 3024",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["communication"],
            environment="laboratory",
        ),
        data_processed=False,
        file_format="BDF",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 32)),
            sessions_per_subject=3,
            events=dict(_SSVEP_EVENTS),
            code="GuttmannFlury2025-SSVEP",
            interval=[0, 5],
            paradigm="ssvep",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _load_ssvep_raw(self, bdf_path):
        """Load SSVEP BDF and decode frequency events from sync CSV."""
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")

        annot_onset = []
        annot_dur = []
        annot_desc = []

        # Try annotations JSON first (arrow field).
        ann_records = _load_annotations_json(bdf_path)
        if ann_records is not None:
            for rec in ann_records:
                phase = rec.get("phase", "")
                if phase == "fixation" or phase == "rest":
                    continue
                arrow = rec.get("arrow", "")
                freq_label = _SSVEP_DIRECTION_TO_FREQ.get(arrow)
                if freq_label and freq_label in self.event_id:
                    annot_onset.append(rec["onset"])
                    annot_dur.append(0.0)
                    annot_desc.append(freq_label)

        # Fall back to sync CSV if annotations had no frequency events.
        if not annot_onset:
            sync_records = _decode_ssvep_from_sync_csv(bdf_path)
            if sync_records:
                for rec in sync_records:
                    annot_onset.append(rec["onset"])
                    annot_dur.append(0.0)
                    annot_desc.append(rec["description"])

        if annot_onset:
            annotations = mne.Annotations(
                onset=np.array(annot_onset),
                duration=np.array(annot_dur),
                description=annot_desc,
            )
            raw.set_annotations(annotations)
        else:
            log.warning("No SSVEP frequency events in %s", bdf_path.name)

        return stim_channels_with_selected_ids(raw, self.event_id)

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(_data_path_for_paradigm("SSVEP", subject, self.code))
        n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)
        sessions = {}

        for sess_idx in range(1, n_sess + 1):
            bdf_path = _find_bdf(base, "SSVEP", subject, sess_idx)
            if bdf_path is None:
                continue

            try:
                raw = self._load_ssvep_raw(bdf_path)
            except Exception as e:
                log.warning("Failed to load %s: %s", bdf_path.name, e)
                continue

            sessions[str(sess_idx - 1)] = {"0": raw}

        if not sessions:
            raise FileNotFoundError(f"No SSVEP data for subject {subject}")
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
        return _data_path_for_paradigm(
            "SSVEP", subject, self.code, path, force_update, verbose
        )


# ---------------------------------------------------------------------------
# P300 adapter
# ---------------------------------------------------------------------------


class GuttmannFlury2025_P300(BaseDataset):
    """Eye-BCI multimodal P300 speller dataset from Guttmann-Flury et al 2025.

    Dataset from the article *Dataset combining EEG, eye-tracking,
    and high-speed video for ocular activity analysis across BCI
    paradigms* [1]_.

    This adapter loads the **P300 speller** paradigm. Two grid sizes
    were tested: 4-letter (P3004L) and 5-letter (P3005L). Events are
    decoded from the ``_annotations.json`` sidecar.

    Parameters
    ----------
    grid_size : str (default "4L")
        Speller grid size: ``"4L"`` for 4-letter grid or ``"5L"`` for
        5-letter grid.
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
        acquisition=_ACQUISITION,
        participants=_PARTICIPANTS,
        experiment=ExperimentMetadata(
            events=dict(_P300_EVENTS),
            paradigm="p300",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            study_design=(
                "Multi-paradigm BCI (MI/ME/SSVEP/P300). "
                "P300: row/column speller with 4L and 5L grid sizes."
            ),
            feedback_type="none",
            stimulus_type="row-column flash",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=_DOCUMENTATION,
        sessions_per_subject=3,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["ERP"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_trials=2520,
            trials_context="63 sessions x 40 trials = 2520 (P300-4L default)",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
        ),
        data_processed=False,
        file_format="BDF",
    )

    def __init__(
        self,
        grid_size="4L",
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        if grid_size not in ("4L", "5L"):
            raise ValueError(f"grid_size must be '4L' or '5L', got '{grid_size}'")

        self.grid_size = grid_size
        self._paradigm_code = f"P300{grid_size}"

        super().__init__(
            subjects=list(range(1, 32)),
            sessions_per_subject=3,
            events=dict(_P300_EVENTS),
            code="GuttmannFlury2025-P300",
            interval=[0, 1],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _load_p300_raw(self, bdf_path):
        """Load P300 BDF and decode Target/NonTarget events.

        Tries the annotations JSON first; if it contains MI-style labels
        instead of proper Target/NonTarget events (known packaging bug),
        falls back to decoding flash events from the sync CSV.
        """
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="ERROR")

        # Try annotations JSON first.
        ann_records = _load_annotations_json(bdf_path)
        annot_onset = []
        annot_dur = []
        annot_desc = []

        if ann_records is not None:
            for rec in ann_records:
                desc = rec.get("description", "")
                if desc in ("Target", "NonTarget"):
                    annot_onset.append(rec["onset"])
                    annot_dur.append(0.0)
                    annot_desc.append(desc)

        # If no proper P300 events found, decode from sync CSV.
        if not annot_onset:
            sync_records = _decode_p300_from_sync_csv(bdf_path)
            if sync_records:
                for rec in sync_records:
                    annot_onset.append(rec["onset"])
                    annot_dur.append(0.0)
                    annot_desc.append(rec["description"])
                log.info(
                    "Decoded %d P300 flash events from sync CSV for %s",
                    len(annot_onset),
                    bdf_path.name,
                )

                # Cache corrected annotations for future loads.
                stem = bdf_path.stem
                json_path = bdf_path.parent / f"{stem}_annotations.json"
                try:
                    with open(json_path, "w") as f:
                        json.dump(sync_records, f, indent=2)
                except OSError:
                    pass  # read-only or full disk — not critical
            else:
                log.warning("No P300 events decoded for %s", bdf_path.name)

        if annot_onset:
            annotations = mne.Annotations(
                onset=np.array(annot_onset),
                duration=np.array(annot_dur),
                description=annot_desc,
            )
            raw.set_annotations(annotations)

        return stim_channels_with_selected_ids(raw, self.event_id)

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        paradigm = self._paradigm_code
        base = Path(_data_path_for_paradigm(paradigm, subject, self.code))
        n_sess = _SESSIONS_PER_SUBJECT.get(subject, 1)
        sessions = {}

        for sess_idx in range(1, n_sess + 1):
            bdf_path = _find_bdf(base, paradigm, subject, sess_idx)
            if bdf_path is None:
                continue

            try:
                raw = self._load_p300_raw(bdf_path)
            except Exception as e:
                log.warning("Failed to load %s: %s", bdf_path.name, e)
                continue

            sessions[str(sess_idx - 1)] = {"0": raw}

        if not sessions:
            raise FileNotFoundError(f"No P300 ({paradigm}) data for subject {subject}")
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
        return _data_path_for_paradigm(
            self._paradigm_code, subject, self.code, path, force_update, verbose
        )
