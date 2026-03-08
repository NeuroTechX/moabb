"""Multimodal signal dataset for 11 intuitive movement tasks.

Jeong, Cho, Shim, et al. (2020), GigaScience.
DOI: 10.1093/gigascience/giaa098
Data DOI: 10.5524/100788
"""

import logging
from pathlib import Path

import mne

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


log = logging.getLogger(__name__)

# GigaDB Wasabi S3 base URL for dataset 100788.
_S3_BASE = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100788"

# 60 EEG channel names (10-20 system, BrainProducts actiCap).
# fmt: off
_CH_NAMES_EEG = [
    "Fp1", "AF7", "AF3", "AFz", "F7", "F5", "F3", "F1", "Fz",
    "FT7", "FC5", "FC3", "FC1",
    "T7", "C5", "C3", "C1", "Cz",
    "TP7", "CP5", "CP3", "CP1", "CPz",
    "P7", "P5", "P3", "P1", "Pz",
    "PO7", "PO3", "POz",
    "Fp2", "AF4", "AF8",
    "F2", "F4", "F6", "F8",
    "FC2", "FC4", "FC6", "FT8",
    "C2", "C4", "C6", "T8",
    "CP2", "CP4", "CP6", "TP8",
    "P2", "P4", "P6", "P8",
    "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
]
# fmt: on

# Movement onset markers for the reaching task (MI condition).
# These mark the start of the motor imagery execution period.
_REACHING_MI_EVENTS = {
    "reach_forward": 11,
    "reach_backward": 21,
    "reach_left": 31,
    "reach_right": 41,
    "reach_up": 51,
    "reach_down": 61,
}

# Multigrasp MI events.
_GRASP_MI_EVENTS = {
    "grasp_cup": 11,
    "grasp_ball": 21,
    "grasp_card": 61,
}

# Twist MI events.
_TWIST_MI_EVENTS = {
    "twist_pronation": 91,
    "twist_supination": 101,
}

# Default events: all 11 classes combined.
_ALL_EVENTS = {
    "reach_forward": 1,
    "reach_backward": 2,
    "reach_left": 3,
    "reach_right": 4,
    "reach_up": 5,
    "reach_down": 6,
    "grasp_cup": 7,
    "grasp_ball": 8,
    "grasp_card": 9,
    "twist_pronation": 10,
    "twist_supination": 11,
}

_SFREQ = 2500.0


class Jeong2020(BaseDataset):
    """Multimodal MI+ME dataset from Jeong et al 2020.

    Dataset from the article *Multimodal signal dataset for 11 intuitive
    movement tasks from single upper extremity during multiple recording
    sessions* [1]_.

    The dataset contains EEG, EOG, and EMG recordings from 25 subjects
    performing 11 intuitive movement tasks (6 reaching directions,
    3 grasping types, 2 wrist twists) during both motor imagery (MI)
    and motor execution (ME/realMove) conditions across 3 sessions.

    By default, only the **motor imagery** condition is loaded.

    Each session contains 3 task types:

    - **reaching**: 6 directions x 50 trials = 300 trials
    - **multigrasp**: 3 objects x 50 trials = 150 trials
    - **twist**: 2 motions x 50 trials = 100 trials

    Total: 550 MI trials per session, 1650 per subject (3 sessions).

    File format is BrainVision (.vhdr/.eeg/.vmrk), natively supported
    by MNE-Python. Data is hosted on GigaDB (Wasabi S3, CC0 license).

    Parameters
    ----------
    condition : str
        Which condition to load: ``"MI"`` (default) or ``"realMove"``.

    References
    ----------
    .. [1] Jeong, J.-H., Cho, J.-H., Shim, K.-H., et al. (2020).
           Multimodal signal dataset for 11 intuitive movement tasks
           from single upper extremity during multiple recording
           sessions. GigaScience, 9(10), giaa098.
           https://doi.org/10.1093/gigascience/giaa098
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=2500.0,
            n_channels=60,
            channel_types={"eeg": 60, "eog": 4, "emg": 7},
            montage="standard_1005",
            hardware="BrainAmp (BrainProducts GmbH)",
            sensor_type="actiCap",
            reference="FCz",
            ground="Fpz",
            filters={"highpass": 0.016, "lowpass": 1000},
            sensors=list(_CH_NAMES_EEG),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender={"female": 10, "male": 15},
            age_min=24.0,
            age_max=32.0,
            handedness="right-handed",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_ALL_EVENTS),
            paradigm="imagery",
            n_classes=11,
            class_labels=list(_ALL_EVENTS.keys()),
            trial_duration=4.0,
            study_design=(
                "11 intuitive upper-limb movement tasks: "
                "6 reaching + 3 grasping + 2 wrist twisting. "
                "MI and real movement conditions, 3 sessions."
            ),
            feedback_type="none",
            stimulus_type="text cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/gigascience/giaa098",
            investigators=[
                "Ji-Hoon Jeong",
                "Jeong-Hyun Cho",
                "Kyung-Hwan Shim",
                "Byoung-Hee Lee",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            country="KR",
            data_url="https://gigadb.org/dataset/100788",
            publication_year=2020,
            license="CC0-1.0",
        ),
        sessions_per_subject=3,
        runs_per_session=3,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_ALL_EVENTS.keys()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=41250,
            trials_context=(
                "25 subjects x 3 sessions x 550 trials (300 reaching + "
                "150 grasping + 100 twisting)"
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "prosthetics"],
            environment="laboratory",
        ),
        data_processed=False,
        file_format="BrainVision",
    )

    def __init__(self, condition="MI", subjects=None, sessions=None):
        self.condition = condition
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=3,
            events=dict(_ALL_EVENTS),
            code="Jeong2020",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1093/gigascience/giaa098",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))
        subj_str = f"sub{subject:02d}"

        sessions = {}
        for sess_idx in range(1, 4):
            sess_str = f"session{sess_idx}"
            runs = {}

            # 3 task types per session
            for run_idx, task_type in enumerate(["reaching", "multigrasp", "twist"]):
                # File naming: sub01_session1_reaching_MI.vhdr
                vhdr_name = f"{subj_str}_{sess_str}_{task_type}_{self.condition}.vhdr"
                vhdr_path = base / vhdr_name

                if not vhdr_path.exists():
                    # Try alternative paths
                    vhdr_path = base / subj_str / vhdr_name
                if not vhdr_path.exists():
                    alt = list(base.rglob(f"*{task_type}*{self.condition}*.vhdr"))
                    if alt:
                        # Filter for correct session
                        sess_matches = [f for f in alt if sess_str in str(f)]
                        if sess_matches:
                            vhdr_path = sess_matches[0]
                        else:
                            vhdr_path = alt[0]

                if not vhdr_path.exists():
                    log.warning(
                        "Missing: %s %s %s %s",
                        subj_str,
                        sess_str,
                        task_type,
                        self.condition,
                    )
                    continue

                raw = mne.io.read_raw_brainvision(
                    str(vhdr_path), preload=True, verbose=False
                )

                # Remap stimulus markers to unified event codes.
                self._remap_annotations(raw, task_type)
                runs[str(run_idx)] = raw

            if runs:
                sessions[str(sess_idx - 1)] = runs

        if not sessions:
            raise FileNotFoundError(f"No data found for {subj_str} in {base}")
        return sessions

    def _remap_annotations(self, raw, task_type):
        """Remap BrainVision stimulus markers to unified event codes."""
        # Map task-specific movement onset markers to unified event names.
        if task_type == "reaching":
            marker_map = {
                "S 11": "reach_forward",
                "S 21": "reach_backward",
                "S 31": "reach_left",
                "S 41": "reach_right",
                "S 51": "reach_up",
                "S 61": "reach_down",
            }
        elif task_type == "multigrasp":
            marker_map = {
                "S 11": "grasp_cup",
                "S 21": "grasp_ball",
                "S 61": "grasp_card",
            }
        elif task_type == "twist":
            marker_map = {
                "S 91": "twist_pronation",
                "S101": "twist_supination",
            }
        else:
            return

        new_annotations = []
        for ann in raw.annotations:
            desc = ann["description"].strip()
            if desc in marker_map:
                new_annotations.append((ann["onset"], ann["duration"], marker_map[desc]))

        if new_annotations:
            onsets, durations, descriptions = zip(*new_annotations)
            raw.set_annotations(
                mne.Annotations(
                    onset=list(onsets),
                    duration=list(durations),
                    description=list(descriptions),
                )
            )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Jeong2020", path)
        basepath = Path(path) / "MNE-jeong2020-data"
        basepath.mkdir(parents=True, exist_ok=True)

        subj_str = f"sub{subject:02d}"

        # Check for existing data.
        existing = list(basepath.rglob(f"*{subj_str}*{self.condition}*.vhdr"))
        if existing:
            return str(basepath)

        # Download from GigaDB (Wasabi S3).
        task_types = ["reaching", "multigrasp", "twist"]
        for sess_idx in range(1, 4):
            sess_str = f"session{sess_idx}"
            for task_type in task_types:
                fname = f"{subj_str}_{sess_str}_{task_type}_{self.condition}"
                for ext in [".vhdr", ".eeg", ".vmrk"]:
                    url = f"{_S3_BASE}/{fname}{ext}"
                    dest = basepath / f"{fname}{ext}"
                    if not dest.exists():
                        try:
                            dl_path = dl.data_dl(
                                url,
                                "Jeong2020",
                                path=str(basepath),
                                force_update=force_update,
                                verbose=verbose,
                            )
                            # Rename to expected location.
                            dl_path = Path(dl_path)
                            if dl_path != dest:
                                dl_path.rename(dest)
                        except Exception as e:
                            log.warning("Download failed for %s: %s", url, e)

        return str(basepath)
