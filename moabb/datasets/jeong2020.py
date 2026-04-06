"""Multimodal signal dataset for 11 intuitive movement tasks.

Jeong, Cho, Shim, et al. (2020), GigaScience.
DOI: 10.1093/gigascience/giaa098
Data DOI: 10.5524/100788
"""

import logging
import zipfile
from pathlib import Path

import mne

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

# Zenodo records (resampled 2500→1000 Hz, per-subject ZIPs).
# 5 records, 5 subjects each.
_ZENODO_RECORDS = {
    1: "19021436",
    2: "19021436",
    3: "19021436",
    4: "19021436",
    5: "19021436",
    6: "19021439",
    7: "19021439",
    8: "19021439",
    9: "19021439",
    10: "19021439",
    11: "19021438",
    12: "19021438",
    13: "19021438",
    14: "19021438",
    15: "19021438",
    16: "19021435",
    17: "19021435",
    18: "19021435",
    19: "19021435",
    20: "19021435",
    21: "19021437",
    22: "19021437",
    23: "19021437",
    24: "19021437",
    25: "19021437",
}

_SIGN = "Jeong2020"
_DOI = "10.1093/gigascience/giaa098"

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

_SFREQ = 1000.0  # Resampled from 2500 Hz


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
    by MNE-Python. Data is re-hosted on Zenodo (resampled from 2500
    to 1000 Hz, per-subject ZIPs). Original data on GigaDB (CC0).

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
            sampling_rate=1000.0,
            n_channels=71,
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
            hed_tags={
                # Reaching: 6 directions (Forward/Backward not in HED schema)
                "reach_forward": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Reach, (Label/forward)))"
                ),
                "reach_backward": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Reach, (Label/backward)))"
                ),
                "reach_left": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Reach, Left))"
                ),
                "reach_right": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Reach, Right))"
                ),
                "reach_up": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Reach, Upward))"
                ),
                "reach_down": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Reach, Downward))"
                ),
                # Grasping: 3 types (cylindrical/spherical/lateral per paper)
                "grasp_cup": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Grasp, Hand, (Label/cylindrical)))"
                ),
                "grasp_ball": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Grasp, Hand, (Label/spherical)))"
                ),
                "grasp_card": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Grasp, Hand, (Label/lateral)))"
                ),
                # Twisting: 2 types (Forearm for consistency with Ofner2017)
                "twist_pronation": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Turn, Forearm, (Label/pronation)))"
                ),
                "twist_supination": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation), "
                    "(Agent-action, (Imagine, Turn, Forearm, (Label/supination)))"
                ),
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/gigascience/giaa098",
            investigators=[
                "Ji-Hoon Jeong",
                "Jeong-Hyun Cho",
                "Kyung-Hwan Shim",
                "Byoung-Hee Kwon",
                "Byeong-Hoo Lee",
                "Do-Yeun Lee",
                "Dae-Hyeok Lee",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            country="KR",
            data_url="https://zenodo.org/records/19021436",
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
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+RLDA"],
            feature_extraction=["CSP"],
            frequency_bands={
                "mu_beta": [8.0, 30.0],
            },
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10x10-fold",
            cv_folds=10,
            evaluation_type=["within_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "prosthetics"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="BrainVision",
    )

    def __init__(
        self,
        condition="MI",
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
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
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))
        subj_str = f"sub{subject}"

        sessions = {}
        for sess_idx in range(1, 4):
            sess_str = f"session{sess_idx}"
            runs = {}

            for run_idx, task_type in enumerate(["reaching", "multigrasp", "twist"]):
                vhdr_name = f"{sess_str}_{subj_str}_{task_type}_{self.condition}.vhdr"
                vhdr_path = base / vhdr_name

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
            # MNE BrainVision reader adds "Stimulus/" prefix
            desc = ann["description"].strip()
            if desc.startswith("Stimulus/"):
                desc = desc[len("Stimulus/") :].strip()
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

        path = dl.get_dataset_path(_SIGN, path)
        basepath = Path(path) / "MNE-jeong2020-data"
        subj_str = f"sub{subject}"
        subj_dir = basepath / subj_str

        # Check if already extracted.
        if subj_dir.exists() and not force_update:
            vhdrs = list(subj_dir.glob("*.vhdr"))
            if vhdrs:
                return str(subj_dir)

        # Download per-subject ZIP from Zenodo.
        record_id = _ZENODO_RECORDS[subject]
        zenodo_base = f"https://zenodo.org/records/{record_id}/files"
        zip_name = f"{subj_str}.zip"
        url = f"{zenodo_base}/{zip_name}"

        dl_path = Path(dl.data_dl(url, _SIGN, path, force_update, verbose))

        # Extract ZIP.
        basepath.mkdir(parents=True, exist_ok=True)
        subj_dir.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s to %s", zip_name, subj_dir)
        with zipfile.ZipFile(str(dl_path)) as zf:
            zf.extractall(str(subj_dir))

        return str(subj_dir)
