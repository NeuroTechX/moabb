"""Motor imagery vs rest low-cost EEG dataset.

Peterson, Galvan, Hernandez, and Spies (2020), Heliyon.
Paper DOI: 10.1016/j.heliyon.2020.e03425
Data DOI: 10.18112/openneuro.ds003810.v2.0.2
"""

import json
import logging
from pathlib import Path

import numpy as np
import requests

from .base import BaseBIDSDataset
from .download import get_dataset_path
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
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# OpenNeuro dataset ID.
_OPENNEURO_ID = "ds003810"

# S3 base URL for direct download (no auth needed for OpenNeuro).
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# Subjects present in the archive (sub-01 and sub-11 are absent).
_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

# MI-vs-Rest runs. RUN0 is a real-movement demonstration and is excluded.
_RUNS = ["1", "2", "3", "4"]

# 15 EEG channels (consumer-grade device, old 10-20 nomenclature).
# fmt: off
_CH_NAMES = [
    "Pz", "Cz", "T6", "T4", "F8", "P4", "C4", "F4",
    "Fz", "T5", "T3", "F7", "P3", "C3", "F3",
]
# fmt: on

# Two-class event mapping.
_EVENTS = {"motor_imagery": 1, "rest": 2}

# EDF annotation descriptions (OpenViBE GDF stimulation labels) -> class names.
# OVTK_GDF_Right = MI cue, OVTK_GDF_Tongue = Rest cue.
_ANNOT_TO_NAME = {"OVTK_GDF_Right": "motor_imagery", "OVTK_GDF_Tongue": "rest"}

# Minimal BIDS dataset_description.json for mne_bids compatibility.
_DATASET_DESCRIPTION = {
    "Name": "Motor Imagery vs Rest - Low-Cost EEG System",
    "BIDSVersion": "1.1.1",
    "License": "CC0",
    "Authors": [
        "Victoria Peterson",
        "Catalina Maria Galvan",
        "Hugo Sacha Hernadez",
        "Ruben Spies",
    ],
    "DatasetDOI": "10.18112/openneuro.ds003810.v2.0.2",
}


class Peterson2022(BaseBIDSDataset):
    """Motor imagery vs rest low-cost EEG dataset from Peterson et al 2020.

    Dataset from the feasibility study *A feasibility study of a complete
    low-cost consumer-grade brain-computer interface system* [1]_.

    EEG was recorded from 10 novice participants with a 15-channel
    consumer-grade device at 125 Hz. The paradigm is a binary
    kinesthetic motor imagery task: participants either imagined
    grasping with their dominant hand (**motor_imagery**) or stayed
    idle (**rest**).

    Each subject completed five runs. RUN0 is a real-movement
    demonstration run and is excluded; RUN1-RUN4 contain the MI-vs-rest
    trials (20 MI + 20 rest per run). The data are hosted on OpenNeuro
    (ds003810) in BIDS EDF format. Events are stored as native EDF
    annotations (OpenViBE GDF stimulation labels): ``OVTK_GDF_Right``
    marks a motor-imagery cue and ``OVTK_GDF_Tongue`` marks a rest cue.

    References
    ----------
    .. [1] Peterson, V., Galvan, C., Hernandez, H., & Spies, R. (2020).
           A feasibility study of a complete low-cost consumer-grade
           brain-computer interface system. Heliyon, 6(3), e03425.
           https://doi.org/10.1016/j.heliyon.2020.e03425
    """

    nemar_id = "ds003810"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=125.0,
            n_channels=15,
            channel_types={"eeg": 15},
            montage="10-20",
            hardware="consumer-grade EEG device",
            reference="left ear lobe",
            filters="none (no hardware/software filters applied during recording)",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            study_design=(
                "Binary kinesthetic motor imagery of dominant-hand grasping "
                "versus rest. RUN0 real-movement demo (excluded); RUN1-RUN4 "
                "MI-vs-rest (20 MI + 20 rest trials per run)."
            ),
            feedback_type="continuous visual",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.heliyon.2020.e03425",
            investigators=[
                "Victoria Peterson",
                "Catalina Maria Galvan",
                "Hugo Sacha Hernadez",
                "Ruben Spies",
            ],
            institution="IMAL, CONICET-UNL",
            institution_address="Santa Fe, Argentina",
            country="AR",
            data_url="https://openneuro.org/datasets/ds003810",
            publication_year=2020,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            notes=(
                "Signals distributed raw. The reference paper applied a "
                "3rd-order Butterworth bandpass filter (0.5-45 Hz) offline."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["CSP", "bandpower"],
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=160,
            n_trials_per_class={"motor_imagery": 80, "rest": 80},
            trials_context=(
                "Per subject: 4 MI-vs-rest runs x 40 trials (20 MI + 20 rest per run)."
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"], environment="laboratory", online_feedback=True
        ),
        data_processed=False,
        file_format="EDF (BIDS)",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(_SUBJECTS),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Peterson2022",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1016/j.heliyon.2020.e03425",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_path_search_params(self, subject):
        """Zero-padded subjects, MI runs only (exclude the RUN0 demo)."""
        out = {"extensions": [".edf"], "runs": list(_RUNS)}
        if subject is not None:
            out["subjects"] = f"{subject:02d}"
        return out

    def _get_single_subject_data(self, subject):
        """Load BIDS data and remap EDF annotation labels to class names."""
        data = super()._get_single_subject_data(subject)

        result = {}
        for sess_key, session_runs in data.items():
            runs = {}
            for run_key, raw in session_runs.items():
                desc = raw.annotations.description.astype(np.dtype("<25U"))
                for label, name in _ANNOT_TO_NAME.items():
                    desc[desc == label] = name
                raw.annotations.description = desc
                runs[run_key] = stim_channels_with_selected_ids(raw, self.event_id)
            result[sess_key] = runs

        return result

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download BIDS data from OpenNeuro S3 and return the BIDS root path."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        bids_root = Path(get_dataset_path("Peterson2022", path))
        bids_root = bids_root / "MNE-peterson2022-data"
        bids_root.mkdir(parents=True, exist_ok=True)

        subj_str = f"sub-{subject:02d}"
        self._download_subject_s3(bids_root, subj_str, force_update)
        self._ensure_dataset_description(bids_root)

        return str(bids_root)

    @staticmethod
    def _ensure_dataset_description(bids_root):
        """Create a minimal dataset_description.json if missing."""
        dd_path = bids_root / "dataset_description.json"
        if not dd_path.exists():
            with open(dd_path, "w") as f:
                json.dump(_DATASET_DESCRIPTION, f, indent=2)

    @staticmethod
    def _download_subject_s3(bids_root, subj_str, force_update):
        """Download per-subject MI-vs-rest run files directly from OpenNeuro S3."""
        rel_paths = []
        for run in _RUNS:
            stem = f"{subj_str}/eeg/{subj_str}_task-MIvsRest_run-{run}"
            rel_paths.extend(
                [f"{stem}_eeg.edf", f"{stem}_eeg.json", f"{stem}_channels.tsv"]
            )

        for rel_path in rel_paths:
            url = f"{_S3_BASE}/{rel_path}"
            local_path = bids_root / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists() or force_update:
                log.info("Downloading %s ...", rel_path)
                resp = requests.get(url, stream=True, timeout=120)
                if resp.status_code == 404:
                    log.warning("Not found: %s (skipping)", url)
                    continue
                resp.raise_for_status()
                with open(local_path, "wb") as fout:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fout.write(chunk)
