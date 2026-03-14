"""Sit-stand motor imagery EEG dataset.

Triana-Guzman, Orjuela-Canon, and Jutinico (2022), Frontiers in Neuroinformatics.
DOI: 10.3389/fninf.2022.961089
Data DOI: 10.18112/openneuro.ds005342.v1.0.3
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
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# OpenNeuro dataset ID.
_OPENNEURO_ID = "ds005342"

# S3 base URL for direct download (no auth needed for OpenNeuro).
# No version prefix — OpenNeuro serves latest version at the bare path.
_S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{_OPENNEURO_ID}"

# Event codes from events.tsv value column.
# 1 = MotorImageryA (sitting, imagine stand-up)
# 2 = IdleStateA (sitting, idle)
# 3 = MotorImageryB (standing, imagine sit-down)
# 4 = IdleStateB (standing, idle)
_EVENTS = {
    "imagery_sit_to_stand": 1,
    "idle_sitting": 2,
    "imagery_stand_to_sit": 3,
    "idle_standing": 4,
}

# 17 EEG channels (g.tec g.Nautilus PRO, 10-20 motor cortex coverage).
# fmt: off
_CH_NAMES = [
    "F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
    "P3", "Pz", "P4",
]
# fmt: on


class TrianaGuzman2024(BaseDataset):
    """Sit-stand motor imagery dataset from Triana-Guzman et al 2022.

    Dataset from the article *Decoding EEG Rhythms Offline and Online
    During Motor Imagery for Standing and Sitting Based on a
    Brain-Computer Interface* [1]_.

    It contains EEG data from 32 healthy subjects recorded with a
    17-channel g.tec g.Nautilus PRO system at 250 Hz. The paradigm
    involves 4 conditions:

    - **MotorImageryA**: Sitting, imagining stand-up movement
    - **IdleStateA**: Sitting, no imagery (idle)
    - **MotorImageryB**: Standing, imagining sit-down movement
    - **IdleStateB**: Standing, no imagery (idle)

    Each trial consists of 4 s fixation, ~2 s action observation,
    ~1 s preparation cue, 4 s motor imagery/idle, and 4 s rest
    (~15 s total).

    The data is hosted on OpenNeuro in BIDS format (.set files).
    Both offline and online phases are recorded in a single
    continuous file per subject. By default, only offline MI task
    markers (events 1-4) are used for epoching.

    Parameters
    ----------
    use_all_events : bool
        If True, include both MI and idle events (4 classes).
        If False (default), include only MI events (2 classes).

    References
    ----------
    .. [1] Triana-Guzman, N., Orjuela-Canon, A. D., & Jutinico, A. L.
           (2022). Decoding EEG Rhythms Offline and Online During Motor
           Imagery for Standing and Sitting Based on a Brain-Computer
           Interface. Frontiers in Neuroinformatics, 16, 961089.
           https://doi.org/10.3389/fninf.2022.961089
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=17,
            channel_types={"eeg": 17},
            montage="standard_1020",
            hardware="g.tec g.Nautilus PRO",
            sensor_type="active wet (g.LADYbird)",
            reference="right mastoid (M2)",
            ground="AFz",
            filters={"bandpass": [0.01, 60]},
            sensors=list(_CH_NAMES),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=32,
            health_status="healthy",
            gender={"female": 16, "male": 16},
            age_min=19.0,
            age_max=29.0,
            age_mean=22.4,
            handedness="mixed (29 right, 3 left)",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=list(_EVENTS.keys()),
            trial_duration=15.0,
            study_design=(
                "Sit-stand MI paradigm: 2 MI conditions "
                "(sit-to-stand, stand-to-sit) and 2 idle conditions. "
                "Offline (6 runs x 30 trials) + online BCI phase."
            ),
            feedback_type="none",
            stimulus_type="visual figure",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fninf.2022.961089",
            investigators=[
                "Nayid Triana-Guzman",
                "Alvaro D. Orjuela-Cañon",
                "Andres L. Jutinico",
                "Omar Mendoza-Montoya",
                "Javier M. Antelis",
            ],
            institution="Universidad Antonio Nariño",
            country="CO",
            data_url="https://openneuro.org/datasets/ds005342",
            publication_year=2022,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=7680,
            trials_context=("32 subjects x ~240 trials (offline, variable per subject)"),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["RLDA"],
            feature_extraction=["FBCSP", "log-variance"],
            frequency_bands={
                "theta": [4.0, 8.0],
                "alpha": [8.0, 12.0],
                "low_beta": [12.0, 16.0],
                "mid_beta": [16.0, 20.0],
                "high_beta": [20.0, 30.0],
            },
            spatial_filters=["FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold",
            cv_folds=5,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
            online_feedback=True,
        ),
        data_processed=False,
        file_format="SET (EEGLAB)",
    )

    def __init__(self, use_all_events=True, subjects=None, sessions=None):
        if use_all_events:
            events = dict(_EVENTS)
        else:
            events = {
                "imagery_sit_to_stand": 1,
                "imagery_stand_to_sit": 3,
            }

        super().__init__(
            subjects=list(range(1, 33)),
            sessions_per_subject=1,
            events=events,
            code="TrianaGuzman2024",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.3389/fninf.2022.961089",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self.use_all_events = use_all_events

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        set_path = self.data_path(subject)
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")

        # Map numeric string annotations to descriptive names.
        desc = raw.annotations.description.astype(np.dtype("<21U"))
        desc[desc == "1"] = "imagery_sit_to_stand"
        desc[desc == "2"] = "idle_sitting"
        desc[desc == "3"] = "imagery_stand_to_sit"
        desc[desc == "4"] = "idle_standing"
        raw.annotations.description = desc

        raw = stim_channels_with_selected_ids(raw, self.event_id)
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("TrianaGuzman2024", path)
        basepath = Path(path) / "MNE-trianaguzman2024-data"
        basepath.mkdir(parents=True, exist_ok=True)

        subj_str = f"sub-{subject:03d}"
        subj_dir = basepath / subj_str / "eeg"

        set_file = subj_dir / f"{subj_str}_task-sitstand_eeg.set"
        if set_file.exists() and not force_update:
            return str(set_file)

        # Download from OpenNeuro S3 (no auth required).
        subj_dir.mkdir(parents=True, exist_ok=True)

        files_to_download = [
            f"{subj_str}/eeg/{subj_str}_task-sitstand_eeg.set",
            f"{subj_str}/eeg/{subj_str}_task-sitstand_events.tsv",
            f"{subj_str}/eeg/{subj_str}_task-sitstand_eeg.json",
            f"{subj_str}/eeg/{subj_str}_task-sitstand_channels.tsv",
        ]

        import requests as _requests

        for rel_path in files_to_download:
            url = f"{_S3_BASE}/{rel_path}"
            local_path = basepath / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists() or force_update:
                log.info("Downloading %s ...", rel_path)
                resp = _requests.get(url, stream=True, timeout=120)
                if resp.status_code == 404:
                    log.warning("Not found: %s (skipping)", url)
                    continue
                resp.raise_for_status()
                with open(local_path, "wb") as fout:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fout.write(chunk)

        return str(set_file)
