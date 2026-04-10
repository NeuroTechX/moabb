"""Visual imagery EEG dataset.

Gao, Liu, Li, Huang, Wang, Xu, Zhao, Li, and Fu (2026), Scientific Data.
DOI: 10.1038/s41597-025-06512-5
Data DOI: 10.6084/m9.figshare.30227503.v1
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
from .utils import download_and_extract_subject_zip


log = logging.getLogger(__name__)

# Per-subject Figshare file IDs (ndownloader.figshare.com/files/{id}).
_FIGSHARE_FILE_IDS = {
    1: 58328968,
    2: 58329043,
    3: 58329064,
    4: 58329169,
    5: 58329229,
    6: 58329265,
    7: 58329289,
    8: 58424254,
    9: 58336849,
    10: 58336852,
    11: 58337044,
    12: 58337038,
    13: 58337278,
    14: 58337281,
    15: 58337410,
    16: 58337413,
    17: 58337728,
    18: 58337734,
    19: 58338067,
    20: 58338136,
    21: 58338541,
    22: 58338544,
}

# Task codes and their per-task event mappings.
# Event codes in the BDF Status channel are local (1-4) per task file.
# We remap to globally unique codes.
_TASKS = {
    "AVI": {"dog": 1, "bird": 2, "fish": 3},
    "FVI": {"pentagram": 11, "square": 12, "circle": 13},
    "OVI": {"scissor": 21, "watch": 22, "cup": 23, "chair": 24},
}

# Local (Status channel) -> global code mapping per task.
_LOCAL_TO_GLOBAL = {
    "AVI": {1: 1, 2: 2, 3: 3},
    "FVI": {1: 11, 2: 12, 3: 13},
    "OVI": {1: 21, 2: 22, 3: 23, 4: 24},
}

# Flat event dict combining all tasks.
_EVENTS = {}
for _task_events in _TASKS.values():
    _EVENTS.update(_task_events)

# 32 EEG channel names (standard 10-20).
_CH_NAMES = [
    "Fpz",
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "FCz",
    "FC3",
    "FC4",
    "FT7",
    "FT8",
    "Cz",
    "C3",
    "C4",
    "T7",
    "T8",
    "CP3",
    "CP4",
    "TP7",
    "TP8",
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "PO3",
    "PO4",
    "Oz",
    "O1",
    "O2",
]

# Subjects with only 1 session (personal reasons per README).
_SINGLE_SESSION_SUBJECTS = {9, 10}


class Gao2026(BaseDataset):
    """Visual imagery EEG dataset from Gao et al 2026.

    Dataset from the article *An EEG Dataset for Visual Imagery-Based
    Brain-Computer Interface* [1]_. Data hosted on Figshare [2]_.

    **Note**: This is a **visual** imagery dataset (not motor imagery).
    Participants imagined visual stimuli across three categories: animals,
    geometric figures, and everyday objects.

    EEG was recorded from 22 healthy participants using a 32-channel
    Neuracle NeuSenW32 wireless cap at 1000 Hz, with Cpz reference.

    Three task categories were recorded per session:

    - **Animals (AVI)**: dog, bird, fish (3 classes, 40 trials each)
    - **Figures (FVI)**: pentagram, square, circle (3 classes, 40 trials each)
    - **Objects (OVI)**: scissor, watch, cup, chair (4 classes, 40 trials each)

    Each trial: 3 s fixation, 4 s image presentation, 4 s imagery period
    (the epoch of interest), then rest. Events mark imagery onset.

    Most subjects completed 2 sessions. Subjects 9 and 10 completed only
    session 1.

    References
    ----------
    .. [1] Gao, J., Liu, Y., Li, Z., Huang, K., Wang, F., Xu, J.,
           Zhao, L., Li, T., & Fu, Y. (2026). An EEG Dataset for Visual
           Imagery-Based Brain-Computer Interface. Scientific Data.
           https://doi.org/10.1038/s41597-025-06512-5

    .. [2] Gao, J. et al. (2026). EEG Dataset for Visual Imagery.
           Figshare. https://doi.org/10.6084/m9.figshare.30227503.v1
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="standard_1005",
            hardware="Neuracle NeuSenW32",
            sensor_type="Ag/AgCl",
            reference="CPz",
            ground="AFz",
            filters={"sampling_rate": 1000},
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=22,
            health_status="healthy",
            gender={"male": 17, "female": 5},
            age_mean=None,
            age_min=20.0,
            age_max=23.0,
            handedness=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=10,
            class_labels=list(_EVENTS.keys()),
            trial_duration=4.0,
            study_design=(
                "Visual imagery of animals, figures, and objects "
                "with simultaneous 32-channel EEG recording"
            ),
            feedback_type="none",
            stimulus_type="image cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-06512-5",
            investigators=[
                "Jing'ao Gao",
                "Yao Liu",
                "Zhengshuang Li",
                "Kaixin Huang",
                "Fan Wang",
                "Jiaping Xu",
                "Lei Zhao",
                "Tianwen Li",
                "Yunfa Fu",
            ],
            institution="Kunming University of Science and Technology",
            country="CN",
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.30227503.v1",
            publication_year=2026,
            license="CC-BY-NC-ND-4.0",
        ),
        sessions_per_subject=2,
        runs_per_session=3,
        tags=Tags(pathology=["Healthy"], modality=["Visual"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", imagery_tasks=list(_EVENTS.keys())
        ),
        data_structure=DataStructureMetadata(
            n_trials=16800,
            trials_context=(
                "20 subjects x 2 sessions x 400 trials + "
                "2 subjects x 1 session x 400 trials = 16800"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["EEGNet", "CSP+KNN"],
            feature_extraction=["CSP", "deep_learning"],
            frequency_bands={"bandpass": [5.0, 30.0]},
            spatial_filters=["CSP", "CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="train-test split", evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["human_machine_interaction"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="BDF",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 23)),
            sessions_per_subject=2,
            events=dict(_EVENTS),
            code="Gao2026",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-06512-5",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))
        subj_str = f"sub-{subject:02d}"

        n_sessions = 1 if subject in _SINGLE_SESSION_SUBJECTS else 2
        sessions = {}

        for ses_idx in range(1, n_sessions + 1):
            ses_str = f"ses-{ses_idx:02d}"
            eeg_dir = base / subj_str / ses_str / "eeg"

            if not eeg_dir.exists():
                log.warning("Missing session directory: %s", eeg_dir)
                continue

            runs = {}
            for run_idx, (task, local_to_global) in enumerate(_LOCAL_TO_GLOBAL.items()):
                bdf_file = eeg_dir / f"{subj_str}_{ses_str}_task-{task}_eeg.bdf"
                if not bdf_file.exists():
                    log.warning("Missing BDF file: %s", bdf_file)
                    continue

                raw = mne.io.read_raw_bdf(str(bdf_file), preload=True, verbose=False)

                # Neuracle BDF header has 16-bit digital min/max (-32768/32767)
                # for 24-bit BDF data, making amplitudes 256x too large.
                eeg_picks = mne.pick_types(raw.info, eeg=True)
                raw._data[eeg_picks] /= 256

                # Pick only EEG channels (drop Status).
                raw.pick(["eeg"])

                # Set standard montage.
                montage = mne.channels.make_standard_montage("standard_1005")
                raw.set_montage(montage, on_missing="warn")

                # Extract events from Status channel of the original file.
                raw_with_status = mne.io.read_raw_bdf(
                    str(bdf_file), preload=True, verbose=False
                )
                stim_data = raw_with_status.get_data(picks="Status")[0].astype(int)

                # Find baseline (most frequent value: 0 or 131071).
                vals, counts = np.unique(stim_data, return_counts=True)
                baseline = vals[np.argmax(counts)]

                # Find event onsets (first sample of each non-baseline group).
                non_bl = np.where(stim_data != baseline)[0]
                if len(non_bl) == 0:
                    log.warning("No events found in %s", bdf_file)
                    continue

                diffs = np.diff(non_bl)
                group_starts = np.concatenate([[0], np.where(diffs > 1)[0] + 1])
                event_samples = non_bl[group_starts]
                local_codes = stim_data[event_samples]

                # Build MNE events array with global codes.
                events_list = []
                for sample, local_code in zip(event_samples, local_codes):
                    if local_code in local_to_global:
                        events_list.append([sample, 0, local_to_global[local_code]])

                if not events_list:
                    log.warning("No valid events after remapping in %s", bdf_file)
                    continue

                events_arr = np.array(events_list, dtype=int)

                # Create annotations from events.
                global_to_name = {v: k for k, v in _TASKS[task].items()}
                event_desc = {
                    code: global_to_name[code] for code in np.unique(events_arr[:, 2])
                }
                annot = mne.annotations_from_events(
                    events=events_arr,
                    event_desc=event_desc,
                    sfreq=raw.info["sfreq"],
                    orig_time=raw.info["meas_date"],
                )
                raw.set_annotations(annot)

                runs[str(run_idx)] = raw

            if runs:
                sessions[str(ses_idx - 1)] = runs

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}. Valid: {self.subject_list}")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_str = f"sub-{subject:02d}"
        subj_dir = data_dir / subj_str

        # Return cached if already extracted.
        if subj_dir.is_dir() and not force_update:
            bdf_files = list(subj_dir.rglob("*.bdf"))
            if bdf_files:
                return str(data_dir)

        # Download per-subject ZIP from Figshare and extract.
        file_id = _FIGSHARE_FILE_IDS[subject]
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        download_and_extract_subject_zip(url, sign, data_dir, path, force_update, verbose)

        if not list(subj_dir.rglob("*.bdf")):
            raise FileNotFoundError(
                f"No BDF files found for subject {subject} in {subj_dir}"
            )
        return str(data_dir)
