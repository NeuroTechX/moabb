"""Multi-paradigm upper limb rehabilitation EEG dataset.

Chang, Kong, Yan, Lv, and Du (2025), Scientific Data.
DOI: 10.1038/s41597-025-06147-6
Data DOI: 10.6084/m9.figshare.28831730.v2
"""

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
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

# Figshare download URLs for raw data ZIPs (article 28831730).
_FIGSHARE_BASE = "https://ndownloader.figshare.com/files/"
_ZIP_FILES = {
    # ZIP name -> (figshare_file_id, subjects_in_zip)
    "Raw_EEG_Data_01_05": (57432889, list(range(1, 6))),
    "Raw_EEG_Data_06_10": (57433693, list(range(6, 11))),
    "Raw_EEG_Data_11_15": (57435691, list(range(11, 16))),
    "Raw_EEG_Data_15_20": (57435694, list(range(16, 21))),
    "Raw_EEG_Data_21_25": (57560707, list(range(21, 26))),
    "Raw_EEG_Data_26_30": (57560710, list(range(26, 31))),
}

# Subject IDs: S201-S230 excluding S207 and S218 (28 usable).
# MOABB subjects 1-28 map to these original IDs.
_ORIG_IDS = [f"S{i}" for i in range(201, 231) if i not in (207, 218)]

# Event codes (embedded as EEGLAB annotations in .set files).
_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
    "both_hands": 3,
}

# Six paradigm types, each with N sessions per subject.
_PARADIGM_SESSIONS = {
    "ME": 4,  # Motor Execution
    "MI": 4,  # Motor Imagery
    "VR-MI": 4,  # VR Motor Imagery
    "MT": 4,  # Mirror Therapy (2 classes only: L/R)
    "SRG": 6,  # Soft Rehab Gloves
    "MG": 4,  # Mirror Glove (2 classes only: L/R)
}


class Chang2025(BaseDataset):
    """Multi-paradigm upper limb rehabilitation dataset from Chang et al 2025.

    Dataset from the article *A multi-paradigm EEG dataset for studying
    upper limb rehabilitation exercises* [1]_.

    It contains EEG data from 28 healthy subjects recorded with a
    59-channel Neuracle BRK-NSW 2.0 system at 1000 Hz. Six rehabilitation
    paradigms were tested:

    - **ME**: Motor Execution (grasp-and-release)
    - **MI**: Motor Imagery (imagine grasp-and-release)
    - **VR-MI**: VR-assisted Motor Imagery
    - **MT**: Mirror Therapy (left/right only)
    - **SRG**: Soft Rehabilitation Gloves
    - **MG**: Mirror Glove (left/right only)

    Each paradigm has multiple sessions. Trial structure:
    fixation + cue + 4 s task + 2 s rest = ~6 s per trial,
    40 trials per class per paradigm.

    By default, only the **MI** paradigm is loaded (4 sessions,
    3 classes: left/right/both hands).

    Parameters
    ----------
    paradigm_type : str
        Which paradigm to load. One of ``"MI"`` (default), ``"ME"``,
        ``"VR-MI"``, ``"MT"``, ``"SRG"``, ``"MG"``.

    References
    ----------
    .. [1] Chang, W., Kong, W., Yan, G., Lv, R., Du, K. (2025).
           A multi-paradigm EEG dataset for studying upper limb
           rehabilitation exercises. Scientific Data, 12, 1877.
           https://doi.org/10.1038/s41597-025-06147-6
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=59,
            channel_types={"eeg": 59},
            montage="standard_1005",
            hardware="Neuracle BRK-NSW 2.0",
            sensor_type="Ag/AgCl",
            filters={},
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=28,
            health_status="healthy",
            gender={"female": 14, "male": 14},
            age_min=20.0,
            age_max=33.0,
            handedness="right-handed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=3,
            class_labels=list(_EVENTS.keys()),
            trial_duration=6.0,
            study_design=(
                "6 rehabilitation paradigms (ME, MI, VR-MI, MT, SRG, MG). "
                "Default: MI paradigm with 3 classes (L/R/both hands), "
                "4 sessions, 40 trials/class."
            ),
            feedback_type="none",
            stimulus_type="visual cue (square)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-06147-6",
            investigators=[
                "Wenwen Chang",
                "Weixuan Kong",
                "Guanghui Yan",
                "Renjie Lv",
                "Kaiyue Du",
                "Muhammad Tariq Sadiq",
                "Bin Guo",
                "Rong Yin",
                "Xuan Liu",
            ],
            institution="Lanzhou Jiaotong University",
            institution_department="School of Electronic and Information Engineering",
            country="CN",
            data_url="https://figshare.com/articles/dataset/28831730",
            publication_year=2025,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=4,
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
            n_trials=13440,
            trials_context=(
                "28 subjects x 4 MI sessions x 3 classes x 40 trials = 13440"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "FBCSP+SVM"],
            feature_extraction=["CSP", "FBCSP"],
            frequency_bands={
                "alpha": [8.0, 13.0],
                "FBCSP_range": [4.0, 28.0],
            },
            spatial_filters=["CSP", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="SET (EEGLAB)",
    )

    def __init__(
        self,
        paradigm_type="MI",
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        self.paradigm_type = paradigm_type

        if paradigm_type not in _PARADIGM_SESSIONS:
            raise ValueError(
                f"paradigm_type must be one of {list(_PARADIGM_SESSIONS.keys())}, "
                f"got {paradigm_type!r}"
            )

        n_sessions = _PARADIGM_SESSIONS[paradigm_type]

        # MT and MG only have 2 classes (left/right).
        if paradigm_type in ("MT", "MG"):
            events = {"left_hand": 1, "right_hand": 2}
        else:
            events = dict(_EVENTS)

        super().__init__(
            subjects=list(range(1, 29)),
            sessions_per_subject=n_sessions,
            events=events,
            code="Chang2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-06147-6",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))
        orig_id = _ORIG_IDS[subject - 1]
        subj_dir = base / orig_id

        if not subj_dir.exists():
            # Try alternative directory structures.
            found = False
            if base.exists():
                for candidate in base.iterdir():
                    if candidate.is_dir() and orig_id in candidate.name:
                        subj_dir = candidate
                        found = True
                        break
            if not found:
                log.warning(
                    "Subject directory for %s not found under %s. "
                    "The upstream data may be missing for this subject. Skipping.",
                    orig_id,
                    base,
                )
                return {}

        # Find .set files for the selected paradigm type.
        set_files = sorted(subj_dir.rglob("*.set"))
        if not set_files:
            log.warning(
                "No .set files found for %s in %s. "
                "The download may be incomplete or the upstream data may be "
                "missing for this subject. Skipping.",
                orig_id,
                subj_dir,
            )
            return {}

        # Filter files by paradigm type.
        # File naming: {orig_id}_{prefix}{session_num}.set
        # MI→"_MI\d+", ME→"_ME\d+", VR-MI→"_VR\d+",
        # MT→"_Mirror_", SRG→"_Aux_", MG→"_Image_"
        _PARADIGM_FILE_PATTERNS = {
            "MI": re.compile(r"_MI\d+", re.IGNORECASE),
            "ME": re.compile(r"_ME\d+", re.IGNORECASE),
            "VR-MI": re.compile(r"_VR\d+", re.IGNORECASE),
            "MT": re.compile(r"_Mirror_", re.IGNORECASE),
            "SRG": re.compile(r"_Aux_", re.IGNORECASE),
            "MG": re.compile(r"_Image_", re.IGNORECASE),
        }

        paradigm_files = []
        pt = self.paradigm_type
        pattern = _PARADIGM_FILE_PATTERNS[pt]
        for sf in set_files:
            if pattern.search(sf.stem):
                paradigm_files.append(sf)

        if not paradigm_files:
            # Fallback: use all files if paradigm filtering failed.
            log.warning(
                "Could not filter %s files by paradigm '%s' for %s. "
                "Using all %d .set files.",
                len(set_files),
                pt,
                orig_id,
                len(set_files),
            )
            paradigm_files = set_files

        sessions = {}
        for sess_idx, sf in enumerate(paradigm_files):
            try:
                raw = mne.io.read_raw_eeglab(str(sf), preload=True, verbose="ERROR")

                # Map event annotations to descriptive names.
                desc = raw.annotations.description.astype(np.dtype("<15U"))
                desc[desc == "1"] = "left_hand"
                desc[desc == "2"] = "right_hand"
                desc[desc == "3"] = "both_hands"
                raw.annotations.description = desc

                raw = stim_channels_with_selected_ids(raw, self.event_id)
                sessions[str(sess_idx)] = {"0": raw}
            except Exception as e:
                log.warning("Failed to load %s: %s", sf.name, e)

        if not sessions:
            log.warning(
                "No loadable %s session data for %s. "
                "All .set files failed to load. Skipping.",
                pt,
                orig_id,
            )
            return {}
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Chang2025", path)
        basepath = Path(path) / "MNE-chang2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        orig_id = _ORIG_IDS[subject - 1]
        subj_dir = basepath / orig_id

        # Check if data already extracted.
        if subj_dir.exists() and list(subj_dir.rglob("*.set")):
            return str(basepath)

        # Find which ZIP contains this subject.
        target_zip = None
        for zip_name, (file_id, subj_range) in _ZIP_FILES.items():
            if subject in subj_range:
                target_zip = (zip_name, file_id)
                break

        if target_zip is None:
            raise ValueError(f"No download ZIP found for subject {subject}")

        zip_name, file_id = target_zip
        url = f"{_FIGSHARE_BASE}{file_id}"

        log.info("Downloading Chang2025 %s for subject %d ...", zip_name, subject)
        local_zip = dl.data_dl(
            url,
            "Chang2025",
            path=str(basepath),
            force_update=force_update,
            verbose=verbose,
        )

        # Extract the ZIP.
        local_zip_path = Path(local_zip)
        if local_zip_path.exists() and not subj_dir.exists():
            log.info("Extracting %s ...", local_zip_path.name)
            with zipfile.ZipFile(str(local_zip_path), "r") as zf:
                zf.extractall(str(basepath))

        return str(basepath)
