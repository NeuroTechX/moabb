"""Multi-day motor imagery EEG dataset (WBCIC-SHU).

Yang, Rong, Xie, et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-04826-y
Data DOI: 10.25452/figshare.plus.22671172
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

# Figshare download URL for the single ZIP archive.
_ZIP_URL = "https://ndownloader.figshare.com/files/51001884"
_ZIP_MD5 = "e5384a58ac51b0d0c78a28a500185479"

# 59 EEG channel names (Neuracle NeuSen W 64-ch cap, 10-10 system).
# Channels 1-59: EEG, Channel 60: ECG, Channels 61-64: EOG.
# fmt: off
_CH_NAMES_EEG = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]
# fmt: on

# 2C events (51 subjects): left hand, right hand
_EVENTS_2C = {
    "left_hand": 1,
    "right_hand": 2,
}

# 3C events (11 subjects): left hand, right hand, feet
_EVENTS_3C = {
    "left_hand": 1,
    "right_hand": 2,
    "feet": 3,
}


class Yang2025(BaseDataset):
    """Multi-day MI-BCI dataset (WBCIC-SHU) from Yang et al 2025.

    Dataset from the article *A multi-day and high-quality EEG dataset
    for motor imagery brain-computer interface* [1]_.

    It contains data recorded on 62 subjects with 64-channel EEG
    (59 EEG + 1 ECG + 4 EOG) across 3 sessions on different days.
    Two paradigms were used:

    - **2C paradigm** (subjects 1-51): left hand vs right hand MI
    - **3C paradigm** (subjects 52-62): left hand, right hand, and
      foot-hooking MI

    Each session contains 5 blocks of 40 trials (2C) or 60 trials
    (3C), giving 200 or 300 trials per session.

    Trial timing: 1.5 s cue (video) + 4.0 s MI + 2.0 s rest = 7.5 s.

    The raw data is in Neuracle BDF format organized in an EEG-BIDS
    structure, hosted on Figshare (65.6 GB single ZIP file, CC-BY 4.0).

    .. note::

       Neuracle BDF files store trial events in a separate ``evt.bdf``
       file using BDF+ annotations. This adapter reads events via
       ``mne.read_annotations()`` on the ``evt.bdf`` and merges them
       into the main data BDF.

    Parameters
    ----------
    paradigm_type : str
        Which paradigm to load: ``"2C"`` (default, 51 subjects) or
        ``"3C"`` (11 subjects).

    References
    ----------
    .. [1] Yang, B., Rong, F., Xie, Y., et al. (2025). A multi-day
           and high-quality EEG dataset for motor imagery brain-computer
           interface. Scientific Data, 12, 488.
           https://doi.org/10.1038/s41597-025-04826-y
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=59,
            channel_types={"eeg": 59, "ecg": 1, "eog": 4},
            montage="standard_1005",
            hardware="Neuracle NeuSen W",
            sensor_type="Ag/AgCl",
            filters={},
            sensors=list(_CH_NAMES_EEG),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=62,
            health_status="healthy",
            gender={"female": 18, "male": 44},
            age_min=17.0,
            age_max=30.0,
            handedness="right-handed",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS_2C),
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "feet"],
            trial_duration=7.5,
            study_design=(
                "Multi-day MI-BCI: 2C (left/right hand, 51 subj) and "
                "3C (left hand, right hand, foot-hooking, 11 subj). "
                "3 sessions per subject on different days."
            ),
            feedback_type="none",
            stimulus_type="video cues",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-04826-y",
            investigators=[
                "Banghua Yang",
                "Fenqi Rong",
                "Yunlong Xie",
                "Du Li",
                "Jiayang Zhang",
                "Fu Li",
                "Guangming Shi",
                "Xiaorong Gao",
            ],
            institution="Shanghai University",
            country="CN",
            data_url="https://plus.figshare.com/articles/dataset/22671172",
            publication_year=2025,
            license="CC-BY-4.0",
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
            imagery_tasks=["left_hand", "right_hand", "feet"],
            cue_duration_s=1.5,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=39600,
            trials_context=(
                "51 subjects x 3 sessions x 200 trials (2C) + "
                "11 subjects x 3 sessions x 300 trials (3C) = 39600"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "FBCSP+SVM", "EEGNet", "deepConvNet", "FBCNet"],
            feature_extraction=["CSP", "FBCSP"],
            frequency_bands={
                "bandpass": [0.5, 40.0],
            },
            spatial_filters=["CSP", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["within_session"],
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
        paradigm_type="2C",
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        self.paradigm_type = paradigm_type

        if paradigm_type == "2C":
            subj_list = list(range(1, 52))
            events = dict(_EVENTS_2C)
        elif paradigm_type == "3C":
            subj_list = list(range(1, 12))
            events = dict(_EVENTS_3C)
        else:
            raise ValueError(f"paradigm_type must be '2C' or '3C', got {paradigm_type!r}")

        super().__init__(
            subjects=subj_list,
            sessions_per_subject=3,
            events=events,
            code="Yang2025",
            interval=[1.5, 5.5],  # MI period relative to cue onset
            paradigm="imagery",
            doi="10.1038/s41597-025-04826-y",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    # Map evt.bdf annotation codes to MOABB event names.
    _ANNOT_MAP_2C = {"1": "left_hand", "2": "right_hand"}
    _ANNOT_MAP_3C = {"1": "left_hand", "2": "right_hand", "3": "feet"}

    # Non-EEG channel types in the 64-ch Neuracle cap.
    _CH_TYPE_MAP = {
        "ECG": "ecg",
        "HEOR": "eog",
        "HEOL": "eog",
        "VEOU": "eog",
        "VEOL": "eog",
    }

    def _get_single_subject_data(self, subject):
        """Return data for a single subject from raw BDF + evt.bdf files.

        Events are read from the separate Neuracle ``evt.bdf`` file using
        ``mne.read_annotations()`` and merged into the main data BDF.
        """
        base = Path(self.data_path(subject))
        subj_str = f"sub-{subject:03d}"
        annot_map = (
            self._ANNOT_MAP_3C if self.paradigm_type == "3C" else self._ANNOT_MAP_2C
        )

        # Find the BIDS root (contains "sourcedata" folder)
        bids_root = None
        for candidate in [base, *base.iterdir()]:
            if candidate.is_dir() and (candidate / "sourcedata").exists():
                bids_root = candidate
                break
        if bids_root is None:
            raise FileNotFoundError(f"No BIDS root with sourcedata in {base}")

        source_dir = bids_root / "sourcedata" / f"{self.paradigm_type} dataset" / subj_str

        sessions = {}
        for sess_idx in range(1, 4):
            data_bdf, evt_bdf = self._find_bdf_pair(source_dir, subj_str, sess_idx)
            if data_bdf is None:
                log.warning("Missing BDF for %s session %d", subj_str, sess_idx)
                continue

            raw = mne.io.read_raw_bdf(str(data_bdf), preload=True, verbose=False)

            # Neuracle BDF files use "nV" (nanovolts) as physical dimension,
            # which MNE does not recognize — values are treated as Volts.
            # Scale all signal channels from nV to V.
            picks = mne.pick_types(raw.info, eeg=True, ecg=True, eog=True)
            raw._data[picks] *= 1e-9

            # Set proper channel types for non-EEG channels
            type_mapping = {
                ch: self._CH_TYPE_MAP[ch]
                for ch in raw.ch_names
                if ch in self._CH_TYPE_MAP
            }
            if type_mapping:
                raw.set_channel_types(type_mapping)

            # Read trial events from the separate evt.bdf
            annots = mne.read_annotations(str(evt_bdf))
            raw.set_annotations(annots)
            raw.annotations.rename(annot_map)

            sessions[str(sess_idx - 1)] = {"0": raw}

        if not sessions:
            raise FileNotFoundError(f"No BDF files found for {subj_str} in {source_dir}")
        return sessions

    @staticmethod
    def _find_bdf_pair(source_dir, subj_str, sess_idx):
        """Locate data.bdf and evt.bdf for one session.

        Handles two folder layouts:
        - 2C: ``sub-NNN/ses-NN/eeg/{data,evt}.bdf``
        - 3C: ``sub-NNN/sub-NNN_ses-NN_task-motorimagery_eeg/{data,evt}.bdf``
          (some subjects omit the dash in ``sesNN``)
        """
        # 2C-style BIDS path
        bids_dir = source_dir / f"ses-{sess_idx:02d}" / "eeg"
        if (bids_dir / "data.bdf").exists():
            return bids_dir / "data.bdf", bids_dir / "evt.bdf"

        # 3C-style Neuracle path (with or without dash in ses-NN)
        for sep in ["-", ""]:
            neuracle_dir = (
                source_dir / f"{subj_str}_ses{sep}{sess_idx:02d}_task-motorimagery_eeg"
            )
            if (neuracle_dir / "data.bdf").exists():
                return neuracle_dir / "data.bdf", neuracle_dir / "evt.bdf"

        return None, None

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Yang2025", path)
        basepath = Path(path) / "MNE-yang2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        # Check if data already extracted (look for the extracted directory)
        extracted_dir = basepath / "WBCIC_SHU Motor Imagery dataset"
        already_extracted = extracted_dir.is_dir() and any(extracted_dir.rglob("*.mat"))

        # Single 65.6 GB ZIP - download if needed.
        zip_path = basepath / "WBCIC_SHU_Motor_Imagery_dataset.zip"
        if not zip_path.exists() and not already_extracted:
            log.info("Downloading Yang2025 dataset (65.6 GB) from Figshare...")
            dl_path = dl.data_dl(
                _ZIP_URL,
                "Yang2025",
                path=str(basepath),
                force_update=force_update,
                verbose=verbose,
            )
            # Rename downloaded file to expected location.
            dl_path = Path(dl_path)
            if dl_path != zip_path:
                dl_path.rename(zip_path)

        # Extract if needed
        if zip_path.exists() and not already_extracted:
            log.info("Extracting Yang2025 dataset...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(basepath))

        return str(basepath)
