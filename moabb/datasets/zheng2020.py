"""RSVP collaborative BCI dataset.

Zheng, Sun, Zhao, et al. (2020), Frontiers in Neuroscience.
DOI: 10.3389/fnins.2020.579469
Data DOI: 10.6084/m9.figshare.12824771.v1
"""

import logging
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

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

_DOI = "10.3389/fnins.2020.579469"
_SIGN = "Zheng2020"

# Figshare file IDs for each group's zip.
_GROUP_FILES = {
    1: 24332537,
    2: 24332561,
    3: 24332573,
    4: 24332585,
    5: 24332600,
    6: 24332615,
    7: 24332633,
}

# Subject mapping: MOABB subject 1-14 -> (group, "Sa" or "Sb").
_SUBJECT_MAP = {}
for _g in range(1, 8):
    _SUBJECT_MAP[2 * _g - 1] = (_g, "Sa")
    _SUBJECT_MAP[2 * _g] = (_g, "Sb")

# Channel name fixes for standard_1020 compatibility.
_CH_RENAME = {"FZ": "Fz", "PZ": "Pz"}

# fmt: off
_CH_NAMES = [
    "FP1", "FPz", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "O1", "CB1", "Oz", "O2", "CB2",
]
# fmt: on

# Standard-case fixes.
_CH_FIX = {"FP1": "Fp1", "FP2": "Fp2", "FPz": "Fpz", "FZ": "Fz", "PZ": "Pz"}


class Zheng2020(BaseDataset):
    """RSVP collaborative BCI dataset from Zheng et al 2020.

    Dataset from the paper [1]_.

    **Dataset Description**

    Fourteen subjects (7 pairs) performed an RSVP target detection
    task across 2 sessions separated by ~23 days. EEG was recorded
    at 1000 Hz from 62 channels using two synchronized Neuroscan
    Synamps2 systems. Each session contains 3 blocks of 14 RSVP
    trials (100 images at 10 Hz), with 4 target images per trial.

    Events: Target (image containing a human) = 2,
    NonTarget (no human) = 1.

    References
    ----------
    .. [1] Zheng, L., Sun, S., Zhao, H., et al. (2020). A
           Cross-Session Dataset for Collaborative Brain-Computer
           Interfaces Based on Rapid Serial Visual Presentation.
           Frontiers in Neuroscience, 14, 579469.
           https://doi.org/10.3389/fnins.2020.579469
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="standard_1020",
            hardware="Neuroscan Synamps2",
            reference="vertex (Cz)",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            gender={"female": 10, "male": 4},
            age_mean=24.9,
            age_min=23,
            age_max=29,
            handedness="all right-handed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=1.0,
            study_design=(
                "RSVP target detection (human vs non-human images); "
                "14 subjects in 7 pairs, synchronized EEG recording"
            ),
            feedback_type="visual",
            stimulus_type="RSVP images",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Li Zheng",
                "Sen Sun",
                "Hongze Zhao",
                "Weihua Pei",
                "Hongda Chen",
                "Xiaorong Gao",
                "Lijian Zhang",
                "Yijun Wang",
            ],
            institution="Chinese Academy of Sciences",
            country="CN",
            publication_year=2020,
            data_url="https://figshare.com/articles/dataset/12824771",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=2,
        runs_per_session=3,
        tags=Tags(pathology=["Healthy"], modality=["ERP"], type=["RSVP"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300", soa_ms=100.0
        ),
        data_structure=DataStructureMetadata(
            n_trials={"target": 168, "nontarget": 4032},
            trials_context="per subject across both sessions",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["HDCA"],
            feature_extraction=["SIM", "CSP", "TRCA", "PCA"],
            frequency_bands={"bandpass": [2.0, 30.0]},
            spatial_filters=["SIM", "CSP", "PCA", "CAR", "TRCA"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="holdout", evaluation_type=["within_subject", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["target_image_detection", "collaborative_BCI"],
            environment="laboratory",
            online_feedback=True,
        ),
        data_processed=False,
        file_format="MATLAB",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=2,
            events={"Target": 2, "NonTarget": 1},
            code="Zheng2020",
            interval=[0, 1],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        self.data_path(subject)
        group, ab = _SUBJECT_MAP[subject]

        base = Path(dl.get_dataset_path(_SIGN, None)) / f"MNE-{_SIGN}-data"
        sessions = {}

        for ses_idx in range(1, 3):
            mat_name = f"G{group}D{ses_idx}.mat"
            mat_path = base / f"G{group}" / mat_name
            if not mat_path.exists():
                continue

            data = loadmat(str(mat_path), squeeze_me=False)
            key = "Sa" if ab == "Sa" else "Sb"

            if key not in data:
                log.warning("Key %s not found in %s", key, mat_path)
                continue

            blocks = data[key]  # (1, n_blocks) cell array
            runs = {}

            for block_idx in range(blocks.shape[1]):
                block = blocks[0, block_idx]  # (63, n_samples)
                if block.ndim != 2 or block.shape[0] < 63:
                    continue

                eeg = block[:62, :].astype(np.float64)
                trig = block[62, :].astype(np.float64)

                # Scale: Neuroscan data is in uV.
                eeg = eeg * 1e-6

                # Build stim channel: trig==1 -> Target=2, trig==2 -> NonTarget=1
                # (in this dataset: 1=target image, 2=non-target image)
                stim = np.zeros(eeg.shape[1])
                stim[trig == 1] = 2  # Target
                stim[trig == 2] = 1  # NonTarget

                all_data = np.vstack([eeg, stim[np.newaxis]])

                ch_names = [_CH_FIX.get(ch, ch) for ch in _CH_NAMES] + ["STI"]
                ch_types = ["eeg"] * 62 + ["stim"]
                info = mne.create_info(ch_names, 1000.0, ch_types)
                raw = mne.io.RawArray(all_data, info, verbose=False)
                raw.set_montage("standard_1020", on_missing="warn")

                runs[str(block_idx)] = raw

            if runs:
                sessions[str(ses_idx - 1)] = runs

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        group, _ = _SUBJECT_MAP[subject]
        base = Path(dl.get_dataset_path(_SIGN, None)) / f"MNE-{_SIGN}-data"
        group_dir = base / f"G{group}"

        # Check if already extracted.
        mat1 = group_dir / f"G{group}D1.mat"
        mat2 = group_dir / f"G{group}D2.mat"
        if mat1.exists() and mat2.exists() and not force_update:
            return str(group_dir)

        # Download zip from Figshare.
        file_id = _GROUP_FILES[group]
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        zip_path = dl.data_dl(url, _SIGN)

        # Extract.
        group_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".mat"):
                    fname = Path(member).name
                    target = group_dir / fname
                    if not target.exists() or force_update:
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())

        return str(group_dir)
