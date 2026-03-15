"""Continuous pursuit EEG-BCI dataset for online deep learning.

Forenzo, Zhu, and He (2024), Scientific Data.
DOI: 10.1038/s41597-024-04090-6
Data DOI: 10.1184/R1/25360300
"""

import logging
from pathlib import Path

import mne
import numpy as np
from pymatreader import read_mat

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

# KiltHub (CMU Figshare) download URLs per subject.
_FIGSHARE_BASE = "https://ndownloader.figshare.com/files/"
_FILE_IDS = {
    1: 45214414,
    2: 45214594,
    3: 45214612,
    4: 45215872,
    5: 45217489,
    6: 45218278,
    7: 45218578,
    8: 45218743,
    9: 45218953,
    10: 45219391,
    11: 45219436,
    12: 45219460,
    13: 45219478,
    14: 45219511,
    15: 45219529,
    16: 45219532,
    17: 45219622,
    18: 45220030,
    19: 45220501,
    20: 45220864,
    21: 45221296,
    22: 45221695,
    23: 45221752,
    24: 45221755,
    25: 45221767,
    26: 45214207,
    27: 45214231,
    28: 45214243,
}

# Event mapping for the continuous pursuit paradigm.
# Each 60s trial has a target that can be in 4 quadrants.
# For MOABB: targets 1=left, 2=right, 3=up, 4=down
_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
}

# Standard Neuroscan 62-channel names (64 cap minus M1/M2 mastoids).
# fmt: off
_CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz",
    "O2", "CB2",
]
# fmt: on

_SFREQ = 1000.0


class Forenzo2024(BaseDataset):
    """Continuous pursuit MI-BCI dataset from Forenzo et al 2024.

    Dataset from the article *A continuous pursuit dataset for online
    deep learning-based EEG brain-computer interface* [1]_.

    It contains EEG data from 28 subjects recorded with a 62-channel
    Neuroscan system (64-ch cap minus M1/M2 mastoids) across multiple
    sessions. Two sub-studies were conducted:

    - **Sub-study 1** (subjects 1-14): 8 sessions
    - **Sub-study 2** (subjects 15-28): 4 sessions

    Each session has multiple 60-second continuous pursuit runs with
    different decoders (AR=traditional, EG=EEGNet, PN=PointNet,
    TL=transfer learning, CL=calibration).

    **Note**: The .mat files use MATLAB v7.3 (HDF5) format and require
    the ``mat73`` or ``h5py`` package for loading.

    References
    ----------
    .. [1] Forenzo, D., Zhu, H., & He, B. (2024). A continuous pursuit
           dataset for online deep learning-based EEG brain-computer
           interface. Scientific Data, 11, 1236.
           https://doi.org/10.1038/s41597-024-04090-6
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="standard_1005",
            hardware="Neuroscan Quik-Cap 64-ch, SynAmps/RT",
            sensor_type="Ag/AgCl",
            filters={"bandpass": [0.1, 200], "notch_hz": 60},
            sensors=list(_CH_NAMES),
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=28,
            health_status="healthy",
            gender={"female": 12, "male": 16},
            age_mean=23.67,
            handedness="right-handed (27 of 28)",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=60.0,
            study_design=(
                "Continuous pursuit MI-BCI with multiple decoders "
                "(traditional, EEGNet, PointNet). 2 sub-studies: "
                "S01-S14 (8 sessions), S15-S28 (4 sessions)."
            ),
            feedback_type="cursor",
            stimulus_type="continuous pursuit",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-024-04090-6",
            investigators=["Dylan Forenzo", "Hao Zhu", "Bin He"],
            institution="Carnegie Mellon University",
            country="US",
            data_url="https://kilthub.cmu.edu/articles/dataset/25360300",
            publication_year=2024,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=8,
        runs_per_session=12,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            imagery_duration_s=60.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=10080,
            trials_context=(
                "14 subjects x 8 sessions x 12 runs x 5 trials + "
                "14 subjects x 4 sessions x 12 runs x 5 trials"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["AR_linear_decoder", "EEGNet", "PointNet"],
            feature_extraction=["AR_spectral_estimation", "deep_learning"],
            frequency_bands={
                "alpha_mu": [8.0, 13.0],
            },
            spatial_filters=["Laplacian", "CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["cursor_control"],
            environment="laboratory",
            online_feedback=True,
        ),
        data_processed=False,
        file_format="MAT (v7.3/HDF5)",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 29)),
            sessions_per_subject=8,
            events=dict(_EVENTS),
            code="Forenzo2024",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-024-04090-6",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))
        subj_dir = base / f"S{subject:02d}"
        if not subj_dir.exists():
            subj_dir = base

        sessions = {}
        mat_files = sorted(subj_dir.rglob("*.mat"))

        if not mat_files:
            raise FileNotFoundError(
                f"No .mat files found for subject {subject} in {subj_dir}"
            )

        # Group mat files by session
        session_files = {}
        for mf in mat_files:
            # Try to extract session number from filename.
            # Expected pattern: S##_Se##_R##_decoder.mat
            parts = mf.stem.split("_")
            sess_num = None
            for p in parts:
                if p.startswith("Se") and p[2:].isdigit():
                    sess_num = int(p[2:])
                    break
            if sess_num is None:
                sess_num = 1
            session_files.setdefault(sess_num, []).append(mf)

        for sess_num, files in sorted(session_files.items()):
            runs = {}
            for run_idx, mf in enumerate(sorted(files)):
                try:
                    raw = self._load_hdf5_mat(mf)
                    runs[str(run_idx)] = raw
                except Exception as e:
                    log.warning("Failed to load %s: %s", mf.name, e)
            if runs:
                sessions[str(sess_num - 1)] = runs

        if not sessions:
            raise FileNotFoundError(f"No loadable data for subject {subject}")
        return sessions

    def _load_hdf5_mat(self, mat_path):
        """Load a MATLAB v7.3 (HDF5) .mat file into MNE Raw."""
        mat = read_mat(str(mat_path))

        # Extract EEG data from nested 'eeg' struct or top-level keys
        if "eeg" in mat and isinstance(mat["eeg"], dict):
            data = np.asarray(mat["eeg"]["data"])
        else:
            data = np.asarray(mat.get("data", mat.get("eeg")))

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # pymatreader returns MATLAB's (n_samples, n_channels); ensure
        # shape is (n_channels, n_samples).
        if data.shape[0] != len(_CH_NAMES) and data.shape[1] == len(_CH_NAMES):
            data = data.T

        n_ch = data.shape[0]

        # Use standard channel names if channel count matches
        if n_ch == len(_CH_NAMES):
            ch_names = list(_CH_NAMES)
        else:
            ch_names = [f"EEG{i + 1}" for i in range(n_ch)]

        ch_types = ["eeg"] * n_ch + ["stim"]
        info = mne.create_info(
            ch_names=ch_names + ["STI"],
            ch_types=ch_types,
            sfreq=_SFREQ,
        )

        # Build stim channel from events
        stim = np.zeros((1, data.shape[1]))

        # Scale to volts if in microvolts
        if np.abs(data).max() > 1e-3:
            data = data * 1e-6

        full_data = np.concatenate([data, stim], axis=0)
        raw = mne.io.RawArray(data=full_data, info=info, verbose=False)

        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"S{subject:02d}"

        if subj_dir.exists() and list(subj_dir.rglob("*.mat")):
            return str(data_dir)

        # Download per-subject ZIP from KiltHub and extract.
        file_id = _FILE_IDS.get(subject)
        if file_id is None:
            raise ValueError(f"No download URL for subject {subject}")

        url = f"{_FIGSHARE_BASE}{file_id}"
        download_and_extract_subject_zip(url, sign, data_dir, path, force_update, verbose)

        return str(data_dir)
