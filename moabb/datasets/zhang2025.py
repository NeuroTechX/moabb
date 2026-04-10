"""RSVP ERP dataset for EEG-based identity authentication.

Zhang, Zhang, Li, Wang, Gao, and Yang (2025), Scientific Data.
DOI: 10.1038/s41597-025-05378-x
Data DOI: 10.6084/m9.figshare.27201003.v1
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


log = logging.getLogger(__name__)

_DOI = "10.1038/s41597-025-05378-x"
_SIGN = "Zhang2025"

# Figshare file IDs for S1.mat through S15.mat.
_FILE_IDS = {
    1: 49718502,
    2: 49718661,
    3: 49719249,
    4: 49719288,
    5: 49720080,
    6: 49720824,
    7: 49721490,
    8: 49721532,
    9: 49721652,
    10: 49721724,
    11: 49721979,
    12: 49723263,
    13: 49723317,
    14: 49723386,
    15: 49724028,
}

# Session keys in the HDF5 files (Group A: 4 sessions).
_SESSION_KEYS = ["Day_1", "Day_7", "Day_80", "Day_200"]

# fmt: off
_CH_NAMES = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO7", "PO8",
    "Oz", "O1", "O2",
]
# fmt: on


class Zhang2025(BaseDataset):
    """RSVP ERP dataset for authentication from Zhang et al 2025.

    Dataset from the paper [1]_.

    **Dataset Description**

    Fifteen subjects (Group A) completed 4 RSVP sessions on days 1,
    7, 80, and 200, viewing sequences of face images at 10 Hz (100 ms
    per image, 200 images per sequence). Target images are the
    subject's own face; non-target images are AI-generated faces.

    EEG was recorded at 1000 Hz from 57 channels using a Neuracle
    system. Events: Target (self-face, code 2) = Target ERP,
    NonTarget (other face, code 1) = NonTarget.

    Each session has 4 blocks of 8 RSVP sequences. The data is
    stored as HDF5 .mat files (~3-4 GB each, ~52 GB total).

    References
    ----------
    .. [1] Zhang, Y., Zhang, H., Li, Y., Wang, Y., Gao, X., &
           Yang, C. (2025). A longitudinal EEG dataset of
           event-related potential for EEG-based identity
           authentication. Scientific Data, 12, 1069.
           https://doi.org/10.1038/s41597-025-05378-x
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=57,
            channel_types={"eeg": 57},
            montage="standard_1020",
            hardware="Neuracle Neusen",
            reference="CPz",
            ground="AFz",
            sensors=list(_CH_NAMES),
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            gender={"female": 6, "male": 9},
            age_min=22,
            age_max=26,
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
                "RSVP face authentication; self-face vs AI-generated faces; "
                "4 sessions over 200 days (longitudinal)"
            ),
            feedback_type="none",
            stimulus_type="RSVP face images",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            investigators=[
                "Yufeng Zhang",
                "Hongxin Zhang",
                "Yixuan Li",
                "Yijun Wang",
                "Xiaorong Gao",
                "Chen Yang",
            ],
            institution="Beijing University of Posts and Telecommunications",
            country="CN",
            publication_year=2025,
            data_url="https://figshare.com/articles/dataset/27201003",
            license="CC-BY-NC-ND-4.0",
        ),
        sessions_per_subject=4,
        runs_per_session=4,
        tags=Tags(pathology=["Healthy"], modality=["ERP"], type=["RSVP"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300", soa_ms=100.0
        ),
        data_structure=DataStructureMetadata(
            n_trials="~160 target + ~6240 nontarget per session",
            trials_context="per session (4 blocks x 8 sequences x 200 images)",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["HDCA"],
            feature_extraction=["HDCA"],
            frequency_bands={"ERP_dominant": [0.0, 10.0]},
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        bci_application=BCIApplicationMetadata(
            applications=["identity_authentication", "target_detection"],
            environment="laboratory",
            online_feedback=None,
        ),
        data_processed=False,
        file_format="MATLAB (HDF5)",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=4,
            events={"Target": 2, "NonTarget": 1},
            code="Zhang2025",
            interval=[0, 0.6],
            paradigm="p300",
            doi=_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}}."""
        mat_path = self.data_path(subject)

        mat = read_mat(mat_path)
        sessions = {}

        for ses_idx, ses_key in enumerate(_SESSION_KEYS):
            if ses_key not in mat:
                continue

            # pymatreader resolves the cell array of blocks to a list.
            ses_data = mat[ses_key]
            if not isinstance(ses_data, list):
                ses_data = [ses_data]

            runs = {}
            for block_idx, block in enumerate(ses_data):
                block = np.asarray(block)
                if block.ndim != 2:
                    continue

                # Rows 0-56: EEG, Row 57: trigger.
                # pymatreader may return (n_samples, 58) — handle both.
                if block.shape[0] == 58:
                    eeg = block[:57, :].astype(np.float64)
                    trig = block[57, :].astype(np.float64)
                elif block.shape[1] == 58:
                    eeg = block[:, :57].T.astype(np.float64)
                    trig = block[:, 57].astype(np.float64)
                else:
                    log.warning(
                        "Unexpected shape %s in %s/%s block %d",
                        block.shape,
                        ses_key,
                        subject,
                        block_idx,
                    )
                    continue

                # Scale to Volts (data is in uV).
                eeg = eeg * 1e-6

                # Build stim channel.
                # Per MATLAB code: trigger==1 -> non-target, trigger==2 -> target.
                stim = np.zeros(eeg.shape[1])
                stim[trig == 1] = 1  # NonTarget
                stim[trig == 2] = 2  # Target

                all_data = np.vstack([eeg, stim[np.newaxis]])
                ch_names = list(_CH_NAMES) + ["STI"]
                ch_types = ["eeg"] * 57 + ["stim"]
                info = mne.create_info(ch_names, 1000.0, ch_types)
                raw = mne.io.RawArray(all_data, info, verbose=False)
                raw.set_montage("standard_1020", on_missing="warn")

                runs[str(block_idx)] = raw

            if runs:
                sessions[str(ses_idx)] = runs

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        base = Path(dl.get_dataset_path(_SIGN, None)) / f"MNE-{_SIGN}-data"
        base.mkdir(parents=True, exist_ok=True)

        fname = f"S{subject}.mat"
        local = base / fname

        if not local.exists() or force_update:
            file_id = _FILE_IDS[subject]
            url = f"https://ndownloader.figshare.com/files/{file_id}"
            downloaded = dl.data_dl(url, _SIGN)
            downloaded = Path(downloaded)
            if downloaded != local:
                downloaded.rename(local)

        return str(local)
