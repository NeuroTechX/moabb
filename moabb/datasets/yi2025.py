"""Multi-joint upper-limb motor imagery EEG dataset.

Yi, Chen, Wang, et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-05286-0
Data DOI: 10.6084/m9.figshare.24123303.v3
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

# Event codes for 8 multi-joint MI classes.
_EVENTS = {
    "hand_open_close": 1,
    "wrist_flex_ext": 2,
    "wrist_abd_add": 3,
    "elbow_pron_sup": 4,
    "elbow_flex_ext": 5,
    "shoulder_pron_sup": 6,
    "shoulder_abd_add": 7,
    "shoulder_flex_ext": 8,
}

# 62 EEG channels (same as raw .cnt after dropping aux channels).
# fmt: off
_CH_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "CB1", "O1", "Oz", "O2", "CB2",
]
# fmt: on

# Raw .cnt files use uppercase; these need case correction for standard_1005.
_CH_RENAME = {
    "FP1": "Fp1",
    "FPZ": "Fpz",
    "FP2": "Fp2",
    "FZ": "Fz",
    "FCZ": "FCz",
    "CZ": "Cz",
    "CPZ": "CPz",
    "PZ": "Pz",
    "POZ": "POz",
    "OZ": "Oz",
}

# Non-EEG channels to set type on (then drop).
_AUX_CHANNELS = {
    "M1": "misc",
    "M2": "misc",
    "HEO": "eog",
    "VEO": "eog",
    "EKG": "ecg",
    "EMG": "emg",
}

# Figshare download URL for the FineMI.zip archive containing raw .cnt files.
_FINEMI_URL = "https://ndownloader.figshare.com/files/42320127"


class Yi2025(BaseDataset):
    """Multi-joint upper-limb MI dataset from Yi et al. 2025.

    Dataset from [1]_.

    This dataset contains EEG recordings from 18 healthy subjects
    performing 8-class motor imagery of different upper-limb joints.
    Recorded with 64-channel Neuroscan SynAmps2 at 1000 Hz.

    The 8 MI classes correspond to multi-joint movements:

    1. Hand open/close
    2. Wrist flexion/extension
    3. Wrist abduction/adduction
    4. Elbow pronation/supination
    5. Elbow flexion/extension
    6. Shoulder pronation/supination
    7. Shoulder abduction/adduction
    8. Shoulder flexion/extension

    Each subject performed 1 session with 8 blocks of 40 trials
    (5 per class), for 320 total trials. Trial structure: 2 s fixation,
    2 s cue, 4 s MI, 10-12 s rest.

    Raw Neuroscan ``.cnt`` files are loaded from the ``FineMI.zip``
    archive on Figshare (~14.2 GB). Auxiliary channels (M1, M2, HEO,
    VEO, EKG, EMG) are dropped, leaving 62 EEG channels. Each block
    is loaded as a separate run.

    .. note::

       The first download requires the full FineMI.zip (14.2 GB).

    References
    ----------
    .. [1] Yi, W., Chen, J., Wang, D., et al. (2025). A multi-modal
           dataset of EEG and fNIRS for motor imagery of multi-types of
           joints from unilateral upper limb. Scientific Data, 12, 953.
           https://doi.org/10.1038/s41597-025-05286-0

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            hardware="Neuroscan SynAmps2",
            reference="left mastoid (M1)",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
            montage="standard_1005",
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            gender={"female": 10, "male": 8},
            age_min=22,
            age_max=27,
            health_status="healthy",
            species="human",
            handedness="right",
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=8,
            class_labels=list(_EVENTS.keys()),
            events=dict(_EVENTS),
            trial_duration=4.0,
            study_design=(
                "8-class multi-joint upper-limb MI. 8 blocks of 40 "
                "trials (5 per class), 320 total trials per subject."
            ),
            stimulus_type="video + text",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            synchronicity="cue-based",
            feedback_type="none",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-05286-0",
            investigators=[
                "Weibo Yi",
                "Jiaming Chen",
                "Dan Wang",
                "Xinkang Hu",
                "Meng Xu",
                "Fangda Li",
                "Shuhan Wu",
                "Jin Qian",
            ],
            institution="Beijing University of Technology",
            country="CN",
            data_url="https://figshare.com/articles/dataset/Data/24123303",
            publication_year=2025,
            license="CC-BY-NC-ND-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=8,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor Imagery"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=list(_EVENTS.keys()),
            cue_duration_s=2.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=320,
            n_trials_per_class={k: 40 for k in _EVENTS},
            n_blocks=8,
            trials_context="8 blocks x 40 trials (5 per class x 8 classes)",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["ShallowConvNet"],
            feature_extraction=["ERSP"],
            frequency_bands={
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "bandpass": [4.0, 40.0],
            },
            spatial_filters=["CAR"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold",
            cv_folds=5,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="CNT",
    )

    # Annotation codes in raw .cnt -> MOABB event names.
    _ANNOT_MAP = {str(v): k for k, v in _EVENTS.items()}

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 19)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Yi2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-05286-0",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject from raw .cnt files.

        Each subject has 8 block files loaded as separate runs within
        a single session.
        """
        subj_dir = Path(self.data_path(subject))

        # Find block .cnt files
        cnt_files = sorted(subj_dir.glob("block*.cnt"))
        if not cnt_files:
            raise FileNotFoundError(
                f"No block .cnt files found for subject {subject} in {subj_dir}"
            )

        runs = {}
        for run_idx, cnt_path in enumerate(cnt_files):
            raw = mne.io.read_raw_cnt(str(cnt_path), preload=True, verbose=False)

            # Fix channel name case for standard_1005 montage
            raw.rename_channels(
                {ch: _CH_RENAME[ch] for ch in raw.ch_names if ch in _CH_RENAME}
            )

            # Set non-EEG channel types
            aux_present = {ch: t for ch, t in _AUX_CHANNELS.items() if ch in raw.ch_names}
            if aux_present:
                raw.set_channel_types(aux_present)
                # Drop aux channels only when return_all_modalities is False
                if not self.return_all_modalities:
                    raw.drop_channels(list(aux_present.keys()))

            # Rename event annotations
            raw.annotations.rename(self._ANNOT_MAP)

            runs[str(run_idx)] = raw

        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")

        path = dl.get_dataset_path("Yi2025", path)
        basepath = Path(path) / "MNE-yi2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        subj_dir = basepath / f"subject{subject}" / "EEG"

        # Check if .cnt files already exist
        if subj_dir.is_dir() and list(subj_dir.glob("block*.cnt")):
            return str(subj_dir)

        # Download and extract FineMI.zip
        finemi_dir = basepath / "FineMI"
        if not finemi_dir.is_dir():
            log.info("Downloading Yi2025 FineMI.zip (14.2 GB) from Figshare...")
            dl_path = dl.data_dl(_FINEMI_URL, "Yi2025", path, force_update, verbose)
            dl_path = Path(dl_path)

            # Rename to .zip if needed
            zip_path = basepath / "FineMI.zip"
            if dl_path != zip_path:
                dl_path.rename(zip_path)

            log.info("Extracting FineMI.zip...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(basepath))

        # The .cnt files should now be at basepath/FineMI/subject{N}/EEG/
        extracted_dir = finemi_dir / f"subject{subject}" / "EEG"
        if not extracted_dir.is_dir():
            raise FileNotFoundError(
                f"Expected directory {extracted_dir} after extraction"
            )

        return str(extracted_dir)
