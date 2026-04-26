"""EEG data for voluntary finger tapping movement.

Wairagkar, Hayashi, and Nasuto (2018), PLOS ONE.
DOI: 10.1371/journal.pone.0193722
Data DOI: 10.17864/1947.117
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
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import safe_extract_zip


log = logging.getLogger(__name__)

_ZIP_URL = "https://researchdata.reading.ac.uk/117/2/EEG_finger_tapping_data.zip"

# Channel names in order (1-indexed in README).
# Channels 1-19: EEG (standard 10-20).
# Channel 20: Rt (right tap binary), Channel 21: Lt (left tap binary).
_CH_NAMES_EEG = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",  # T3 in original, mapped to 10-10
    "C3",
    "Cz",
    "C4",
    "T8",  # T4 in original, mapped to 10-10
    "P7",  # T5 in original, mapped to 10-10
    "P3",
    "Pz",
    "P4",
    "P8",  # T6 in original, mapped to 10-10
    "O1",
    "O2",
]

# Condition index -> MOABB event name.
# From README: 1=right tap, 2=rest, 3=left tap.
_CONDITION_MAP = {
    0: "right_hand",  # 0-indexed condition 1
    1: "rest",  # 0-indexed condition 2
    2: "left_hand",  # 0-indexed condition 3
}

_SFREQ = 1024.0
_TRIAL_SAMPLES = 6144  # 6 s at 1024 Hz
_ONSET_SAMPLE = 3072  # movement onset at 3 s into trial


class Wairagkar2018(BaseDataset):
    """Motor execution dataset from Wairagkar et al 2018.

    Dataset from the article *Exploration of neural correlates of movement
    intention based on characterisation of temporal dependencies in
    electroencephalography* [1]_.

    **Important**: This is a motor **execution** dataset, not motor imagery.
    Participants physically tapped their index fingers.

    It contains data recorded on 14 subjects with 19 EEG electrodes
    (standard 10-20 system) plus 2 binary tap-detection channels. Data
    is pre-epoched (6 s trials centered on movement onset) and
    preprocessed (ICA artifact removal, bandpass 0.5-60 Hz).

    Three conditions were recorded:

    - **right_hand**: right index finger tap
    - **left_hand**: left index finger tap
    - **rest**: resting state (no movement)

    Each subject has 40 trials per condition (120 total), except
    subject 2 who has 35 trials per condition (105 total).

    References
    ----------
    .. [1] Wairagkar, M., Hayashi, Y., & Nasuto, S. J. (2018).
           Exploration of neural correlates of movement intention based
           on characterisation of temporal dependencies in
           electroencephalography. PLOS ONE, 13(3), e0193722.
           https://doi.org/10.1371/journal.pone.0193722
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1024.0,
            n_channels=19,
            channel_types={"eeg": 19},
            montage="standard_1020",
            hardware="Deymed TruScan 32",
            sensor_type="Ag/AgCl ring",
            reference="FCz",
            ground="AFz",
            filters={"highpass": 0.5, "lowpass": 60, "notch_hz": 50},
            sensors=list(_CH_NAMES_EEG),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            gender={"female": 8, "male": 6},
            age_mean=26.0,
            age_std=4.0,
            handedness="mixed (12 right, 2 left)",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"right_hand": 1, "rest": 2, "left_hand": 3},
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "rest", "left_hand"],
            trial_duration=6.0,
            study_design=(
                "Asynchronous voluntary finger tapping: right tap, "
                "left tap, and resting state"
            ),
            feedback_type="none",
            stimulus_type="text cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="asynchronous",
            mode="offline",
            instructions=(
                "Participants were asked to tap their index finger at a "
                "self-chosen time within a 10-second window after the cue"
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0193722",
            investigators=[
                "Maitreyee Wairagkar",
                "Yoshikatsu Hayashi",
                "Slawomir J. Nasuto",
            ],
            institution="University of Reading",
            institution_department="Brain Embodiment Lab, Biomedical Engineering",
            country="GB",
            data_url="https://researchdata.reading.ac.uk/117/",
            publication_year=2018,
            senior_author="Slawomir J. Nasuto",
            license="CC-BY-4.0",
            repository="University of Reading Research Data Archive",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "DC offset removal",
                "0.5 Hz high-pass filter",
                "50 Hz notch filter",
                "60 Hz low-pass filter",
                "ICA artifact removal (EEGLAB infomax)",
                "trial segmentation (-3 to +3 s around movement onset)",
            ],
            highpass_hz=0.5,
            lowpass_hz=60.0,
            notch_hz=50.0,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", imagery_tasks=["right_hand", "left_hand", "rest"]
        ),
        data_structure=DataStructureMetadata(
            n_trials=1665,
            trials_context=(
                "14 subjects x 120 trials (40 per condition), except "
                "subject 2 with 105 trials (35 per condition)"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["autocorrelation_relaxation_time", "ERD"],
            frequency_bands={
                "broadband": [0.5, 30.0],
                "mu": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "low": [0.5, 8.0],
            },
            spatial_filters=["bipolar_montage"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10x10-fold", cv_folds=10, evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=True,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events={"right_hand": 1, "rest": 2, "left_hand": 3},
            code="Wairagkar2018",
            interval=[0, 3],  # 0-3 s post movement onset
            paradigm="imagery",
            doi="10.1371/journal.pone.0193722",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        fpath = self.data_path(subject)
        mat = loadmat(fpath, squeeze_me=False)

        # Variable name is "ParticipantN" where N = subject number.
        var_name = f"Participant{subject}"
        cell_array = mat[var_name]  # shape: (21, 3, n_trials) object array

        n_eeg = len(_CH_NAMES_EEG)  # 19
        n_conditions = cell_array.shape[1]
        n_trials = cell_array.shape[2]

        # Build channel info: 19 EEG + 1 stim channel.
        ch_names = list(_CH_NAMES_EEG) + ["STI"]
        ch_types = ["eeg"] * n_eeg + ["stim"]
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=_SFREQ)

        # Collect all trials, add stim channel, concatenate with buffers.
        all_segments = []
        buffer_samples = 50  # ~49 ms buffer between trials

        for cond_idx in range(n_conditions):
            event_id = cond_idx + 1  # 1=right_hand, 2=rest, 3=left_hand
            for trial_idx in range(n_trials):
                # Extract EEG data for this trial (19 channels only).
                trial_data = np.zeros((n_eeg, _TRIAL_SAMPLES))
                for ch_idx in range(n_eeg):
                    cell = cell_array[ch_idx, cond_idx, trial_idx]
                    trial_data[ch_idx] = cell.ravel()[:_TRIAL_SAMPLES]

                # De-mean each trial.
                trial_data -= trial_data.mean(axis=1, keepdims=True)

                # Scale to volts (data is in microvolts).
                trial_data *= 1e-6

                # Create stim channel: event marker at movement onset.
                stim = np.zeros((1, _TRIAL_SAMPLES))
                stim[0, _ONSET_SAMPLE] = event_id

                # Combine EEG + stim.
                trial_block = np.concatenate([trial_data, stim], axis=0)

                # Add zero buffer before and after.
                buf = np.zeros((trial_block.shape[0], buffer_samples))
                all_segments.append(buf)
                all_segments.append(trial_block)
                all_segments.append(buf)

        continuous = np.concatenate(all_segments, axis=1)
        raw = mne.io.RawArray(data=continuous, info=info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)

        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Wairagkar2018", path)
        basepath = Path(path) / "MNE-wairagkar2018-data"
        basepath.mkdir(parents=True, exist_ok=True)

        mat_file = basepath / f"Participant{subject}.mat"
        if mat_file.exists():
            return str(mat_file)

        # Check if mat files exist in a subfolder (e.g. from prior download).
        for mat in basepath.rglob(f"Participant{subject}.mat"):
            if mat != mat_file:
                mat.rename(mat_file)
                break

        if mat_file.exists():
            return str(mat_file)

        # Download and extract the single ZIP containing all subjects.
        # Find the zip: it may be at various locations after dl.data_dl.
        zip_path = None
        for candidate in basepath.rglob("EEG_finger_tapping_data.zip"):
            zip_path = candidate
            break

        if zip_path is None:
            dl.data_dl(
                _ZIP_URL,
                "Wairagkar2018",
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
            # Find the downloaded zip.
            for candidate in basepath.rglob("EEG_finger_tapping_data.zip"):
                zip_path = candidate
                break

        # Extract all .mat files from the ZIP.
        if zip_path is not None and zip_path.exists() and not mat_file.exists():
            with zipfile.ZipFile(zip_path) as zf:
                safe_extract_zip(zf, basepath)

        # Move mat files from subfolders to basepath.
        for mat in basepath.rglob("Participant*.mat"):
            dest = basepath / mat.name
            if mat != dest:
                mat.rename(dest)

        if not mat_file.exists():
            raise FileNotFoundError(
                f"Could not find {mat_file} after download and extraction."
            )

        return str(mat_file)
