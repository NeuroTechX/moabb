"""7-day motor imagery BCI EEG dataset.

Zhou et al. (2021), Frontiers in Human Neuroscience.
DOI: 10.3389/fnhum.2021.701091
Data DOI: 10.21227/f1c7-7x89
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

# Zenodo re-hosted data (originally from IEEE DataPort DOI: 10.21227/f1c7-7x89).
_ZENODO_RECORD = "18988317"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

# Two subject groups with different channel counts:
# S-subjects (1-12): 41 EEG + 4 EOG + 1 trigger = 46 channels
# A-subjects (13-20): 26 EEG + 2 EOG + 1 trigger = 29 channels
# We map A1-A8 -> subjects 13-20 for consistent MOABB numbering.

# 26 EEG channel names used by A-subjects (also the analysis subset in paper).
_26CH_NAMES = [
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
]

# GDF/BioSig event codes in MarkOnSignal → MOABB event names.
_GDF_EVENT_MAP = {
    769: "left_hand",  # 0x0301 — class 1 (left arrow)
    770: "right_hand",  # 0x0302 — class 2 (right arrow)
    771: "feet",  # 0x0303 — class 3 (down arrow)
    780: "rest",  # 0x030C — class 12 (up arrow = idle)
}

# MOABB event_id mapping (integer codes for stim channel).
_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
    "feet": 3,
    "rest": 4,
}

# Channel counts per subject group (EEG only, excluding EOG and trigger).
_S_N_EEG = 41  # S-subjects (1-12)
_A_N_EEG = 26  # A-subjects (13-20)

_SFREQ = 500.0


class Zhou2020(BaseDataset):
    """7-day motor imagery BCI EEG dataset from Zhou et al 2021.

    Dataset from the article *Relative Power Correlates With the Decoding
    Performance of Motor Imagery Both Across Time and Subjects* [1]_.

    It contains data recorded on 20 subjects over 7 sessions (one session
    every ~2 days over 2 weeks) with no feedback training. Two groups of
    subjects were recorded with a 64-channel Neuroscan SynAmps2 system
    at 500 Hz:

    - **S-subjects** (subjects 1-12): 41 EEG + 4 EOG channels
    - **A-subjects** (subjects 13-20): 26 EEG + 2 EOG channels

    Four MI classes were recorded: left hand, right hand, both feet, and
    idle/rest. Each session contains ~6 runs of 40 trials each (10 per
    class), giving ~240 trials per session and ~1680 trials per subject.

    The data is stored as Neuroscan NSsignal NPZ files with continuous
    recordings (band-pass 0.5-100 Hz, 50 Hz notch). Events are encoded
    using GDF/BioSig codes: 769 (left), 770 (right), 771 (feet), 780 (rest).

    References
    ----------
    .. [1] Zhou, Q., Lin, J., Yao, L., Wang, Y., Han, Y., Xu, K. (2021).
           Relative Power Correlates With the Decoding Performance of Motor
           Imagery Both Across Time and Subjects. Frontiers in Human
           Neuroscience, 15, 701091.
           https://doi.org/10.3389/fnhum.2021.701091
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=41,
            channel_types={"eeg": 41},
            montage="standard_1005",
            hardware="Neuroscan SynAmps2",
            sensor_type="Ag/AgCl",
            reference="vertex (Cz)",
            ground="AFz",
            filters={"bandpass": [0.5, 100], "notch_hz": 50},
            sensors=list(_26CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=20,
            health_status="healthy",
            gender={"female": 9, "male": 11},
            age_mean=23.2,
            age_min=21,
            age_max=27,
            age_std=1.47,
            handedness="right-handed",
            bci_experience="mixed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "feet", "rest"],
            trial_duration=5.0,
            study_design=(
                "7-day longitudinal MI-BCI study without feedback training. "
                "4 classes: left hand, right hand, both feet, idle"
            ),
            feedback_type="none",
            stimulus_type="arrow cues",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2021.701091",
            investigators=[
                "Qing Zhou",
                "Jiafan Lin",
                "Lin Yao",
                "Yueming Wang",
                "Yan Han",
                "Kedi Xu",
            ],
            institution="Zhejiang University",
            country="CN",
            repository="Zenodo",
            data_url=f"https://zenodo.org/records/{_ZENODO_RECORD}",
            publication_year=2021,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=7,
        runs_per_session=6,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "feet", "rest"],
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=33600,
            trials_context=("20 subjects x 7 sessions x 6 runs x 40 trials = 33600"),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["SVM"],
            feature_extraction=["CSP"],
            frequency_bands={
                "classification": [8.0, 30.0],
            },
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["within_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["research"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=True,
        file_format="NPZ",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 21)),
            sessions_per_subject=7,
            events=dict(_EVENTS),
            code="Zhou2020",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.3389/fnhum.2021.701091",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        data_dir = Path(self.data_path(subject))
        subj_dir = data_dir / f"S{subject:02d}"

        # Determine EEG channel count and names for this subject group.
        if subject <= 12:
            n_eeg = _S_N_EEG
            ch_names = [f"EEG{i + 1}" for i in range(n_eeg)]
        else:
            n_eeg = _A_N_EEG
            ch_names = list(_26CH_NAMES)

        sessions = {}
        session_dirs = sorted(d for d in subj_dir.iterdir() if d.is_dir())

        for sess_idx, sess_dir in enumerate(session_dirs):
            sess_key = str(sess_idx)
            runs = {}

            npz_files = sorted(sess_dir.glob("*.npz"))
            for run_idx, npz_file in enumerate(npz_files):
                try:
                    raw = self._npz_to_raw(npz_file, n_eeg, ch_names)
                    runs[str(run_idx)] = raw
                except Exception as e:
                    log.warning("Failed to load %s: %s", npz_file.name, e)

            if runs:
                sessions[sess_key] = runs

        if not sessions:
            raise FileNotFoundError(f"No data found for subject {subject} in {subj_dir}")
        return sessions

    def _npz_to_raw(self, npz_path, n_eeg, ch_names):
        """Convert an NSsignal NPZ file to MNE Raw.

        Parameters
        ----------
        npz_path : Path
            Path to the .npz file.
        n_eeg : int
            Number of EEG channels (41 for S-subjects, 26 for A-subjects).
        ch_names : list of str
            EEG channel names.
        """
        npz = np.load(npz_path, allow_pickle=True)
        signal = npz["signal"]  # (n_samples, n_channels_total)
        mos = npz["MarkOnSignal"]  # (n_events, 2): [sample, code]
        sfreq = float(npz["SampleRate"][0])

        # Extract only the EEG channels (first n_eeg columns).
        eeg_data = signal[:, :n_eeg].T  # (n_eeg, n_samples)

        # Scale from microvolts to volts for MNE.
        if np.abs(eeg_data).max() > 1e-3:
            eeg_data = eeg_data * 1e-6

        # Build stim channel from MarkOnSignal events.
        stim = np.zeros((1, eeg_data.shape[1]))
        for sample_idx, gdf_code in mos:
            event_name = _GDF_EVENT_MAP.get(int(gdf_code))
            if event_name is not None and 0 <= sample_idx < stim.shape[1]:
                stim[0, int(sample_idx)] = _EVENTS[event_name]

        # Create MNE Raw.
        all_data = np.concatenate([eeg_data, stim], axis=0)
        ch_types = ["eeg"] * n_eeg + ["stim"]
        ch_names_full = list(ch_names) + ["STI"]
        info = mne.create_info(
            ch_names=ch_names_full,
            ch_types=ch_types,
            sfreq=sfreq,
        )
        raw = mne.io.RawArray(data=all_data, info=info, verbose=False)

        # Set montage for A-subjects (standard channel names).
        if ch_names[0] != "EEG1":
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage, on_missing="warn")

        return raw

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"S{subject:02d}"

        # Check if subject data already exists (session dirs with NPZs).
        if subj_dir.is_dir():
            has_data = any(subj_dir.rglob("*.npz"))
            if has_data and not force_update:
                return str(data_dir)

        # Download per-subject ZIP from Zenodo and extract.
        url = f"{_ZENODO_BASE}/S{subject:02d}.zip"
        download_and_extract_subject_zip(url, sign, subj_dir, path, force_update, verbose)

        if not any(subj_dir.rglob("*.npz")):
            raise FileNotFoundError(
                f"No .npz files found for subject {subject} in {subj_dir}"
            )
        return str(data_dir)
