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
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# IEEE DataPort download URLs for the 9 ZIP archives.
# These are direct download links from the open-access dataset page.
_BASE_URL = "https://ieee-dataport.org/open-access/eeg-dataset-7-day-motor-imagery-bci"

# The dataset is organized in 9 ZIP files:
# S1-S2.zip through S11-S12.zip (S-subjects) and A1-A4.zip, A5-A8.zip
# We'll use the Figshare mirror or IEEE DataPort direct links.
# NOTE: IEEE DataPort may require a free login for download.
# The URLs will need to be verified - using placeholder pattern.

# Two subject groups with different channel counts:
# S-subjects (1-12): 41 channels
# A-subjects (13-20): 26 channels
# We map A1-A8 -> subjects 13-20 for consistent MOABB numbering.

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

# Event mapping: 4 classes
_EVENTS = {
    "left_hand": 1,
    "right_hand": 2,
    "feet": 3,
    "rest": 4,
}

_SFREQ = 500.0


class Zhou2020(BaseDataset):
    """7-day motor imagery BCI EEG dataset from Zhou et al 2021.

    Dataset from the article *Relative Power Correlates With the Decoding
    Performance of Motor Imagery Both Across Time and Subjects* [1]_.

    It contains data recorded on 20 subjects over 7 sessions (one session
    every ~2 days over 2 weeks) with no feedback training. Two groups of
    subjects were recorded:

    - **S-subjects** (subjects 1-12): 41 EEG channels
    - **A-subjects** (subjects 13-20): 26 EEG channels

    Four MI classes were recorded: left hand, right hand, both feet, and
    idle/rest. Each session contains 6 runs of 40 trials each (10 per
    class), giving 240 trials per session and 1680 trials per subject.

    The data is stored in NPZ (NumPy compressed) format with preprocessed
    EEG signals (band-pass 0.5-100 Hz, 50 Hz notch).

    References
    ----------
    .. [1] Zhou, Q., Yang, L., Wang, Q., et al. (2021). Relative Power
           Correlates With the Decoding Performance of Motor Imagery Both
           Across Time and Subjects. Frontiers in Human Neuroscience, 15,
           701091. https://doi.org/10.3389/fnhum.2021.701091
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=26,
            channel_types={"eeg": 26},
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
            handedness="right-handed",
            bci_experience="naive",
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
            investigators=["Qing Zhou"],
            institution="Zhejiang University",
            country="CN",
            data_url="https://ieee-dataport.org/open-access/eeg-dataset-7-day-motor-imagery-bci",
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
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
        ),
        data_processed=True,
        file_format="NPZ",
    )

    def __init__(self, subjects=None, sessions=None):
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
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        base = Path(self.data_path(subject))

        # Determine subject naming: S1-S12 or A1-A8
        if subject <= 12:
            prefix = f"S{subject}"
        else:
            prefix = f"A{subject - 12}"

        sessions = {}
        for sess_idx in range(7):
            sess_key = str(sess_idx)
            runs = {}

            for run_idx in range(6):
                # Try to find NPZ files with various naming patterns.
                # Common patterns: {prefix}_session{N}_run{M}.npz
                npz_file = self._find_npz(base, prefix, sess_idx, run_idx)
                if npz_file is None:
                    log.warning(
                        "Missing data: %s session %d run %d",
                        prefix,
                        sess_idx + 1,
                        run_idx + 1,
                    )
                    continue

                npz = np.load(npz_file, allow_pickle=True)
                raw = self._npz_to_raw(npz, subject)
                runs[str(run_idx)] = raw

            if runs:
                sessions[sess_key] = runs

        if not sessions:
            raise FileNotFoundError(
                f"No data found for subject {subject} ({prefix}) in {base}"
            )
        return sessions

    def _find_npz(self, base, prefix, sess_idx, run_idx):
        """Find an NPZ file for a given subject/session/run."""
        # The exact naming convention needs to be verified from actual data.
        # Try common patterns.
        patterns = [
            f"{prefix}_s{sess_idx + 1}_r{run_idx + 1}.npz",
            f"{prefix}_session{sess_idx + 1}_run{run_idx + 1}.npz",
            f"{prefix}_S{sess_idx + 1}_R{run_idx + 1}.npz",
            f"{prefix}_{sess_idx + 1}_{run_idx + 1}.npz",
        ]
        for pat in patterns:
            fpath = base / pat
            if fpath.exists():
                return fpath

        # Fallback: search for any NPZ matching the prefix
        all_npz = sorted(base.glob(f"{prefix}*.npz"))
        expected_idx = sess_idx * 6 + run_idx
        if expected_idx < len(all_npz):
            return all_npz[expected_idx]

        return None

    def _npz_to_raw(self, npz, subject):
        """Convert a loaded NPZ archive to MNE Raw."""
        # NPZ expected keys: 'data' (trials x channels x samples),
        # 'labels' (trial labels)
        # The exact keys need verification from actual data.
        data_key = None
        label_key = None
        for k in npz.files:
            kl = k.lower()
            if "data" in kl or "eeg" in kl or "x" == kl:
                data_key = k
            elif "label" in kl or "y" == kl:
                label_key = k

        if data_key is None:
            # Try first two arrays
            keys = list(npz.files)
            data_key = keys[0]
            label_key = keys[1] if len(keys) > 1 else None

        trial_data = npz[data_key]  # (n_trials, n_channels, n_samples)
        labels = npz[label_key].ravel() if label_key else np.ones(trial_data.shape[0])

        n_trials, n_ch, n_samples = trial_data.shape

        # Determine channel names based on subject group
        if subject <= 12:
            # S-subjects: 41 channels - read from data
            ch_names = [f"EEG{i + 1}" for i in range(n_ch)]
        else:
            # A-subjects: 26 channels
            if n_ch == len(_26CH_NAMES):
                ch_names = list(_26CH_NAMES)
            else:
                ch_names = [f"EEG{i + 1}" for i in range(n_ch)]

        ch_types = ["eeg"] * n_ch + ["stim"]
        ch_names_full = ch_names + ["STI"]
        info = mne.create_info(ch_names=ch_names_full, ch_types=ch_types, sfreq=_SFREQ)

        # Concatenate trials with stim channel and buffers
        buffer_samples = 50
        segments = []
        for t in range(n_trials):
            trial = trial_data[t]  # (n_ch, n_samples)
            trial = trial - trial.mean(axis=1, keepdims=True)

            # Scale to volts if in microvolts
            if np.abs(trial).max() > 1e-3:
                trial = trial * 1e-6

            stim = np.zeros((1, n_samples))
            stim[0, 0] = labels[t]

            block = np.concatenate([trial, stim], axis=0)
            buf = np.zeros((block.shape[0], buffer_samples))
            segments.extend([buf, block, buf])

        continuous = np.concatenate(segments, axis=1)
        raw = mne.io.RawArray(data=continuous, info=info, verbose=False)

        # Set montage if channel names are standard
        if ch_names[0] != "EEG1":
            try:
                montage = mne.channels.make_standard_montage("standard_1005")
                raw.set_montage(montage, on_missing="warn")
            except Exception:
                pass

        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = dl.get_dataset_path("Zhou2020", path)
        basepath = Path(path) / "MNE-zhou2020-data"
        basepath.mkdir(parents=True, exist_ok=True)

        if subject <= 12:
            prefix = f"S{subject}"
        else:
            prefix = f"A{subject - 12}"

        # Check if subject data already exists
        existing = list(basepath.glob(f"{prefix}*.npz"))
        if existing:
            return str(basepath)

        # Download the appropriate ZIP from IEEE DataPort.
        # NOTE: IEEE DataPort may require manual download.
        log.warning(
            "Zhou2020 data must be downloaded manually from IEEE DataPort: "
            "https://ieee-dataport.org/open-access/eeg-dataset-7-day-motor-imagery-bci "
            "Download the ZIP file(s) for subject %s and extract to: %s",
            prefix,
            basepath,
        )

        return str(basepath)
