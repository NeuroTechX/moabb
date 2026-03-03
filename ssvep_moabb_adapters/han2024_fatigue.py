"""SSVEP Fatigue Dataset with Dynamic Stopping Strategy.

Han et al. (2024), IEEE TNSRE.
DOI: 10.1109/TNSRE.2024.3380635
"""

import logging
import zipfile
from pathlib import Path

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/10507229/files/"

# The 4 data conditions and their directory names after zip extraction
_CONDITIONS = [
    ("low_frequency_train_data", "low", "train"),
    ("low_frequency_fatigue_data", "low", "fatigue"),
    ("high_frequency_train_data", "high", "train"),
    ("high_frequency_fatigue_data", "high", "fatigue"),
]


class Han2024Fatigue(BaseDataset):
    """SSVEP fatigue dataset with two frequency paradigms.

    Dataset from [1]_.

    This dataset contains 64-channel EEG recordings from 24 healthy subjects
    (12 males, 12 females, aged 18-26) performing two SSVEP-BCI tasks:

    - **Low-frequency paradigm:** 16 targets (8.0-15.5 Hz, 0.5 Hz step)
    - **High-frequency paradigm:** 16 targets (25.5-33.0 Hz, 0.5 Hz step)

    Both paradigms used JFPM encoding with phases cycling through
    0, 0.5*pi, pi, 1.5*pi in a 4x4 matrix layout.

    The experiment consisted of two phases: training (6 blocks per frequency
    condition) and fatigue (24 blocks per condition). Each block contained
    16 trials (2 s stimulation per trial).

    EEG was recorded at 1000 Hz with a Synamps2 system (Neuroscan) and 64
    channels. Each epoch spans 3000 samples (3 s at 1000 Hz).

    Data is stored as [16, 64, 3000, N_blocks] matrices (targets, channels,
    timepoints, blocks) in per-subject zip files on Zenodo. Each subject has
    4 separate files: low_frequency_train, low_frequency_fatigue,
    high_frequency_train, high_frequency_fatigue.

    In MOABB, this is mapped as:
    - Session '0': Training blocks (6 blocks per condition, 12 total)
    - Session '1': Fatigue blocks (24 blocks per condition, 48 total)

    References
    ----------
    .. [1] Y. Han, Y. Ke, R. Wang, T. Wang, and D. Ming, "Enhancing
       SSVEP-BCI Performance Under Fatigue State Using Dynamic Stopping
       Strategy," IEEE Trans. Neural Syst. Rehab. Eng., vol. 32,
       pp. 1407-1415, 2024. DOI: 10.1109/TNSRE.2024.3380635
    """

    # fmt: off
    # Low-frequency events (16): 8.0-15.5 Hz, 0.5 Hz step
    # High-frequency events (16): 25.5-33.0 Hz, 0.5 Hz step
    _events = {
        "8": 1, "8.5": 2, "9": 3, "9.5": 4,
        "10": 5, "10.5": 6, "11": 7, "11.5": 8,
        "12": 9, "12.5": 10, "13": 11, "13.5": 12,
        "14": 13, "14.5": 14, "15": 15, "15.5": 16,
        "25.5": 17, "26": 18, "26.5": 19, "27": 20,
        "27.5": 21, "28": 22, "28.5": 23, "29": 24,
        "29.5": 25, "30": 26, "30.5": 27, "31": 28,
        "31.5": 29, "32": 30, "32.5": 31, "33": 32,
    }

    _ch_names = [
        "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
        "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
        "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz",
        "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6",
        "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2",
        "CB2", "stim",
    ]
    # fmt: on

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 25)),
            sessions_per_subject=2,
            events=self._events,
            code="Han2024Fatigue",
            interval=[0.0, 2.0],
            paradigm="ssvep",
            doi="10.1109/TNSRE.2024.3380635",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across training and fatigue sessions.

        Loads 4 separate .mat files per subject (low/high × train/fatigue).
        Each file has shape [16, 64, 3000, N_blocks] = (targets, channels,
        timepoints, blocks). Training files have 6 blocks, fatigue files
        have 24 blocks.
        """
        n_channels, n_targets = 64, 16
        sfreq = 1000

        file_paths = self.data_path(subject)

        # Group conditions by session: train→session '0', fatigue→session '1'
        session_data = {"0": [], "1": []}
        for dir_name, freq_band, phase in _CONDITIONS:
            mat_path = file_paths[dir_name]
            mat = loadmat(mat_path, squeeze_me=True)
            data = mat["data"]  # (16, 64, 3000, N_blocks)
            n_samples = data.shape[2]
            n_blocks = data.shape[3]

            # Event offset: low-freq classes are 1-16, high-freq are 17-32
            event_offset = 16 if freq_band == "high" else 0
            sess_key = "0" if phase == "train" else "1"

            for block_idx in range(n_blocks):
                block_data = data[:, :, :, block_idx]  # (16, 64, 3000)
                block_data = block_data - block_data.mean(axis=2, keepdims=True)

                stim = np.zeros((n_targets, 1, n_samples))
                stim[:, 0, 0] = np.arange(1, n_targets + 1) + event_offset

                trial_data = np.concatenate([1e-6 * block_data, stim], axis=1)
                buff = np.zeros((n_targets, n_channels + 1, 50))
                trial_data = np.concatenate([buff, trial_data, buff], axis=2)
                session_data[sess_key].append(trial_data)

        sessions = {}
        for sess_name, trials_list in session_data.items():
            if not trials_list:
                continue
            all_trials = np.concatenate(trials_list, axis=0)
            ch_types = ["eeg"] * n_channels + ["stim"]
            info = create_info(self._ch_names, sfreq, ch_types)
            log.info(
                "Trial data de-meaned and concatenated with a buffer"
                " to create continuous data"
            )
            raw = RawArray(
                data=np.concatenate(list(all_trials), axis=1),
                info=info,
                verbose=False,
            )
            montage = make_standard_montage("standard_1005")
            raw.set_montage(montage, on_missing="ignore")
            sessions[sess_name] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = "HAN2024FATIGUE"
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if all 4 condition files exist
        all_paths = {}
        all_found = True
        for dir_name, _, _ in _CONDITIONS:
            mat_file = data_dir / dir_name / f"S{subject}.mat"
            if mat_file.exists() and not force_update:
                all_paths[dir_name] = str(mat_file)
            else:
                all_found = False

        if all_found:
            return all_paths

        # Download the zip file
        url = f"{ZENODO_URL}S{subject}.zip"
        zip_path = dl.data_dl(url, sign, path, force_update, verbose)

        # Extract
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

        # Find the .mat files in subdirectories
        all_paths = {}
        for dir_name, _, _ in _CONDITIONS:
            mat_file = data_dir / dir_name / f"S{subject}.mat"
            if mat_file.exists():
                all_paths[dir_name] = str(mat_file)
            else:
                # Search recursively
                found = list(data_dir.rglob(f"{dir_name}/S{subject}.mat"))
                if found:
                    all_paths[dir_name] = str(found[0])
                else:
                    raise FileNotFoundError(
                        f"Could not find {dir_name}/S{subject}.mat "
                        f"after extracting S{subject}.zip"
                    )

        return all_paths
