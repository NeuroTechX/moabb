"""eldBETA: A Large Eldercare-oriented Benchmark Database of SSVEP-BCI.

Liu et al. (2022), Scientific Data.
DOI: 10.1038/s41597-022-01372-9
"""

import logging
import tarfile
from pathlib import Path

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)

FIGSHARE_DL_URL = "https://ndownloader.figshare.com/files/"

# Figshare file IDs for per-subject tar.gz files (S1.tar.gz through S100.tar.gz)
# fmt: off
_FIGSHARE_FILE_IDS = {
    1: 34516952, 2: 34516955, 3: 34516958, 4: 34516961, 5: 34516964,
    6: 34516967, 7: 34516970, 8: 34516973, 9: 34516976, 10: 34516979,
    11: 34516982, 12: 34516985, 13: 34516988, 14: 34516994, 15: 34517015,
    16: 34517018, 17: 34517024, 18: 34517027, 19: 34517030, 20: 34517033,
    21: 34517036, 22: 34517039, 23: 34519832, 24: 34517045, 25: 34517048,
    26: 34517051, 27: 34517054, 28: 34519643, 29: 34517060, 30: 34517063,
    31: 34517066, 32: 34517069, 33: 34517072, 34: 34517075, 35: 34517078,
    36: 34517081, 37: 34517084, 38: 34517087, 39: 34517090, 40: 34517093,
    41: 34517096, 42: 34517099, 43: 34517102, 44: 34517105, 45: 34517108,
    46: 34517111, 47: 34517114, 48: 34517117, 49: 34517120, 50: 34517123,
    51: 34517126, 52: 34517129, 53: 34517132, 54: 34517135, 55: 34517138,
    56: 34517141, 57: 34517144, 58: 34517147, 59: 34517150, 60: 34517153,
    61: 34517156, 62: 34517159, 63: 34517162, 64: 34517165, 65: 34517168,
    66: 34517171, 67: 34517174, 68: 34517177, 69: 34517180, 70: 34517183,
    71: 34517186, 72: 34517192, 73: 34517195, 74: 34517198, 75: 34517201,
    76: 34517204, 77: 34517207, 78: 34517210, 79: 34517213, 80: 34517216,
    81: 34517219, 82: 34517222, 83: 34517225, 84: 34517228, 85: 34517231,
    86: 34517234, 87: 34517237, 88: 34517240, 89: 34517243, 90: 34517246,
    91: 34517249, 92: 34517252, 93: 34517255, 94: 34517258, 95: 34517261,
    96: 34517264, 97: 34517267, 98: 34517270, 99: 34517273, 100: 34517276,
}
# fmt: on


class Liu2022EldBETA(BaseDataset):
    """eldBETA SSVEP benchmark dataset for elderly population.

    Dataset from [1]_.

    The eldBETA database contains 64-channel EEG recordings from 100 elderly
    participants (33 males, 67 females, aged 52-81, mean age 63.17) performing
    a 9-target SSVEP-BCI task. Stimuli used joint frequency and phase
    modulation (JFPM) with 9 targets in a 3x3 matrix. Frequencies ranged
    from 8.0 to 12.0 Hz (0.5 Hz step).

    Each subject completed 7 blocks of 9 trials. Each trial consisted of a
    4 s target cue followed by 5 s of SSVEP stimulation and 1 s rest (10 s
    total per trial). EEG was recorded at 1000 Hz with a Synamps2 system
    (Neuroscan) using 64 channels, then downsampled to 250 Hz.

    Data is stored as 4D matrices [64, 1500, 9, 7] corresponding to
    [channels, time points, target index, block index]. Each epoch is 6 s
    (1500 samples at 250 Hz).

    Warnings
    --------
    Like Wang2016 and BETA, this dataset uses the same 64-channel Tsinghua
    Neuroscan cap layout including 'CB1' and 'CB2' channels.

    The Figshare archive for each subject also includes BIDS-formatted GDF
    files (mislabeled as .edf). This implementation loads the .mat files
    for reliability.

    References
    ----------
    .. [1] B. Liu, Y. Wang, X. Gao, and X. Chen, "eldBETA: A Large
       Eldercare-oriented Benchmark Database of SSVEP-BCI for the Aging
       Population," Scientific Data, vol. 9, p. 252, 2022.
       DOI: 10.1038/s41597-022-01372-9
    """

    # fmt: off
    # Events follow JFPM column-major order matching target indices 1-9:
    # 3x3 grid read column-major: 8.0, 9.5, 11.0, 8.5, 10.0, 11.5, 9.0, 10.5, 12.0
    _events = {
        "8": 1, "9.5": 2, "11": 3, "8.5": 4, "10": 5,
        "11.5": 6, "9": 7, "10.5": 8, "12": 9,
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
            subjects=list(range(1, 101)),
            sessions_per_subject=7,
            events=self._events,
            code="Liu2022-EldBETA",
            interval=[0, 6.0],
            paradigm="ssvep",
            doi="10.1038/s41597-022-01372-9",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across all 7 blocks."""
        n_channels = 64
        sfreq = 250

        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True)

        # Struct: data.EEG.Epoch shape [64, 1500, 9, 7] (ch, time, targets, blocks)
        raw_data = mat["data"]
        eeg = raw_data["EEG"].item()
        epoch = eeg["Epoch"].item()
        n_classes = epoch.shape[2]  # 9
        n_blocks = epoch.shape[3]  # 7
        n_samples = epoch.shape[1]  # 1500

        sessions = {}
        for block_idx in range(n_blocks):
            block_data = epoch[:, :, :, block_idx]  # (64, 1500, 9)
            # Rearrange to (n_classes, n_channels, n_times)
            block_data = np.transpose(block_data, (2, 0, 1))  # (9, 64, 1500)
            block_data = block_data - block_data.mean(axis=2, keepdims=True)

            stim = np.zeros((n_classes, 1, n_samples))
            stim[:, 0, 0] = np.arange(1, n_classes + 1)

            block_data = np.concatenate([1e-6 * block_data, stim], axis=1)

            log.info(
                "Trial data de-meaned and concatenated with a buffer"
                " to create continuous data"
            )
            buff = np.zeros((n_classes, n_channels + 1, 50))
            block_data = np.concatenate([buff, block_data, buff], axis=2)

            ch_types = ["eeg"] * n_channels + ["stim"]
            info = create_info(self._ch_names, sfreq=sfreq, ch_types=ch_types)
            raw = RawArray(
                data=np.concatenate(list(block_data), axis=1),
                info=info,
                verbose=False,
            )
            montage = make_standard_montage("standard_1005")
            raw.set_montage(montage, on_missing="ignore")

            sessions[str(block_idx)] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        sign = "LIU2022ELDBETA"
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if the .mat file already exists
        mat_file = data_dir / "eldBETA" / f"S{subject}.mat"
        if mat_file.exists() and not force_update:
            return str(mat_file)

        # Download and extract the subject archive
        file_id = _FIGSHARE_FILE_IDS[subject]
        url = f"{FIGSHARE_DL_URL}{file_id}"
        tar_path = dl.data_dl(url, sign, path, force_update, verbose)

        extract_dir = data_dir / "eldBETA"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)

        # Search for the .mat file
        if mat_file.exists():
            return str(mat_file)

        for p in extract_dir.rglob(f"S{subject}.mat"):
            return str(p)

        raise FileNotFoundError(f"Could not find S{subject}.mat after extracting archive")
