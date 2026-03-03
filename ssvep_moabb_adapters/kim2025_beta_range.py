"""40-Class Beta-Range SSVEP Speller Dataset.

Kim et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-06032-2
"""

import logging

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)

FIGSHARE_DL_URL = "https://ndownloader.figshare.com/files/"

# Figshare file IDs for raw_eeg_ssvep_subj_NN.mat
# fmt: off
_SSVEP_FILE_IDS = {
    1: 53705183, 2: 53705180, 3: 53705123, 4: 53705168, 5: 53705171,
    6: 53705108, 7: 53705132, 8: 53705153, 9: 53705126, 10: 53705156,
    11: 53705129, 12: 53705099, 13: 53705177, 14: 53705150, 15: 53707388,
    16: 53705135, 17: 53705162, 18: 53705087, 19: 53705114, 20: 53705174,
    21: 53705075, 22: 53705117, 23: 53707391, 24: 53705078, 25: 53705165,
    26: 53705090, 27: 53705144, 28: 53705120, 29: 53707394, 30: 53705141,
    31: 53705207, 32: 53705159, 33: 53705138, 34: 53705111, 35: 53705189,
    36: 53705195, 37: 53705186, 38: 53705198, 39: 53705192, 40: 53705201,
}
# fmt: on

_EVENTS = {
    "14": 1,
    "15": 2,
    "16": 3,
    "17": 4,
    "18": 5,
    "19": 6,
    "20": 7,
    "21": 8,
    "14.2": 9,
    "15.2": 10,
    "16.2": 11,
    "17.2": 12,
    "18.2": 13,
    "19.2": 14,
    "20.2": 15,
    "21.2": 16,
    "14.4": 17,
    "15.4": 18,
    "16.4": 19,
    "17.4": 20,
    "18.4": 21,
    "19.4": 22,
    "20.4": 23,
    "21.4": 24,
    "14.6": 25,
    "15.6": 26,
    "16.6": 27,
    "17.6": 28,
    "18.6": 29,
    "19.6": 30,
    "20.6": 31,
    "21.6": 32,
    "14.8": 33,
    "15.8": 34,
    "16.8": 35,
    "17.8": 36,
    "18.8": 37,
    "19.8": 38,
    "20.8": 39,
    "21.8": 40,
}
# fmt: on


class Kim2025BetaRange(BaseDataset):
    """40-class beta-range SSVEP speller dataset.

    Dataset from [1]_.

    This dataset contains 33-channel EEG (31 scalp + 2 mastoid references)
    recorded from 40 healthy subjects (25 males, 15 females, aged 20-35)
    performing a 40-target SSVEP-BCI speller task using beta-range frequencies
    (14.0-21.8 Hz, 0.2 Hz step). The JFPM approach was used with phase
    differences of 0.5*pi between adjacent stimuli.

    Each subject completed 6 blocks of 40 trials. Trial structure was 1.5 s
    rest, 0.5 s cue, and 5.0 s SSVEP stimulation. EEG was recorded at 1024 Hz
    with a BioSemi ActiveTwo system. Stored epochs span [-2000, 5000] ms
    relative to stimulus onset (7168 samples at 1024 Hz). The event marker is
    placed at stimulus onset (sample 2048), and interval=[0.0, 5.0] extracts
    the 5 s stimulation window.

    The stimuli were presented in a 5x8 matrix on a 120 Hz monitor.

    References
    ----------
    .. [1] H. Kim, K. Won, M. Ahn, and S. C. Jun, "A 40-class SSVEP speller
       dataset: beta range stimulation for low-fatigue BCI applications,"
       Scientific Data, vol. 12, p. 1751, 2025.
       DOI: 10.1038/s41597-025-06032-2
    """

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 41)),
            sessions_per_subject=6,
            events=_EVENTS,
            code="Kim2025BetaRange",
            interval=[0.0, 5.0],
            paradigm="ssvep",
            doi="10.1038/s41597-025-06032-2",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across all 6 blocks."""
        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True, simplify_cells=True)
        eeg_struct = mat["eeg"]

        data = eeg_struct["data"]  # shape: (33, 7168, 40, 6)
        ch_names_raw = list(eeg_struct["chan_locs"])
        srate = int(eeg_struct["srate"])  # 1024
        n_channels = data.shape[0]  # 33
        n_classes = data.shape[2]  # 40
        n_blocks = data.shape[3]  # 6

        # Epoch window is [-2000, 5000] ms; stimulus onset is at +2000 ms
        onset_sample = int(round(2.0 * srate))  # sample 2048

        # Normalize channel names to match MNE standard_1005
        ch_names = _normalize_ch_names(ch_names_raw)

        sessions = {}
        for block_idx in range(n_blocks):
            block_data = data[:, :, :, block_idx]  # (33, 7168, 40)
            # Rearrange to (n_classes, n_channels, n_times)
            block_data = np.transpose(block_data, (2, 0, 1))  # (40, 33, 7168)
            block_data = block_data - block_data.mean(axis=2, keepdims=True)

            n_times = block_data.shape[2]
            stim = np.zeros((n_classes, 1, n_times))
            # Place event marker at stimulus onset (2 s into the epoch)
            stim[:, 0, onset_sample] = np.arange(1, n_classes + 1)

            block_data = np.concatenate([1e-6 * block_data, stim], axis=1)

            log.info(
                "Trial data de-meaned and concatenated with a buffer"
                " to create continuous data"
            )
            buff = np.zeros((n_classes, n_channels + 1, 50))
            block_data = np.concatenate([buff, block_data, buff], axis=2)

            all_ch_names = ch_names + ["stim"]
            ch_types = ["eeg"] * n_channels + ["stim"]
            info = create_info(all_ch_names, sfreq=srate, ch_types=ch_types)
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
        file_id = _SSVEP_FILE_IDS[subject]
        url = f"{FIGSHARE_DL_URL}{file_id}"
        return dl.data_dl(url, "KIM2025BETARANGE", path, force_update, verbose)


def _normalize_ch_names(ch_names):
    """Normalize channel names from .mat file to match MNE conventions.

    The .mat files use uppercase midline names (e.g. 'CZ', 'POZ', 'PZ',
    'CPZ', 'OZ') which must be converted to mixed case ('Cz', 'POz',
    'Pz', 'CPz', 'Oz') for MNE standard_1005 montage compatibility.
    """
    # Map uppercase midline channels to MNE mixed-case convention
    mapping = {
        "CZ": "Cz",
        "PZ": "Pz",
        "OZ": "Oz",
        "CPZ": "CPz",
        "POZ": "POz",
    }
    return [mapping.get(ch, ch) for ch in ch_names]
