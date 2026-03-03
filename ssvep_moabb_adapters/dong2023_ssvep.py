"""59-Subject 40-Class SSVEP Dataset.

Dong and Tian (2023), Brain Science Advances.
DOI: 10.26599/BSA.2023.9050020
"""

import logging
from pathlib import Path

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)

# Google Drive folder: https://drive.google.com/drive/folders/1TXuxU863nZoniZRgNWZy0PRuL8lhBuP4
DONG_DRIVE_FOLDER = "1TXuxU863nZoniZRgNWZy0PRuL8lhBuP4"

# Google Drive file IDs for individual subject .mat files.
# Due to Google Drive's 50-file folder listing limit, only S1-S48 have known IDs.
# fmt: off
_GDRIVE_FILE_IDS = {
    1: "1aR6f5Wq75I6NBdHDoroUDupHvW-7jo7k", 2: "1HBgq8IezHzsUwtcopYmjfzPqzEI4uWWW",
    3: "1cL1KlqTawL__jb3xHmFLtbZEDBF4NAGp", 4: "13G3Co4Twl_iTPMxbsMWdo5ce_bomxvdp",
    5: "1Z4zx1eFKBrVf0ogVJPfqPWDrWdnncaEh", 6: "17fRsWaokYKdW5XLmbm3alFocmXyvZvdH",
    7: "1GbOV1s2OCzXxctT6ilCb7kEYnRewDtIG", 8: "1KcnuTqbMPf21F3HAcqQCJTJ_HRCpIhXg",
    9: "1jzN-oTKYf3o-A8gwpCooPCHppiiQnTjy", 10: "1LEkQ2euH5_dBmvMRGBZizB63d9aioMmy",
    11: "1EnsSmG7jTmwCtqo0oveOPj9i-z2MB-bz", 12: "1njpF_ZO8aTFa8nYyYZcxb5xAIBM6VCfJ",
    13: "1TMUH1ARAyl6tG06pa7Bybv2gbzsvL1Pe", 14: "1m3lCS8Gucg1SJSN8G7FRJLEfZekB_opy",
    15: "1C-VCJFQdheEbAie6o8Vr8J7S_Wuob4Q5", 16: "1TrHRtCleleJ2nh57ZpoPxD2YHnY6ZEVJ",
    17: "1fcPy8Ae-Yio3LtB-Q3ditkC3hI0UxHXM", 18: "115OE8pqTmflQLcXF8m6P5fN1J58vLYIb",
    19: "1ldwCRxi-pcglzOlVkPRfJ16fUX1k4cFq", 20: "1EhgikqW5vl8xsPjpqnCKFOLsM8oQ93R1",
    21: "1lwQqDw97cblFyVo53tys-OEO2q05C-fE", 22: "1a5Ggk149cENZWOfcRBYLlPCiwBOfyPUe",
    23: "1Ss_DgoKi_htZWFNko99-9cpMK5ZUL0vB", 24: "1BK1DyxMSxRcbe0XoP5SEwIOpLB_-WRoX",
    25: "1q__Lvf7NJE9xQCUXXPUYy8kFN4UcuVZM", 26: "1hMrXesaBMxPkv6SDDWIl2kcJbkX4_nwj",
    27: "1UBflugMNYyKPpB-7I3I2w2EScvGCglMW", 28: "17SYONaCKCTVGqtbSvNyb32Cl0wS66nab",
    29: "1R3Uf0mjafzaorf0TC29bYYQcnUPBRsdH", 30: "1CzvcZYhME1_7tkvhtlX9hPsAZ3Q7GXmc",
    31: "1gmDk3DfemSQwHw8Ymk0MXAcT6EDX6XOd", 32: "1WIAoXqFEZJwIznrhdGfsawkjrCeOS_D_",
    33: "1iv0RdQHpluEjVI8cyiMNwPu_X5gAJRor", 34: "1h0ON9UqA4AATXhUjLo9qkPE35kYO0D4E",
    35: "1R-P63ATFQUaQvcIGAypiLo95xaAceF1Y", 36: "1THnsLWVZsIhlKZBhiNMNgJSxIM-edVwY",
    37: "1lNzRRYPPgtyh4texSOrB1TkrY9wNC464", 38: "1ar_f28WdsqQRT5b-1W866jiKXmBCqVfR",
    39: "1yO2H_PUQoHsNEC1tv3UuI_s0gmAxWf-8", 40: "1Q9xr5ogP6hTIsyUb0_8lNSYDXxgIqHL1",
    41: "17DBxOWqOeYb5JRmr2k9dw_jlt_UlbwDF", 42: "1XO7d1gX4BdpJUCbULBofQlPLYy5Fj2vH",
    43: "14w2H3bh8UW7Tj9t3DFSfqnpzXd7z-wlm", 44: "1CIlKRDkosbJuhEFAS60zZhpZ0Ae3pVM2",
    45: "1GjZ5o8-p4IockDd9XTW64HKN1JjDyyhX", 46: "1PtDnEUeAjq-pj3YA_qAW4o1BByjdAwZr",
    47: "1irhi5K-vT68SM1FO897RiKZRypZLxoIp", 48: "1QF81qWc-_4_RzD002vhiQ-SEP5-8n7qQ",
}
# fmt: on


class Dong2023(BaseDataset):
    """59-subject 40-class SSVEP dataset.

    Dataset from [1]_.

    This dataset contains 8-channel EEG recordings from 59 healthy adolescent
    volunteers (38 males, 21 females, aged 10-16, mean age 12.3) from the
    Suzhou Junior Competition of BCI Olympics 2022. Subjects performed a
    40-target SSVEP-BCI task using joint frequency and phase modulation (JFPM).

    Stimulation frequencies ranged from 8.0 to 15.8 Hz (0.2 Hz step) with a
    4x10 matrix layout. Each subject completed 4 blocks of 40 trials with
    1 s cue, 4 s stimulation, and 1 s feedback per trial.

    EEG was recorded at 1000 Hz with a NeuSenW system (Neuracle) using 8
    semi-dry (pre-gelled) electrodes around the occipital lobe, then
    downsampled to 250 Hz. Data is stored as 4D matrices of shape
    [8, 1250, 40, 4] (channels, time points, targets, blocks).

    Each trial epoch spans 5 s (0.5 s pre-stimulus + 4 s stimulation + 0.5 s
    post-stimulus) at 250 Hz = 1250 samples.

    Warnings
    --------
    Data is hosted on Google Drive. Requires the ``gdown`` package for
    automated download (``pip install gdown``). Due to Google Drive's
    folder listing limit, subjects S49-S59 may require manual download
    from the shared folder.

    References
    ----------
    .. [1] Y. Dong and S. Tian, "A large database towards user-friendly
       SSVEP-based BCI," Brain Science Advances, vol. 9, no. 4,
       pp. 297-309, 2023. DOI: 10.26599/BSA.2023.9050020
    """

    # Same 40 frequencies as Wang2016 (8.0-15.8 Hz, 0.2 Hz step)
    # JFPM column-major order: columns of the 5x8 frequency matrix
    # fmt: off
    _events = {
        "8": 1, "9": 2, "10": 3, "11": 4, "12": 5, "13": 6, "14": 7, "15": 8, "8.2": 9,
        "9.2": 10, "10.2": 11, "11.2": 12, "12.2": 13, "13.2": 14, "14.2": 15, "15.2": 16,
        "8.4": 17, "9.4": 18, "10.4": 19, "11.4": 20, "12.4": 21, "13.4": 22, "14.4": 23,
        "15.4": 24, "8.6": 25, "9.6": 26, "10.6": 27, "11.6": 28, "12.6": 29, "13.6": 30,
        "14.6": 31, "15.6": 32, "8.8": 33, "9.8": 34, "10.8": 35, "11.8": 36, "12.8": 37,
        "13.8": 38, "14.8": 39, "15.8": 40,
    }
    # fmt: on

    _ch_names = ["POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2", "stim"]

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 60)),
            sessions_per_subject=1,
            events=self._events,
            code="Dong2023",
            interval=[0.5, 4.5],
            paradigm="ssvep",
            doi="10.26599/BSA.2023.9050020",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        n_channels, n_samples, n_blocks = 8, 1250, 4
        n_classes = len(self.event_id)

        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True)

        # .mat key "eegdata" with shape [8, 1250, 40, 4]
        # (channels, timepoints, targets, blocks)
        data = np.transpose(mat["eegdata"], axes=(2, 3, 0, 1))
        data = np.reshape(data, (-1, n_channels, n_samples))
        data = data - data.mean(axis=2, keepdims=True)

        # Build stim channel
        raw_events = np.zeros((data.shape[0], 1, n_samples))
        raw_events[:, 0, 0] = np.array(
            [n_blocks * [i + 1] for i in range(n_classes)]
        ).flatten()
        data = np.concatenate([1e-6 * data, raw_events], axis=1)

        # Add zero-padding buffers
        log.warning(
            "Trial data de-meaned and concatenated with a buffer"
            " to create continuous data"
        )
        buff = (data.shape[0], n_channels + 1, 50)
        data = np.concatenate([np.zeros(buff), data, np.zeros(buff)], axis=2)

        ch_types = ["eeg"] * n_channels + ["stim"]
        sfreq = 250
        info = create_info(self._ch_names, sfreq, ch_types)
        raw = RawArray(data=np.concatenate(list(data), axis=1), info=info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = "DONG2023"
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Check if already downloaded
        mat_file = data_dir / f"S{subject}.mat"
        if mat_file.exists() and not force_update:
            return str(mat_file)

        data_dir.mkdir(parents=True, exist_ok=True)

        if subject in _GDRIVE_FILE_IDS:
            # Download individual file via gdown
            try:
                import gdown
            except ImportError:
                raise ImportError(
                    "The Dong2023 dataset requires the 'gdown' package. "
                    "Install it with: pip install gdown"
                )
            file_id = _GDRIVE_FILE_IDS[subject]
            gdown.download(id=file_id, output=str(mat_file), quiet=False)
        else:
            raise FileNotFoundError(
                f"No known Google Drive file ID for subject {subject}. "
                f"Please download S{subject}.mat manually from "
                f"https://drive.google.com/drive/folders/{DONG_DRIVE_FOLDER} "
                f"and place it in {data_dir}"
            )

        if not mat_file.exists():
            raise FileNotFoundError(
                f"Download failed for S{subject}.mat. Please download "
                f"manually from https://drive.google.com/drive/folders/"
                f"{DONG_DRIVE_FOLDER}"
            )

        return str(mat_file)
