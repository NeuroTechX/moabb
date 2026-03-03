"""59-Subject 40-Class SSVEP Dataset.

Dong and Tian (2023), Brain Science Advances.
DOI: 10.26599/BSA.2023.9050020
"""

import numpy as np
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset

from ._utils import build_raw_from_epochs


ZENODO_URL = "https://zenodo.org/records/18847318/files/"


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

    _ch_names = ["POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2"]

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
        n_channels, n_blocks = 8, 4
        n_classes = len(self.event_id)

        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True)

        # .mat key "eegdata" with shape [8, 1250, 40, 4]
        # (channels, timepoints, targets, blocks)
        eeg = mat["eegdata"]
        data = np.transpose(eeg, axes=(2, 3, 0, 1))
        data = np.reshape(data, (-1, n_channels, eeg.shape[1]))

        event_ids = np.repeat(np.arange(1, n_classes + 1), n_blocks)
        raw = build_raw_from_epochs(data, self._ch_names, 250, event_ids, "standard_1005")
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = "DONG2023"
        url = f"{ZENODO_URL}S{subject}.mat?download=1"
        return dl.data_dl(url, sign, path, force_update, verbose)
