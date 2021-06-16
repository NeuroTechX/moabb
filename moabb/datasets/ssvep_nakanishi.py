"""
SSVEP Nakanishi dataset.
"""

import logging

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset


log = logging.getLogger(__name__)

NAKAHISHI_URL = "https://github.com/mnakanishi/12JFPM_SSVEP/raw/master/data/"


class Nakanishi2015(BaseDataset):
    """SSVEP Nakanishi 2015 dataset

    This dataset contains 12-class joint frequency-phase modulated steady-state
    visual evoked potentials (SSVEPs) acquired from 10 subjects used to
    estimate an online performance of brain-computer interface (BCI) in the
    reference study [1]_.

    references
    ----------
    .. [1] Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,
           "A Comparison Study of Canonical Correlation Analysis Based Methods for
           Detecting Steady-State Visual Evoked Potentials," PLoS One, vol.10, no.10,
           e140703, 2015.
           `<http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703>`_
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events={
                "9.25": 1,
                "11.25": 2,
                "13.25": 3,
                "9.75": 4,
                "11.75": 5,
                "13.75": 6,
                "10.25": 7,
                "12.25": 8,
                "14.25": 9,
                "10.75": 10,
                "12.75": 11,
                "14.75": 12,
            },
            code="SSVEP Nakanishi",
            interval=[0.15, 4.3],
            paradigm="ssvep",
            doi="doi.org/10.1371/journal.pone.0140703",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject"""
        n_samples, n_channels, n_trials = 1114, 8, 15
        n_classes = len(self.event_id)

        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True)
        data = np.transpose(mat["eeg"], axes=(0, 3, 1, 2))
        data = np.reshape(data, newshape=(-1, n_channels, n_samples))
        data = data - data.mean(axis=2, keepdims=True)
        raw_events = np.zeros((data.shape[0], 1, n_samples))
        raw_events[:, 0, 0] = np.array(
            [n_trials * [i + 1] for i in range(n_classes)]
        ).flatten()
        data = np.concatenate([1e-6 * data, raw_events], axis=1)
        # add buffer in between trials
        log.warning(
            "Trial data de-meaned and concatenated with a buffer"
            " to create continuous data"
        )
        buff = (data.shape[0], n_channels + 1, 50)
        data = np.concatenate([np.zeros(buff), data, np.zeros(buff)], axis=2)
        ch_names = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "stim"]
        ch_types = ["eeg"] * 8 + ["stim"]
        sfreq = 256
        info = create_info(ch_names, sfreq, ch_types)
        raw = RawArray(data=np.concatenate(list(data), axis=1), info=info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        return {"session_0": {"run_0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        url = "{:s}s{:d}.mat".format(NAKAHISHI_URL, subject)
        return dl.data_dl(url, "NAKANISHI", path, force_update, verbose)
