#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

import mne
import numpy as np
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


ALPHAWAVES_URL = "https://zenodo.org/record/2348892/files/"


class Rodrigues2017(BaseDataset):
    """Alphawaves dataset

    Dataset containing EEG recordings of subjects in a simple
    resting-state eyes open/closed experimental protocol. Data were recorded
    during a pilot experiment taking place in the GIPSA-lab, Grenoble,
    France, in 2017 [1]_.

    **Dataset Description**

    This experiment was conducted to
    provide a simple yet reliable set of EEG signals carrying very distinct
    signatures on each experimental condition. It can be useful for researchers
    and students looking for an EEG dataset to perform tests with signal
    processing and machine learning algorithms.

    I. Participants

    A total of 20 volunteers participated in the experiment (7 females), with
    mean (sd) age 25.8 (5.27) and median 25.5. 18 subjects were between 19 and
    28 years old. Two participants with age 33 and 44 were outside this range.

    II. Procedures

    EEG signals were acquired using a standard research grade amplifier
    (g.USBamp, g.tec, Schiedlberg, Austria) and the EC20 cap equipped with 16
    wet electrodes (EasyCap, Herrsching am Ammersee, Germany), placed according
    to the 10-20 international system.
    We acquired the data with no digital filter and a sampling frequency of 512Hz
    was used.

    Each participant underwent one session consisting of
    ten blocks of ten seconds of EEG data recording.
    Five blocks were recorded while a subject was keeping his eyes
    closed (condition 1) and the others while his eyes were open (condition 2).
    The two conditions were alternated. Before the onset of each block, the
    subject was asked to close or open his eyes according to the experimental
    condition.

    We supply an online and open-source example working with Python [2]_.

    References
    ----------

    .. [1] G. Cattan, P. L. Coelho Rodrigues, and M. Congedo,
           ‘EEG Alpha Waves Dataset’, 2018.
           Available: https://hal.archives-ouvertes.fr/hal-02086581

    .. [2] Rodrigues PLC. Alpha-Waves-Dataset [Internet].
           Grenoble: GIPSA-lab; 2018. Available from:
           https://github.com/plcrodrigues/Alpha-Waves-Dataset

    Notes
    -----

    .. versionadded:: 1.1.0

    """

    def __init__(self):
        subject_list = list(range(1, 6 + 1)) + list(range(8, 20 + 1))
        super().__init__(
            subjects=subject_list,
            sessions_per_subject=1,
            events=dict(closed=1, open=2),
            code="Rodrigues2017",
            interval=[0, 10],
            paradigm="rstate",
            doi="https://doi.org/10.5281/zenodo.2348892",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        dirpath = self.data_path(subject)[0]
        filepath = os.listdir(dirpath)[0]

        data = loadmat(os.path.join(dirpath, filepath))

        S = data["SIGNAL"][:, 1:17]
        stim_close = data["SIGNAL"][:, 17]
        stim_open = data["SIGNAL"][:, 18]
        stim = 1 * stim_close + 2 * stim_open
        chnames = [
            "Fp1",
            "Fp2",
            "Fc5",
            "Fz",
            "Fc6",
            "T7",
            "Cz",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
            "stim",
        ]
        chtypes = ["eeg"] * 16 + ["stim"]
        X = np.concatenate([S, stim[:, None]], axis=1).T

        info = mne.create_info(
            ch_names=chnames, sfreq=512, ch_types=chtypes, verbose=False
        )
        raw = mne.io.RawArray(data=X, info=info, verbose=False)

        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = "{:s}subject_{:02d}.mat".format(ALPHAWAVES_URL, subject)
        file_path = dl.data_path(url, "ALPHAWAVES")

        return [file_path]
