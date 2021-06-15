"""
GigaDb Motor imagery dataset.
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
GIGA_URL = "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100295/mat_data/"


class Cho2017(BaseDataset):
    """Motor Imagery dataset from Cho et al 2017.

    Dataset from the paper [1]_.

    **Dataset Description**

    We conducted a BCI experiment for motor imagery movement (MI movement)
    of the left and right hands with 52 subjects (19 females, mean age ± SD
    age = 24.8 ± 3.86 years); Each subject took part in the same experiment,
    and subject ID was denoted and indexed as s1, s2, …, s52.
    Subjects s20 and s33 were both-handed, and the other 50 subjects
    were right-handed.

    EEG data were collected using 64 Ag/AgCl active electrodes.
    A 64-channel montage based on the international 10-10 system was used to
    record the EEG signals with 512 Hz sampling rates.
    The EEG device used in this experiment was the Biosemi ActiveTwo system.
    The BCI2000 system 3.0.2 was used to collect EEG data and present
    instructions (left hand or right hand MI). Furthermore, we recorded
    EMG as well as EEG simultaneously with the same system and sampling rate
    to check actual hand movements. Two EMG electrodes were attached to the
    flexor digitorum profundus and extensor digitorum on each arm.

    Subjects were asked to imagine the hand movement depending on the
    instruction given. Five or six runs were performed during the MI
    experiment. After each run, we calculated the classification
    accuracy over one run and gave the subject feedback to increase motivation.
    Between each run, a maximum 4-minute break was given depending on
    the subject's demands.

    References
    ----------

    .. [1] Cho, H., Ahn, M., Ahn, S., Kwon, M. and Jun, S.C., 2017.
           EEG datasets for motor imagery brain computer interface.
           GigaScience. https://doi.org/10.1093/gigascience/gix034
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 53)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            code="Cho2017",
            interval=[0, 3],  # full trial is 0-3s, but edge effects
            paradigm="imagery",
            doi="10.5524/100295",
        )

        for ii in [32, 46, 49]:
            self.subject_list.remove(ii)

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = self.data_path(subject)

        data = loadmat(
            fname,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )["eeg"]

        # fmt: off
        eeg_ch_names = [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ]
        # fmt: on
        emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
        ch_names = eeg_ch_names + emg_ch_names + ["Stim"]
        ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]
        montage = make_standard_montage("standard_1005")
        imagery_left = data.imagery_left - data.imagery_left.mean(axis=1, keepdims=True)
        imagery_right = data.imagery_right - data.imagery_right.mean(
            axis=1, keepdims=True
        )

        eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
        eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])

        # trials are already non continuous. edge artifact can appears but
        # are likely to be present during rest / inter-trial activity
        eeg_data = np.hstack(
            [eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r]
        )
        log.warning(
            "Trials demeaned and stacked with zero buffer to create "
            "continuous data -- edge effects present"
        )

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=data.srate)
        raw = RawArray(data=eeg_data, info=info, verbose=False)
        raw.set_montage(montage)

        return {"session_0": {"run_0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = "{:s}s{:02d}.mat".format(GIGA_URL, subject)
        return dl.data_dl(url, "GIGADB", path, force_update, verbose)
