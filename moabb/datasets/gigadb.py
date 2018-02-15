"""
GigaDb Motor imagery dataset.
"""

from .base import BaseDataset

from scipy.io import loadmat
import numpy as np

from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
import moabb.datasets.download as dl

import os

GIGA_URL = 'ftp://penguin.genomics.cn/pub/10.5524/100001_101000/100295/mat_data/'

def data_path(subject, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of GIGA dataset URL.

    Parameters
    ----------
    subject : int
        Number of subject to use
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_INRIA_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_INRIA_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """  # noqa: E501
    if subject < 1 or subject > 52:
        raise ValueError("Valid subjects between 1 and 52, subject {:d} requested".format(subject))
    url = '{:s}s{:02d}.mat'.format(GIGA_URL, subject)

    return dl.data_path(url, 'GIGADB', path, force_update, update_path, verbose)

class GigaDbMI(BaseDataset):
    """GigaDb Motor Imagery dataset"""

    def __init__(self):
        self.subject_list = list(range(1, 53))
        # some subject have issues
        for ii in [32, 46, 49]:
            self.subject_list.remove(ii)

        self.name = 'GigaDb Motor Imagery'
        self.tmin = 1
        self.tmax = 3
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(left_hand=1, right_hand=2)

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = data_path(subject)

        data = loadmat(fname, squeeze_me=True, struct_as_record=False,
                       verify_compressed_data_integrity=False)['eeg']

        eeg_ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                        'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                        'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                        'FPz', 'FP2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                        'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                        'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
        emg_ch_names = ['EMG1', 'EMG2', 'EMG3', 'EMG4']
        ch_names = eeg_ch_names + emg_ch_names + ['Stim']
        ch_types = ['eeg'] * 64 + ['emg'] * 4 + ['stim']
        montage = read_montage('standard_1005')

        eeg_data_l = np.vstack([data.imagery_left * 1e-6, data.imagery_event])
        eeg_data_r = np.vstack([data.imagery_right * 1e-6,
                                data.imagery_event * 2])

        eeg_data = np.hstack([eeg_data_l, eeg_data_r])  # stacking causes first trials of _r to be corrupted slightly by any sort of filtering -- can fix by making them two raw arrays?

        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=data.srate, montage=montage)
        raw = RawArray(data=eeg_data, info=info, verbose=False)
        return [raw]
