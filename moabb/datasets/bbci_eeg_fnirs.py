"""
BBCI EEG fNIRS Motor imagery dataset.
"""

from .base import BaseDataset

import numpy as np
from scipy.io import loadmat
from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage


class BBCIEEGfNIRS(BaseDataset):
    """BBCI EEG fNIRS Motor Imagery dataset"""

    def __init__(self, motor=True,
                 base_folder='/home/kirsh/Documents/Data/BBCI_EEG_fNIRS'):
        self.subject_list = range(1, 30)
        self.name = 'BBCI EEG fNIRS'
        self.base_folder = base_folder
        self.motor = motor

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = '%s/subject %02d/with occular artifact/cnt.mat' % (self.base_folder, subject)
        fname_mrk = '%s/subject %02d/with occular artifact/mrk.mat' % (self.base_folder, subject)

        raws = []
        data = loadmat(fname, squeeze_me=True, struct_as_record=False)['cnt']
        mrk = loadmat(fname_mrk, squeeze_me=True, struct_as_record=False)['mrk']
        montage = read_montage('standard_1005')

        if self.motor:
            runs = [0, 2, 4]
        else:
            runs = [1, 3, 5]

        for ii in runs:
            eeg = data[ii].x.T * 1e-6
            trig = np.zeros((1, eeg.shape[1]))
            idx = (mrk[ii].time - 1) // 5
            trig[0, idx] = mrk[ii].event.desc // 16
            eeg = np.vstack([eeg, trig])
            ch_names = list(data[ii].clab) + ['Stim']
            ch_types = ['eeg'] * 30 + ['eog'] * 2 + ['stim']

            info = create_info(ch_names=ch_names, ch_types=ch_types,
                               sfreq=200., montage=montage)
            raw = RawArray(data=eeg, info=info, verbose=False)
            raws.append(raw)
        return raws
