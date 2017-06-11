"""
Alex Motor imagery dataset.
"""

from .base import BaseDataset

from scipy.io import loadmat
import numpy as np

from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage


class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset"""

    def __init__(self, with_rest=False):
        self.subject_list = range(1, 9)
        self.name = 'Alex Motor Imagery'
        self.tmin = 0
        self.tmax = 3
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(right_hand=1, feets=2)
        if with_rest:
            self.event_id['rest'] = 3

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        subject_names = ['A', 'B', 'E', 'G', 'H', 'I', 'J', 'L']
        fname = ('/home/kirsh/Documents/Data/LE2S_Multiclass/User%s.mat'
                 % subject_names[subject - 1])

        data = loadmat(fname)['X'][1:]
        data = np.nan_to_num(data)
        data[0:16] *= 1e-6
        ch_names = ['Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
                    'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Stim']
        ch_types = ['eeg'] * 16 + ['stim']
        montage = read_montage('standard_1005')

        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=512., montage=montage)
        raw = RawArray(data=data, info=info, verbose=False)
        return [raw]
