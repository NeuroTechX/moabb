"""
Openvibe Motor imagery dataset.
"""

from .base import BaseDataset

import pandas as pd

from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage


class OpenvibeMI(BaseDataset):
    """Openvibe Motor Imagery dataset"""

    def __init__(self):
        self.subject_list = range(1, 15)
        self.name = 'Openvibe Motor Imagery'

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = ('/home/kirsh/Documents/Data/OpenvibeMI/%02d-signal.csv'
                 % subject)
        fname_lbs = ('/home/kirsh/Documents/Data/OpenvibeMI/%02d-labels.csv'
                     % subject)

        data = pd.read_csv(fname, index_col=0, sep=';') * 1e-6
        labels = pd.read_csv(fname_lbs, index_col=0, sep=';')
        data['Stim'] = 0
        data.loc[labels.index, 'Stim'] = labels.Identifier

        ch_names = list(data.columns)
        ch_types = ['eeg'] * 11 + ['stim']
        montage = read_montage('standard_1005')

        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=512., montage=montage)
        raw = RawArray(data=data.values.T, info=info, verbose=False)
        return raw
