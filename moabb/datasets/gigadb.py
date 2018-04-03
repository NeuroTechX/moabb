"""
GigaDb Motor imagery dataset.
"""

from .base import BaseDataset

from scipy.io import loadmat
import numpy as np

from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
from . import download as dl

GIGA_URL = 'ftp://penguin.genomics.cn/pub/10.5524/100001_101000/100295/mat_data/'


class GigaDbMI(BaseDataset):
    """GigaDb Motor Imagery dataset"""

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 53)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            code='GigaDb Motor Imagery',
            interval=[1, 3],
            paradigm='imagery',
            doi='10.5524/100295')
        for ii in [32, 46, 49]:
            self.subject_list.remove(ii)

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = self.data_path(subject)

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

        # trials are already non continuous. edge artifact can appears but
        # are likely to be present during rest / inter-trial activity
        eeg_data = np.hstack([eeg_data_l, eeg_data_r])

        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=data.srate, montage=montage)
        raw = RawArray(data=eeg_data, info=info, verbose=False)

        return {'session_0': {'run_0': raw}}

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        url = '{:s}s{:02d}.mat'.format(GIGA_URL, subject)
        return dl.data_path(url, 'GIGADB', path, force_update, update_path,
                            verbose)
