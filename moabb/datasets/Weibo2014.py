'''
Simple and compound motor imagery
https://doi.org/10.1371/journal.pone.0114853
'''

from .base import BaseDataset
import zipfile as z
from scipy.io import loadmat
from mne.datasets.utils import _get_path, _do_path_update
from mne.utils import _fetch_file
import mne
import numpy as np
import os
import shutil

import logging
log = logging.getLogger()

FILES = []
FILES.append('https://dataverse.harvard.edu/api/access/datafile/2499178')
FILES.append('https://dataverse.harvard.edu/api/access/datafile/2499182')
FILES.append('https://dataverse.harvard.edu/api/access/datafile/2499179')


def eeg_data_path(base_path, subject):
    file1_subj = ['cl', 'cyy', 'kyf', 'lnn']
    file2_subj = ['ls', 'ry', 'wcf']
    file3_subj = ['wx', 'yyx', 'zd']

    def get_subjects(sub_inds, sub_names, ind):
        dataname = 'data{}'.format(ind)
        if not os.path.isfile(os.path.join(base_path, dataname+'.zip')):
            _fetch_file(FILES[ind], os.path.join(
                base_path, dataname + '.zip'), print_destination=False)
        with z.ZipFile(os.path.join(base_path, dataname + '.zip'), 'r') as f:
            os.makedirs(os.path.join(base_path, dataname), exist_ok=True)
            f.extractall(os.path.join(base_path, dataname))
            for fname in os.listdir(os.path.join(base_path, dataname)):
                for ind, prefix in zip(sub_inds, sub_names):
                    if fname.startswith(prefix):
                        os.rename(os.path.join(base_path, dataname, fname),
                                  os.path.join(base_path,
                                               'subject_{}.mat'.format(ind)))
        os.remove(os.path.join(base_path, dataname + '.zip'))
        shutil.rmtree(os.path.join(base_path, dataname))

    if not os.path.isfile(os.path.join(base_path,
                                       'subject_{}.mat'.format(subject))):
        if subject in range(1, 5):
            get_subjects(list(range(1, 5)), file1_subj, 0)
        elif subject in range(5, 8):
            get_subjects(list(range(5, 8)), file2_subj, 1)
        elif subject in range(8, 11):
            get_subjects(list(range(8, 11)), file3_subj, 2)
    return os.path.join(base_path, 'subject_{}.mat'.format(subject))


class Weibo2014(BaseDataset):
    """Weibo 2014 Motor Imagery dataset [1]

    References 
    ----------- 
    Yi Weibo, 2014, "EEG data of simple and compound limb
    motor imagery", https://doi.org/10.7910/DVN/27306, Harvard Dataverse, V1

    """



    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2,
                        hands=3, feet=4, left_hand_right_foot=5,
                        right_hand_left_foot=6, rest=7),
            code='Weibo 2014',
            # Full trial is 0-8 but with trialwise bandpass this reduces
            # boundary effects
            interval=[0, 8],
            paradigm='imagery',
            doi='10.7910/DVN/27306')

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = self.data_path(subject)
        # TODO: add 1s 0 buffer between trials and make continuous
        data = loadmat(fname, squeeze_me=True, struct_as_record=False,
                       verify_compressed_data_integrity=False)
        montage = mne.channels.read_montage('standard_1020')
        info = mne.create_info(ch_names=['EEG{}'.format(i) for i in range(1, 65)]+['STIM014'],
                               ch_types=['eeg']*64+['stim'],
                               sfreq=200, montage=None)
        # until we get the channel names montage is None
        event_ids = data['label'].ravel()
        raw_data = np.transpose(data['data'], axes=[2, 0, 1])
        # de-mean each trial
        raw_data = raw_data - np.mean(raw_data, axis=2, keepdims=True)
        raw_events = np.zeros((raw_data.shape[0], 1, raw_data.shape[2]))
        raw_events[:, 0, 0] = event_ids
        data = np.concatenate([raw_data, raw_events], axis=1)
        # add buffer in between trials
        log.warning(
            'Trial data de-meaned and concatenated with a buffer to create cont data')
        zeroshape = (data.shape[0], data.shape[1], 50)
        data = np.concatenate([np.zeros(zeroshape), data,
                               np.zeros(zeroshape)], axis=2)
        raw = mne.io.RawArray(data=np.concatenate(list(data), axis=1),
                              info=info, verbose=False)
        return {'session_0': {'run_0': raw}}

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))
        key = 'MNE_DATASETS_WEIBO2014_PATH'
        path = _get_path(path, key, "Weibo 2014")
        _do_path_update(path, True, key, "Weibo 2014")
        basepath = os.path.join(path, "MNE-weibo-2014")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)
        return eeg_data_path(basepath, subject)
