"""
BBCI EEG fNIRS Motor imagery dataset.
"""

from .base import BaseDataset

import numpy as np
from scipy.io import loadmat
from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
from . import download as dl
import os.path as op
import os
import zipfile as z
from mne.datasets.utils import _get_path, _do_path_update
from mne.utils import _fetch_file, _url_to_local_path

BBCIFNIRS_URL = 'http://doc.ml.tu-berlin.de/hBCI/'


def eeg_data_path(base_path, subject):
    datapath = op.join(base_path, 'EEG', 'subject {:02d}'.format(
        subject), 'with occular artifact')
    if not op.isfile(op.join(datapath, 'cnt.mat')):
        if not op.isdir(op.join(base_path, 'EEG')):
            os.makedirs(op.join(base_path, 'EEG'))
        intervals = [[1, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 29]]
        for low, high in intervals:
            if subject >= low and subject <= high:
                if not op.isfile(op.join(base_path, 'EEG.zip')):
                    _fetch_file('http://doc.ml.tu-berlin.de/hBCI/EEG/EEG_{:02d}-{:02d}.zip'.format(low,
                                                                                               high),
                            op.join(base_path, 'EEG.zip'), print_destination=False)
                with z.ZipFile(op.join(base_path, 'EEG.zip'), 'r') as f:
                    f.extractall(op.join(base_path, 'EEG'))
                os.remove(op.join(base_path, 'EEG.zip'))
                break
    assert op.isfile(op.join(datapath, 'cnt.mat')
                     ), op.join(datapath, 'cnt.mat')
    return [op.join(datapath, fn) for fn in ['cnt.mat', 'mrk.mat']]


def fnirs_data_path(path, subject):
    datapath = op.join(path, 'NIRS', 'subject {:02d}'.format(subject))
    if not op.isfile(op.join(datapath, 'mrk.mat')):
        print('No fNIRS files for subject, suggesting dataset not yet downloaded. All subjects must now be downloaded')
        # fNIRS
        if not op.isfile(op.join(path, 'fNIRS.zip')):
            _fetch_file('http://doc.ml.tu-berlin.de/hBCI/NIRS/NIRS_01-29.zip',
                        op.join(path, 'fNIRS.zip'), print_destination=False)
        if not op.isdir(op.join(path, 'NIRS')):
            os.makedirs(op.join(path, 'NIRS'))
        with z.ZipFile(op.join(path, 'fNIRS.zip'), 'r') as f:
            f.extractall(op.join(path, 'NIRS'))
        os.remove(op.join(path, 'fNIRS.zip'))
    return [op.join(datapath, fn) for fn in ['cnt.mat', 'mrk.mat']]


class BBCIEEGfNIRS(BaseDataset):
    """BBCI EEG fNIRS Motor Imagery dataset"""

    def __init__(self, fnirs=False, motor_imagery=True,
                 mental_arithmetic=False):
        if not any([motor_imagery, mental_arithmetic]):
            raise(ValueError("at least one of motor_imagery or"
                             " mental_arithmetic must be true"))
        events = dict()
        paradigms = []
        n_sessions = 0
        if motor_imagery:
            events.update(dict(left_hand=1, right_hand=2))
            paradigms.append('imagery')
            n_sessions += 3

        if mental_arithmetic:
            events.update(dict(substraction=3, rest=4))
            paradigms.append('arithmetic')
            n_sessions += 3

        self.motor_imagery = motor_imagery
        self.mental_arithmetic = mental_arithmetic

        super().__init__(subjects=list(range(1, 30)),
                         sessions_per_subject=n_sessions,
                         events=events,
                         code='BBCI EEG fNIRS',
                         interval=[3.5, 10],
                         paradigm=('/').join(paradigms),
                         doi='10.1109/TNSRE.2016.2628057')

        self.fnirs = fnirs  # TODO: actually incorporate fNIRS somehow

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname, fname_mrk = self.data_path(subject)
        data = loadmat(fname, squeeze_me=True, struct_as_record=False)['cnt']
        mrk = loadmat(fname_mrk, squeeze_me=True,
                      struct_as_record=False)['mrk']

        sessions = {}
        # motor imagery
        if self.motor_imagery:
            for ii in [0, 2, 4]:
                session = self._convert_one_session(data, mrk, ii,
                                                    trig_offset=0)
                sessions['session_%d' % ii] = session

        # arithmetic/rest
        if self.mental_arithmetic:
            for ii in [1, 3, 5]:
                session = self._convert_one_session(data, mrk, ii,
                                                    trig_offset=2)
                sessions['session_%d' % ii] = session

        return sessions

    def _convert_one_session(self, data, mrk, session, trig_offset=0):
        eeg = data[session].x.T * 1e-6
        trig = np.zeros((1, eeg.shape[1]))
        idx = (mrk[session].time - 1) // 5
        trig[0, idx] = mrk[session].event.desc // 16 + trig_offset
        eeg = np.vstack([eeg, trig])
        ch_names = list(data[session].clab) + ['Stim']
        ch_types = ['eeg'] * 30 + ['eog'] * 2 + ['stim']

        montage = read_montage('standard_1005')
        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=200., montage=montage)
        raw = RawArray(data=eeg, info=info, verbose=False)
        return {'run_0': raw}

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        key = 'MNE_DATASETS_BBCIFNIRS_PATH'
        path = _get_path(path, key, 'BBCI EEG-fNIRS')
        # FIXME: this always update the path
        _do_path_update(path, True, key, 'BBCI EEG-fNIRS')
        if not op.isdir(op.join(path, 'MNE-eegfnirs-data')):
            os.makedirs(op.join(path, 'MNE-eegfnirs-data'))
        if self.fnirs:
            return fnirs_data_path(op.join(path, 'MNE-eegfnirs-data'), subject)
        else:
            return eeg_data_path(op.join(path, 'MNE-eegfnirs-data'), subject)
