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
from mne.utils import _fetch_file, _url_to_local_path, verbose

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


def data_path(subject, path=None, force_update=False, fnirs=False):
    """Get path to local copy of bbci_eeg_fnirs dataset URL.

    Parameters
    ----------
    subject : int
        Number of subject to use
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BBCIFNIRS_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """  # noqa: E501
    if subject < 1 or subject > 30:
        raise ValueError(
            "Valid subjects between 1 and 30, subject {:d} requested".format(subject))
    key = 'MNE_DATASETS_BBCIFNIRS_PATH'
    path = _get_path(path, key, 'BBCI EEG-fNIRS')
    _do_path_update(path, None, key, 'BBCI EEG-fNIRS')
    if not op.isdir(op.join(path, 'MNE-eegfnirs-data')):
        os.makedirs(op.join(path, 'MNE-eegfnirs-data'))
    if fnirs:
        return fnirs_data_path(op.join(path, 'MNE-eegfnirs-data'), subject)
    else:
        return eeg_data_path(op.join(path, 'MNE-eegfnirs-data'), subject)


class BBCIEEGfNIRS(BaseDataset):
    """BBCI EEG fNIRS Motor Imagery dataset"""

    def __init__(self, fnirs=False):
        super().__init__(subjects=list(range(1,30)),
                         sessions_per_subject=1,
                         events=dict(left_hand=1, right_hand=2, subtraction=3, rest=4),
                         code='BBCI EEG fNIRS',
                         interval=[3.5,10],
                         paradigm='imagery')
        self.fnirs = fnirs      # TODO: actually incorporate fNIRS somehow 

    def get_data(self, subjects, stack_sessions=False):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.extend(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname, fname_mrk = data_path(subject)
        raws = []
        data = loadmat(fname, squeeze_me=True, struct_as_record=False)['cnt']
        mrk = loadmat(fname_mrk, squeeze_me=True,
                      struct_as_record=False)['mrk']
        montage = read_montage('standard_1005')

        for ii in [0,2,4]:
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
        # arithmetic/rest
        for ii in [1,3,5]:
            eeg = data[ii].x.T * 1e-6
            trig = np.zeros((1, eeg.shape[1]))
            idx = (mrk[ii].time - 1) // 5
            trig[0, idx] = mrk[ii].event.desc // 16 + 2
            eeg = np.vstack([eeg, trig])
            ch_names = list(data[ii].clab) + ['Stim']
            ch_types = ['eeg'] * 30 + ['eog'] * 2 + ['stim']

            info = create_info(ch_names=ch_names, ch_types=ch_types,
                               sfreq=200., montage=montage)
            raw = RawArray(data=eeg, info=info, verbose=False)
            raws.append(raw)
        return [raws]
