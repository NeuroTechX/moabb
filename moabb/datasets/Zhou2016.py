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

DATA_PATH = 'https://ndownloader.figshare.com/files/3662952'


def local_data_path(base_path, subject):
    if not os.path.isdir(os.path.join(base_path,
                                      'subject_{}'.format(subject))):
        if not os.path.isdir(os.path.join(base_path, 'data')):
            _fetch_file(DATA_PATH, os.path.join(base_path, 'data.zip'),
                        print_destination=False)
            with z.ZipFile(os.path.join(base_path, 'data.zip'), 'r') as f:
                f.extractall(base_path)
            os.remove(os.path.join(base_path, 'data.zip'))
        datapath = os.path.join(base_path, 'data')
        for i in range(1, 5):
            os.makedirs(os.path.join(base_path, 'subject_{}'.format(i)))
            for session in range(1,4):
                for run in ['A','B']:
                    os.rename(os.path.join(datapath, 'S{}_{}{}.cnt'.format(i,session, run)),
                              os.path.join(base_path,
                                           'subject_{}'.format(i),
                                           '{}{}.cnt'.format(session,run)))
        shutil.rmtree(os.path.join(base_path, 'data'))
    subjpath = os.path.join(base_path, 'subject_{}'.format(subject))
    return [[os.path.join(subjpath,
                          '{}{}.cnt'.format(y, x)) for x in ['A', 'B']] for y in ['1', '2', '3']]


class Zhou2016(BaseDataset):
    """Dataset from Zhou et al. 2016 [1]

    Abstract
    ------------

    Independent component analysis (ICA) as a promising spatial filtering method
    can separate motor-related independent components (MRICs) from the
    multichannel electroencephalogram (EEG) signals. However, the unpredictable
    burst interferences may significantly degrade the performance of ICA-based
    brain-computer interface (BCI) system. In this study, we proposed a new
    algorithm frame to address this issue by combining the single-trial-based
    ICA filter with zero-training classifier. We developed a two-round data
    selection method to identify automatically the badly corrupted EEG trials in
    the training set. The “high quality” training trials were utilized to
    optimize the ICA filter. In addition, we proposed an accuracy-matrix method
    to locate the artifact data segments within a single trial and investigated
    which types of artifacts can influence the performance of the ICA-based
    MIBCIs. Twenty-six EEG datasets of three-class motor imagery were used to
    validate the proposed methods, and the classification accuracies were
    compared with that obtained by frequently used common spatial pattern (CSP)
    spatial filtering algorithm. The experimental results demonstrated that the
    proposed optimizing strategy could effectively improve the stability,
    practicality and classification performance of ICA-based MIBCI. The study
    revealed that rational use of ICA method may be crucial in building a
    practical ICA-based MIBCI system.

    References
    ------------

    [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated Trial
    Selection Method for Optimization of Motor Imagery Based Brain-Computer
    Interface. PLoS ONE 11(9):
    e0162657. https://doi.org/10.1371/journal.pone.0162657

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 5)),
            sessions_per_subject=3,
            events=dict(left_hand=1, right_hand=2,
                        feet=3),
            code='Zhou 2016',
            # MI 1-6s, prepare 0-1, break 6-10
            # boundary effects
            interval=[0, 5],
            task_interval=[1,6],
            paradigm='imagery',
            doi='10.1371/journal.pone.0162657')

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        files = self.data_path(subject)

        out = {}
        for sess_ind, runlist in enumerate(files):
            sess_key = 'session_{}'.format(sess_ind)
            out[sess_key] = {}
            for run_ind, fname in enumerate(runlist):
                run_key = 'run_{}'.format(run_ind)
                out[sess_key][run_key] = mne.io.read_raw_cnt(fname,
                                                             preload=True,
                                                             montage='standard_1020')
        return out

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))
        key = 'MNE_DATASETS_ZHOU2016_PATH'
        path = _get_path(path, key, "Zhou 2016")
        _do_path_update(path, True, key, "Zhou 2016")
        basepath = os.path.join(path, "MNE-zhou-2016")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)
        return local_data_path(basepath, subject)
