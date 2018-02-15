"""
Openvibe Motor imagery dataset.
"""

from .base import BaseDataset

import pandas as pd
import os
from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
import moabb.datasets.download as dl


INRIA_URL = 'http://openvibe.inria.fr/private/datasets/dataset-1/'

def data_path(subject, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of INRIA dataset URL.

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
    if subject < 1 or subject > 14:
        raise ValueError("Valid subjects between 1 and 14, subject {:d} requested".format(subject))
    url = '{:s}{:02d}-signal.csv.bz2'.format(INRIA_URL, subject)
    return dl.data_path(url, 'INRIA', path, force_update, update_path, verbose)

def convert_inria_csv_to_mne(path):
    '''
    Convert an INRIA CSV file to a RawArray
    '''

    csv_data = pd.read_csv(path, index_col=0, sep=',')
    csv_data = csv_data.drop(['Epoch','Event Date','Event Duration'],axis=1)
    csv_data = csv_data.rename(columns={'Event Id':'Stim', 'Ref_Nose':'Nz'})
    ch_types=['eeg']*11 + ['stim']
    ch_names = list(csv_data.columns)
    left_hand_ind = csv_data['Stim'] == '769'
    right_hand_ind = csv_data['Stim'] == '770'
    csv_data['Stim'] = 0
    csv_data['Stim'][left_hand_ind] = 2e6
    csv_data['Stim'][right_hand_ind] = 1e6
    montage = read_montage('standard_1005')
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=512., montage=montage)
    return [RawArray(data=csv_data.values.T * 1e-6, info=info, verbose=False)]
    
class OpenvibeMI(BaseDataset):
    """Openvibe Motor Imagery dataset"""

    def __init__(self):
        self.subject_list = range(1, 15)
        self.name = 'Openvibe Motor Imagery'
        self.tmin = 0
        self.tmax = 3
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(right_hand=1, left_hand=2)

    def get_data(self, subjects):
        """return data for a list of subjects. NOTE: these are different recordings same subj"""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, session):
        """return data for a single recordign session"""

        return convert_inria_csv_to_mne(data_path(session))
