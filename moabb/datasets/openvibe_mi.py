"""
Openvibe Motor imagery dataset.
"""

from .base import BaseDataset

import pandas as pd
import os
from mne import create_info
from mne.io import RawArray, Raw
from mne.channels import read_montage
from . import download as dl


INRIA_URL = 'http://openvibe.inria.fr/private/datasets/dataset-1/'

def data_path(session, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of INRIA dataset URL.

    Parameters
    ----------
    session : int
        Number of session to use
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
    if session < 1 or session > 14:
        raise ValueError("Valid sessions between 1 and 14, session {:d} requested".format(session))


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
    raw = RawArray(data=csv_data.values.T * 1e-6, info=info, verbose=False)
    return raw


class OpenvibeMI(BaseDataset):
    """Openvibe Motor Imagery dataset"""

    def __init__(self, tmin=0, tmax=3):
        super().__init__(
            subjects=[1],
            sessions_per_subject=14,
            events=dict(right_hand=1, left_hand=2),
            code='Openvibe Motor Imagery',
            interval=[tmin, tmax],
            paradigm='imagery')

    def _get_single_subject_data(self, subjects, stack_sessions=False):
        """return data for subject"""
        data = []
        for i in range(1, 10):
            data.append(self._get_single_session_data(i))
        if stack_sessions:
            return [data]
        else:
            return [[data]]

    def _get_single_session_data(self, session):
        """return data for a single recording session"""
        csv_path = self.data_path(1)[session]
        fif_path = os.path.join(os.path.dirname(csv_path),
                                'raw_{:d}.fif'.format(session))
        if not os.path.isfile(fif_path):
            print('Resaving .csv file as .fif for ease of future loading')
            raw = convert_inria_csv_to_mne(csv_path)
            raw.save(fif_path)
            return raw
        else:
            return Raw(fif_path, preload=True, verbose='ERROR')

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        paths = []
        for session in range(1, 15):
            url = '{:s}{:02d}-signal.csv.bz2'.format(INRIA_URL, session)
            paths.append(dl.data_path(url, 'INRIA', path, force_update,
                         update_path, verbose))
        return paths
