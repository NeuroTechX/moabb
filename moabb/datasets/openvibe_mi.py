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


def convert_inria_csv_to_mne(path):
    '''
    Convert an INRIA CSV file to a RawArray
    '''

    csv_data = pd.read_csv(path, index_col=0, sep=',')
    csv_data = csv_data.drop(['Epoch', 'Event Date', 'Event Duration'], axis=1)
    csv_data = csv_data.rename(columns={'Event Id': 'Stim', 'Ref_Nose': 'Nz'})
    ch_types = ['eeg']*11 + ['stim']
    ch_names = list(csv_data.columns)
    left_hand_ind = csv_data['Stim'] == '769'
    right_hand_ind = csv_data['Stim'] == '770'
    csv_data['Stim'] = 0
    csv_data['Stim'][left_hand_ind] = 2e6
    csv_data['Stim'][right_hand_ind] = 1e6
    montage = read_montage('standard_1005')
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=512.,
                       montage=montage)
    raw = RawArray(data=csv_data.values.T * 1e-6, info=info, verbose=False)
    return raw


class OpenvibeMI(BaseDataset):
    """Openvibe Motor Imagery dataset.

    This datasets includes 14 records of left and right hand motor imagery from
    a single subject. They include 11 channels : C3, C4, Nz, FC3, FC4, C5, C1,
    C2, C6, CP3 and CP4. The channels are recorded in common average mode and
    Nz can be used as a reference if needed. The signal is sampled at 512 Hz
    and was recorded with our Mindmedia NeXus32B amplifier.

    Each file consists in 40 trials where the subject was requested to imagine
    either left or right hand movements (20 each). The experiment followed the
    Graz University protocol [1]_.

    The files were recorded on three different days of the same month.

    The data set has been used in the paper [2]_.

    references
    ----------

    .. [1] Pfurtscheller, G. & Neuper, C. Motor Imagery and Direct
           Brain-Computer Communication. Proceedings of the IEEE, 89,
           1123-1134, 2001.

    .. [2] N. Brodu, F. Lotte, A. Lécuyer. Exploring Two Novel Features for
           EEG-based Brain-Computer Interfaces: Multifractal Cumulants and
           Predictive Complexity. Neurocomputing 79: 87-94, 2012.


    """

    def __init__(self):
        super().__init__(
            subjects=[1],
            sessions_per_subject=3,
            events=dict(right_hand=1, left_hand=2),
            code='Openvibe Motor Imagery',
            # 5 second is the duration of the feedback in the OV protocol.
            interval=[0, 5],
            paradigm='imagery')

    def _get_single_subject_data(self, subject):
        """return data for subject"""
        data = {}

        # data are recorded on 3 different day (session). it's not specified
        # wich run is wich session, but by looking at the data, we can identify
        # the 3 sessions.

        sessions = [[1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14]]

        for jj, session in enumerate(sessions):
            for ii, run in enumerate(session):
                raw = self._get_single_run_data(run)
                data["session_%d" % jj] = {'run_%d' % ii: raw}

        return data

    def _get_single_run_data(self, run):
        """return data for a single recording session"""
        csv_path = self.data_path(1)[run - 1]
        fif_path = os.path.join(os.path.dirname(csv_path),
                                'raw_{:d}.fif'.format(run))
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
