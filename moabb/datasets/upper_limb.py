from moabb.datasets.base import BaseDataset

from mne.io import read_raw_edf
from mne.channels import read_montage
import numpy as np

from . import download as dl

UPPER_LIMB_URL = 'https://zenodo.org/record/834976/files/'


class UpperLimb(BaseDataset):
    """Upper Limb motor dataset.

    Upper limb dataset :
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182578

    Consist in 6 upper limb movement, recoded over 2 sessions.
    The first session is motor execution, the second session is imagination.

    """

    def __init__(self, imagined=True, executed=False):
        self.imagined = imagined
        self.executed = executed
        event_id = {"right_elbow_flexion": 1536,
                    "right_elbow_extension": 1537,
                    "right_supination": 1538,
                    "right_pronation": 1539,
                    "right_hand_close": 1540,
                    "right_hand_open": 1541,
                    "rest": 1542}

        n_sessions = int(imagined) + int(executed)
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=n_sessions,
            events=event_id,
            code='Upper Limb Imagery',
            interval=[2.5, 5],
            paradigm='imagery',
            doi='10.1371/journal.pone.0182578')

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        sessions = []
        if self.imagined:
            sessions.append('imagination')

        if self.executed:
            sessions.append('execution')

        out = {}
        for session in sessions:
            paths = self.data_path(subject, session=session)

            eog = ['eog-l', 'eog-m', 'eog-r']
            montage = read_montage('standard_1005')
            data = {}
            for ii, path in enumerate(paths):
                raw = read_raw_edf(path, montage=montage, eog=eog,
                                   misc=range(64, 96), preload=True,
                                   verbose='ERROR')
                # there is nan in the data
                raw._data[np.isnan(raw._data)] = 0
                data['run_%d' % ii] = raw

            out[session] = data
        return out

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None, session=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        paths = []

        if session is None:
            sessions = []
            if self.imagined:
                sessions.append('imagination')

            if self.executed:
                sessions.append('execution')
        else:
            sessions = [session]

        for session in sessions:
            for run in range(1, 11):
                url = f"{UPPER_LIMB_URL}motor{session}_subject{subject}_run{run}.gdf"
                p = dl.data_path(url, 'UPPERLIMB', path, force_update,
                                 update_path, verbose)
                paths.append(p)

        return paths
