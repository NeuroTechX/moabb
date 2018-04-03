"""
Physionet Motor imagery dataset.
"""

from .base import BaseDataset
from mne.io import read_raw_edf
import mne
from mne.datasets import eegbci

BASE_URL = 'http://www.physionet.org/pn4/eegmmidb/'


class PhysionetMI(BaseDataset):
    """Physionet Motor Imagery dataset"""

    def __init__(self, imagined=True):
        super().__init__(
            subjects=list(range(1, 110)),
            sessions_per_subject=1,
            events=dict(left_hand=2, right_hand=3, feet=5, hands=4, rest=1),
            code='Physionet Motor Imagery',
            interval=[1, 3],
            paradigm='imagery'
            )

        if imagined:
            self.feet_runs = [6, 10, 14]
            self.hand_runs = [4, 8, 12]
        else:
            self.feet_runs = [5, 9, 13]
            self.hand_runs = [3, 7, 11]

    def _load_one_run(self, subject, run, preload=True):
        raw_fname = eegbci.load_data(subject, runs=[run], verbose='ERROR',
                                     base_url=BASE_URL)[0]
        raw = read_raw_edf(raw_fname, preload=preload, verbose='ERROR')
        raw.rename_channels(lambda x: x.strip('.'))
        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        data = {}

        # hand runs
        for run in self.hand_runs:
            data['run_%d' % run] = self._load_one_run(subject, run)

        # feet_runs runs
        for run in self.feet_runs:
            raw = self._load_one_run(subject, run)

            # modify stim channels to match new event ids. for feets runs,
            # hand = 2 modified to 4, and feet = 3, modified to 5
            stim = raw._data[-1]
            raw._data[-1, stim == 2] = 4
            raw._data[-1, stim == 3] = 5
            data['run_%d' % run] = raw

        return {"session_0": data}

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        paths = []
        paths += eegbci.load_data(subject, runs=self.feet_runs,
                                  verbose=verbose)
        paths += eegbci.load_data(subject, runs=self.hand_runs,
                                  verbose=verbose)
        return paths
