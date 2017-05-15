"""
Physionet Motor imagery dataset.
"""

from .base import BaseDataset

from mne.io import read_raw_edf
from mne.datasets import eegbci


class PhysionetMI(BaseDataset):
    """Physionet Motor Imagery dataset"""

    def __init__(self, imagined=True, feets=True):
        self.subject_list = range(1, 110)
        self.name = 'Physionet Motor Imagery'

        if feets:
            self.events_id = dict(rest=0, hands=1, feets=2)
            if imagined:
                self.selected_runs = [6, 10, 14]
            else:
                self.selected_runs = [5, 9, 13]
        else:
            self.events_id = dict(rest=0, left_hand=1, right_hand=2)
            if imagined:
                self.selected_runs = [4, 8, 12]
            else:
                self.selected_runs = [3, 7, 11]

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        raw_fnames = eegbci.load_data(subject, runs=self.selected_runs)
        raw_files = [read_raw_edf(f, preload=True, verbose=False)
                     for f in raw_fnames]

        # strip channel names of "." characters
        [raw.rename_channels(lambda x: x.strip('.')) for raw in raw_files]

        return raw_files
