"""
Alex Motor imagery dataset.
"""

from .base import BaseDataset
from mne.io import Raw


class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset"""

    def __init__(self, with_rest=False):
        self.subject_list = range(1, 9)
        self.name = 'Alex Motor Imagery'
        self.tmin = 0
        self.tmax = 3
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(right_hand=2, feets=3)
        if with_rest:
            self.event_id['rest'] = 4

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = ('/home/kirsh/Documents/Data/LE2S_Multiclass/subject%d.raw.fif'
                 % subject)
        raw = Raw(fname, preload=True)
        return [raw]
