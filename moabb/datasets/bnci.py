"""
BNCI 2014-001 Motor imagery dataset.
"""

from .base import BaseDataset
from mne.datasets.bnci import load_data


class MNEBNCI(BaseDataset):
    """Base BNCI dataset"""

    def get_data(self, subjects):
        """return data for a list of subjects."""
        data = []
        for subject in subjects:
            data.append(self._get_single_subject_data(subject))
        return data

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        raw_files, event_id = load_data(subject=subject, dataset=self.code,
                                        verbose=False)
        return raw_files


class BNCI2014001(MNEBNCI):
    """BNCI 2014-001 Motor Imagery dataset"""

    def __init__(self, left_hand=True, right_hand=True, feets=True,
                 tongue=True, tmin=3.5, tmax=5.5):
        self.subject_list = range(1, 10)
        self.name = 'BNCI 2014-001 Motor Imagery'
        self.code = '001-2014'
        self.tmin = tmin
        self.tmax = tmax
        self.paradigm = 'Motor Imagery'
        event_id = dict()
        if left_hand:
            event_id['left_hand'] = 1
        if right_hand:
            event_id['right_hand'] = 2
        if feets:
            event_id['feets'] = 3
        if tongue:
            event_id['tongue'] = 4

        self.event_id = event_id


class BNCI2014002(MNEBNCI):
    """BNCI 2014-002 Motor Imagery dataset"""

    def __init__(self, tmin=3.5, tmax=5.5):
        self.subject_list = range(1, 15)
        self.name = 'BNCI 2014-002 Motor Imagery'
        self.code = '002-2014'
        self.tmin = tmin
        self.tmax = tmax
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(right_hand=1, feets=2)


class BNCI2014004(MNEBNCI):
    """BNCI 2014-004 Motor Imagery dataset"""

    def __init__(self, tmin=4.5, tmax=6.5):
        self.subject_list = range(1, 10)
        self.name = 'BNCI 2014-004 Motor Imagery'
        self.code = '004-2014'
        self.tmin = tmin
        self.tmax = tmax
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(left_hand=1, right_hand=2)


class BNCI2015001(MNEBNCI):
    """BNCI 2015-001 Motor Imagery dataset"""

    def __init__(self, tmin=4, tmax=7.5):
        self.subject_list = range(1, 13)
        self.name = 'BNCI 2015-001 Motor Imagery'
        self.code = '001-2015'
        self.tmin = tmin
        self.tmax = tmax
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(right_hand=1, feets=2)
