"""
Physionet Motor imagery dataset.
"""

from .base import BaseDataset
import numpy as np
from mne.io import read_raw_edf
import mne
from mne.datasets import eegbci


class PhysionetMI(BaseDataset):
    """Physionet Motor Imagery dataset"""

    def __init__(self, imagined=True):
        super().__init__(
            subjects=list(range(1,110)),
            sessions_per_subject=1,
            events=dict(left_hand=2, right_hand=3, feet=5, hands=4, rest=1),
            code='Physionet Motor Imagery',
            interval=[1,3],
            paradigm='imagery'
            )

        if imagined:
            self.feet_runs = [6, 10, 14]
            self.hand_runs = [4, 8, 12]
        else:
            self.feet_runs = [5, 9, 13]
            self.hand_runs = [3, 7, 11]

    def _get_single_subject_data(self, subject, stack_sessions):
        """return data for a single subject"""
        all_files = []
        raw_fnames = eegbci.load_data(subject, runs=self.hand_runs, verbose='ERROR', base_url='http://www.physionet.org/pn4/eegmmidb/')
        raw_files = [read_raw_edf(f, preload=True, verbose='ERROR')
                     for f in raw_fnames]

        # strip channel names of "." characters
        [raw.rename_channels(lambda x: x.strip('.')) for raw in raw_files]
        all_files.extend(raw_files)

        raw_feet_fnames = eegbci.load_data(subject, runs=self.feet_runs, verbose='ERROR')
        raw_feet_files = [read_raw_edf(f, preload=True, verbose='ERROR')
                     for f in raw_feet_fnames]
        for raw in raw_feet_files:
            events = mne.find_events(raw)
            # Event values are added together, previous and current (1,2,3->1,4,5)
            events[events[:, 2] != 1, 2] = 2
            events[events[:, 2] == 1, 2] = 0
            raw.add_events(events)
            raw.rename_channels(lambda x: x.strip('.'))
        all_files.extend(raw_feet_files)
        if stack_sessions:
            return [all_files]
        else:
            return [[all_files]]
