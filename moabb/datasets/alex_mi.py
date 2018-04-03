"""
Alex Motor imagery dataset.
"""

from .base import BaseDataset
from mne.io import Raw

from . import download as dl

ALEX_URL = 'https://zenodo.org/record/806023/files/'


class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset"""

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events=dict(right_hand=2, feet=3, rest=4),
            code='Alexandre Motor Imagery',
            interval=[0, 3],
            paradigm='imagery'
            )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        raw = Raw(self.data_path(subject), preload=True, verbose='ERROR')
        return {"session_0": {"run_0": raw}}

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))
        url = '{:s}subject{:d}.raw.fif'.format(ALEX_URL, subject)
        return dl.data_path(url, 'ALEXEEG', path, force_update, update_path,
                            verbose)
