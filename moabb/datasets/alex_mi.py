"""
Alex Motor imagery dataset.
"""

from .base import BaseDataset
from mne.io import Raw
import os

from . import download as dl

ALEX_URL = 'https://zenodo.org/record/806023/files/'

def data_path(subject, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of ALEX dataset URL.

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
    if subject < 1 or subject > 8:
        raise ValueError("Valid subjects between 1 and 8, subject {:d} requested".format(subject))
    url = '{:s}subject{:d}.raw.fif'.format(ALEX_URL, subject)


    return dl.data_path(url, 'ALEXEEG', path, force_update, update_path, verbose)
    
class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset"""

    def __init__(self):
        super().__init__(
            subjects=list(range(1,9)),
            sessions_per_subject=1,
            events=dict(right_hand=2, feet=3, rest=4),
            code='Alexandre Motor Imagery',
            interval=[0,3],
            paradigm='imagery'
            )

    def _get_single_subject_data(self, subject, multi_session):
        """return data for a single subject"""
        raw = Raw(data_path(subject), preload=True)
        return [[raw]]
