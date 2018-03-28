from moabb.datasets.base import BaseDataset

from .base import BaseDataset
from mne.io import read_raw_edf
from mne.channels import read_montage
import os
import numpy as np

from . import download as dl

UPPER_LIMB_URL = 'https://zenodo.org/record/834976/files/'


def data_paths(subject, imagined=True, path=None, force_update=False,
               update_path=None, verbose=None):
    """Get path to local copy of ALEX dataset URL.

    Parameters
    ----------
    subject : int
        Number of subject to use
    imagined : bool
        If true, return motor imagination run, otherwise return motor execution
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
    if subject < 1 or subject > 15:
        raise ValueError("Valid subjects between 1 and 15,"
                         " subject {:d} requested".format(subject))

    paths = []

    if imagined:
        im = 'imagination'
    else:
        im = 'execution'

    for run in range(1, 11):
        url = f"{UPPER_LIMB_URL}/motor{im}_subject{subject}_run{run}.gdf"
        p = dl.data_path(url, 'UPPER_LIMB', path, force_update, update_path,
                         verbose)
        paths.append(p)

    return paths


class UpperLimb(BaseDataset):
    """Upper Limb motor dataset"""

    def __init__(self, imagined=True):
        self.imagined = imagined
        event_id = {"elbow_flexion": 1536,
                    "elbow_extension": 1537,
                    "supination": 1538,
                    "pronation": 1539,
                    "hand_close": 1540,
                    "hand_open": 1541,
                    "rest": 1542}

        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events=event_id,
            code='Upper Limb Imagery',
            interval=[2.5, 5],
            paradigm='imagery')

    def _get_single_subject_data(self, subject, stack_sessions):
        """return data for a single subject"""
        paths = data_paths(subject, self.imagined)

        eog = ['eog-l', 'eog-m', 'eog-r']
        montage = read_montage('standard_1005')
        raws = []
        for path in paths:
            raw = read_raw_edf(path, montage=montage, eog=eog,
                               misc=range(64, 96), preload=True,
                               verbose='ERROR')
            # there is nan in the data
            raw._data[np.isnan(raw._data)] = 0
            raws.append(raw)
        if stack_sessions:
            return [raws]
        else:
            return [[raws]]
