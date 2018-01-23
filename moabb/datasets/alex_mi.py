"""
Alex Motor imagery dataset.
"""

from .base import BaseDataset
from mne.io import Raw

from mne.datasets.utils import _get_path, _do_path_update
from mne.utils import _fetch_file, _url_to_local_path, verbose

ALEX_URL = 'https://zenodo.org/record/806023/files/'


@verbose
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
    key = 'MNE_DATASETS_ALEXEEG_PATH'
    name = 'ALEX'
    path = _get_path(path, key, name)
    if subject < 1 or subject > 8:
        raise ValueError("Valid subjects between 1 and 8, subject {:d} requested".format(subject))
    url = '{:s}subject{:d}.raw.fif'.format(ALEX_URL, subject)

    destination = _url_to_local_path(url, os.path.join(path, 'MNE-alexeeg-data'))

    # Fetch the file
    if not os.path.isfile(destination) or force_update:
        if os.path.isfile(destination):
            os.remove(destination)
        if not os.path.isdir(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        _fetch_file(url, destination, print_destination=False)

    # Offer to update the path
    _do_path_update(path, update_path, key, name)
    return destination

class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset"""

    def __init__(self, with_rest=False):
        self.subject_list = range(1, 9)
        self.name = 'Alex Motor Imagery'
        self.tmin = 0
        self.tmax = 3
        self.paradigm = 'Motor Imagery'
        self.event_id = dict(right_hand=2, feet=3)
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
        raw = Raw(data_path(subject), preload=True)
        return [raw]
