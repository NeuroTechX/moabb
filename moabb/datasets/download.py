# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
# License: BSD Style.

import os
import os.path as osp
from os import path as op

from mne import get_config, set_config
from mne.datasets.utils import _do_path_update, _get_path
from mne.utils import _fetch_file, _url_to_local_path, verbose


@verbose
def data_path(url, sign, path=None, force_update=False, update_path=True,
              verbose=None):
    """Get path to local copy of given dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote dataset

    Parameters
    ----------
    url : str
        Path to remote location of data
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the BNCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_(signifier)_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    """  # noqa: E501
    sign = sign.upper()
    key = 'MNE_DATASETS_{:s}_PATH'.format(sign)
    key_dest = 'MNE-{:s}-data'.format(sign.lower())
    if get_config(key) is None:
        set_config(key, osp.join(osp.expanduser("~"), "mne_data"))
    path = _get_path(path, key, sign)
    destination = _url_to_local_path(url, op.join(path, key_dest))
    # Fetch the file
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False)

    # Offer to update the path
    _do_path_update(path, update_path, key, sign)
    return destination
