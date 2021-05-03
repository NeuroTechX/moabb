# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
# License: BSD Style.

import json
import os
import os.path as osp

import requests
from mne import get_config, set_config
from mne.datasets.utils import _do_path_update, _get_path
from mne.utils import _fetch_file, _url_to_local_path, verbose
from requests.exceptions import HTTPError


@verbose
def data_path(url, sign, path=None, force_update=False, update_path=True, verbose=None):
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
    key = "MNE_DATASETS_{:s}_PATH".format(sign)
    key_dest = "MNE-{:s}-data".format(sign.lower())
    if get_config(key) is None:
        set_config(key, osp.join(osp.expanduser("~"), "mne_data"))
    path = _get_path(path, key, sign)
    destination = _url_to_local_path(url, osp.join(path, key_dest))
    # Fetch the file
    if not osp.isfile(destination) or force_update:
        if osp.isfile(destination):
            os.remove(destination)
        if not osp.isdir(osp.dirname(destination)):
            os.makedirs(osp.dirname(destination))
        _fetch_file(url, destination, print_destination=False)

    # Offer to update the path
    _do_path_update(path, update_path, key, sign)
    return destination


# This function is from https://github.com/cognoma/figshare (BSD-3-Clause)
def issue_request(method, url, headers, data=None, binary=False):
    """Wrapper for HTTP request

    Parameters
    ----------
    method : str
        HTTP method. One of GET, PUT, POST or DELETE
    url : str
        URL for the request
    headers: dict
        HTTP header information
    data: dict
        Figshare article data
    binary: bool
        Whether data is binary or not

    Returns
    -------
    response_data: dict
        JSON response for the request returned as python dict
    """
    if data is not None and not binary:
        data = json.dumps(data)

    response = requests.request(method, url, headers=headers, data=data)

    try:
        response.raise_for_status()
        try:
            response_data = json.loads(response.text)
        except ValueError:
            response_data = response.content
    except HTTPError as error:
        print("Caught an HTTPError: {}".format(error))
        print("Body:\n", response.text)
        raise

    return response_data


def fs_get_file_list(article_id, version=None):
    """List all the files associated with a given article.

    Parameters
    ----------
    article_id : str or int
        Figshare article ID
    version : str or id, default is None
        Figshare article version. If None, selects the most recent version.

    Returns
    -------
    response : dict
        HTTP request response as a python dict
    """
    fsurl = "https://api.figshare.com/v2"
    if version is None:
        url = fsurl + "/articles/{}/files".format(article_id)
        headers = {"Content-Type": "application/json"}
        response = issue_request("GET", url, headers=headers)
        return response
    else:
        url = fsurl + "/articles/{}/versions/{}".format(article_id, version)
        request = issue_request("GET", url, headers=headers)
        return request["files"]


def fs_get_file_hash(filelist):
    """Returns a dict associating figshare file id to MD5 hash

    Parameters
    ----------
    filelist : list of dict
        HTTP request response from fs_get_file_list

    Returns
    -------
    response : dict
        keys are file_id and values are md5 hash
    """
    return {str(f["id"]): "md5:" + f["supplied_md5"] for f in filelist}


def fs_get_file_id(filelist):
    """Returns a dict associating filename to figshare file id

    Parameters
    ----------
    filelist : list of dict
        HTTP request response from fs_get_file_list

    Returns
    -------
    response : dict
        keys are filname and values are file_id
    """
    return {f["name"]: str(f["id"]) for f in filelist}


def fs_get_file_name(filelist):
    """Returns a dict associating figshare file id to filename

    Parameters
    ----------
    filelist : list of dict
        HTTP request response from fs_get_file_list

    Returns
    -------
    response : dict
        keys are file_id and values are file name
    """
    return {str(f["id"]): f["name"] for f in filelist}
