# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Sylvain Chevallier <sylvain.chevallier@uvsq.fr>
#         Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD Style.

import json
import os
import os.path as osp
from pathlib import Path

import requests
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from mne.utils import _url_to_local_path, verbose
from pooch import file_hash, retrieve
from pooch.downloaders import choose_downloader
from requests.exceptions import HTTPError


def get_dataset_path(sign, path):
    """Returns the dataset path allowing for changes in MNE_DATA config.

    Parameters
    ----------
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.

    Returns
    -------
        path : None | str
        Location of where to look for the data storing location
    """
    sign = sign.upper()
    key = "MNE_DATASETS_{:s}_PATH".format(sign)
    if get_config(key) is None:
        if get_config("MNE_DATA") is None:
            path_def = Path.home() / "mne_data"
            print(
                "MNE_DATA is not already configured. It will be set to "
                "default location in the home directory - "
                + str(path_def)
                + "\nAll datasets will be downloaded to this location, if anything is "
                "already downloaded, please move manually to this location"
            )
            if not path_def.is_dir():
                path_def.mkdir(parents=True)
            set_config("MNE_DATA", str(Path.home() / "mne_data"))
        set_config(key, get_config("MNE_DATA"))
    return _get_path(path, key, sign)


@verbose
def data_path(url, sign, path=None, force_update=False, update_path=True, verbose=None):
    """Get path to local copy of given dataset URL. **Deprecated**

    This is a low-level function useful for getting a local copy of a
    remote dataset. It is deprecated in favor of data_dl.

    Parameters
    ----------
    url : str
        Path to remote location of data
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None, **Deprecated**
        Unused, kept for compatibility purpose.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """  # noqa: E501
    path = get_dataset_path(sign, path)
    key_dest = "MNE-{:s}-data".format(sign.lower())
    destination = _url_to_local_path(url, osp.join(path, key_dest))
    # Fetch the file
    if not osp.isfile(destination) or force_update:
        if osp.isfile(destination):
            os.remove(destination)
        if not osp.isdir(osp.dirname(destination)):
            os.makedirs(osp.dirname(destination))
        retrieve(url, None, path=destination)
    return destination


@verbose
def data_dl(url, sign, path=None, force_update=False, verbose=None):
    """Download file from url to specified path.

    This function should replace data_path as the MNE will not support the download
    of dataset anymore. This version is using Pooch.

    Parameters
    ----------
    url : str
        Path to remote location of data
    sign : str
        Signifier of dataset
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_(signifier)_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """
    path = Path(get_dataset_path(sign, path))
    key_dest = "MNE-{:s}-data".format(sign.lower())
    destination = _url_to_local_path(url, path / key_dest)
    destination = str(path) + destination.split(str(path))[1]
    table = {ord(c): "-" for c in ':*?"<>|'}
    destination = Path(str(path) + destination.split(str(path))[1].translate(table))

    downloader = choose_downloader(url, progressbar=True)
    if type(downloader).__name__ in ["HTTPDownloader", "DOIDownloader"]:
        downloader.kwargs.setdefault("verify", False)

    # Fetch the file
    if not destination.is_file() or force_update:
        if destination.is_file():
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        known_hash = None
    else:
        known_hash = file_hash(str(destination))
    dlpath = retrieve(
        url,
        known_hash,
        fname=Path(url).name,
        path=str(destination.parent),
        progressbar=True,
        downloader=downloader,
    )
    return dlpath


# This function is from https://github.com/cognoma/figshare (BSD-3-Clause)
def fs_issue_request(method, url, headers, data=None, binary=False):
    """Wrapper for HTTP request.

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
        response = fs_issue_request("GET", url, headers=headers)
        return response
    else:
        url = fsurl + "/articles/{}/versions/{}".format(article_id, version)
        headers = {"Content-Type": "application/json"}
        request = fs_issue_request("GET", url, headers=headers)
        return request["files"]


def fs_get_file_hash(filelist):
    """Returns a dict associating figshare file id to MD5 hash.

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
    """Returns a dict associating filename to figshare file id.

    Parameters
    ----------
    filelist : list of dict
        HTTP request response from fs_get_file_list

    Returns
    -------
    response : dict
        keys are filename and values are file_id
    """
    return {f["name"]: str(f["id"]) for f in filelist}


def fs_get_file_name(filelist):
    """Returns a dict associating figshare file id to filename.

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
