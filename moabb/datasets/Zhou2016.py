"""Simple and compound motor imagery.

https://doi.org/10.1371/journal.pone.0114853
"""

import json
import logging
import zipfile as ZipFile
from pathlib import Path

import requests

from .base import BaseBIDSDataset
from .download import download_if_missing, get_dataset_path


log = logging.getLogger(__name__)


ZENODO_RECORD_ID = 16534752
# Zenodo API endpoint for published records
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


class Zhou2016(BaseBIDSDataset):
    """Motor Imagery dataset from Zhou et al 2016.

    Dataset from the article *A Fully Automated Trial Selection Method for
    Optimization of Motor Imagery Based Brain-Computer Interface* [1]_.
    This dataset contains data recorded on 4 subjects performing 3 type of
    motor imagery: left hand, right hand and feet.

    Every subject went through three sessions, each of which contained two
    consecutive runs with several minutes inter-run breaks, and each run
    comprised 75 trials (25 trials per class). The intervals between two
    sessions varied from several days to several months.

    A trial started by a short beep indicating 1 s preparation time,
    and followed by a red arrow pointing randomly to three directions (left,
    right, or bottom) lasting for 5 s and then presented a black screen for
    4 s. The subject was instructed to immediately perform the imagination
    tasks of the left hand, right hand or foot movement respectively according
    to the cue direction, and try to relax during the black screen.

    References
    ----------

    .. [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated
           Trial Selection Method for Optimization of Motor Imagery Based
           Brain-Computer Interface. PLoS ONE 11(9).
           https://doi.org/10.1371/journal.pone.0162657
    """

    def __init__(self):
        """Initialize the BIDS dataset."""
        super().__init__(
            subjects=list(range(1, 5)),
            sessions_per_subject=3,
            events=dict(left_hand=1, right_hand=2, feet=3),
            code="Zhou2016",
            # MI 1-6s, prepare 0-1, break 6-10
            # boundary effects
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1371/journal.pone.0162657",
        )
        self.zenodo_record_id = ZENODO_RECORD_ID

    def get_metainfo(record_id, path=None):
        """Fetch a Zenodo record by its ID."""
        # first thing try to get the record from the path if already downloaded
        file_path = f"{path}/{record_id}.json"
        if not Path(file_path).exists():
            # If not found, fetch from Zenodo
            response = requests.get(ZENODO_URL)
            response.raise_for_status()
            # Save the response to a file
            with open(file_path, "w") as f:
                json.dump(response.json(), f, indent=4)
            return response.json()
        else:
            with open(file_path, "r") as f:
                return json.load(f)

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download the subject data."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        dataset_path = get_dataset_path(self.code, path=path)

        metainfo = self.get_metainfo(self.zenodo_record_id, path=dataset_path)

        for file in metainfo["files"]:
            file_name = file["key"]
            file_url = file["links"]["self"]

            file_path = dataset_path / file_name
            if "sub" in file_name:
                # Check if the file corresponds to the current subject
                if file_name == f"sub-{subject}.zip":

                    if not file_path.exists():
                        print(f"Downloading {file_name} for subject {subject}")
                        download_if_missing(
                            file_path=file_path, url=file_url, warn_missing=True
                        )

                        folder_path = file_path.with_suffix("")

                        if not folder_path.exists():
                            with ZipFile(file_path, "r") as zip_ref:
                                zip_ref.extractall(folder_path)
            else:
                download_if_missing(file_path=file_path, url=file_url, warn_missing=True)

            return dataset_path
