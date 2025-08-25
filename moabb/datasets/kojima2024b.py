import json
import os
import re
import string
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

from moabb.datasets import download as dl

from .base import BaseDataset


_manifest_link = "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.7910/DVN/1UJDV6"
_api_base_url = "https://dataverse.harvard.edu/api/access/datafile/"

EVENTS = {
    "D1": [101, 111],
    "D2": [102, 112],
    "D3": [103, 113],
    "D4": [104, 114],
    "S1": [1],
    "S2": [2],
    "Target": [111, 112, 113, 114],
    "NonTarget": [101, 102, 103, 104],
}


def _extract_run_number(path):
    match = re.search(r"run-(\d+)", path.name)
    return int(match.group(1)) if match else -1


class Kojima2024B(BaseDataset):
    """
    Parent class of Kojima2024_2 and Kojima2024_4
    Should not be instantiated.
    """

    def __init__(
        self,
        events,
    ):
        self.subject_list = list(range(1, 16))
        self.n_channels = 64

        super().__init__(
            self.subject_list,
            sessions_per_subject=1,
            events=events,
            code="Kojima2024B",
            interval=[-0.5, 1.2],
            paradigm="p300",
            doi="10.7910/DVN/1UJDV6",
        )

    def _block_rep(self, task, run):
        assert task == "2stream" or task == "4stream"
        assert run >= 0 and run <= 6
        assert int(run) == run  # run should be integer
        return f"{run}{task}"

    def _get_files_list(self, subject, manifest):

        subject_id = self.convert_subject_to_subject_id(subject)

        manifest_files = manifest["datasetVersion"]["files"]

        files_to_load = []

        for file in manifest_files:

            if (
                (f"sub-{subject_id}" not in file["label"])
                or ("stream_" not in file["label"])
                or ("_eeg" not in file["label"])
            ):
                continue

            fname = file["label"]
            directory = file["directoryLabel"]
            file_id = file["dataFile"]["id"]

            files_to_load.append(
                {"fname": fname, "directory": directory, "file_id": file_id}
            )

        return files_to_load

    def convert_subject_to_subject_id(self, subjects):
        """
        Convert subject number(s) to subject ID(s).
        (In this dataset, subject IDs are encoded using alphabet letters.)

        Parameters
        ----------
        subjects : int or list of int
            Subject number(s) to convert.

        Returns
        -------
        subject_id : str or list of str
            Converted subject ID(s).
        """

        if isinstance(subjects, int):
            subject_id = list(string.ascii_uppercase)[subjects - 1]
        elif isinstance(subjects, list):
            subject_id = []
            for subject in subjects:
                subject_id.append(list(string.ascii_uppercase)[subject - 1])
        else:
            raise TypeError("Type of subejcts must be either int or list.")

        return subject_id

    def data_path(self, subject, path=None):
        """
        Return the data paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn't exist, the “~/mne_data” directory is used. If the
            dataset is not found under the given path, the data
            will be automatically downloaded to the specified folder.

        Returns
        -------
        list
            A list containing the Path object for the subject's data file.
        """

        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download and extract the dataset
        dataset_path = self.download_by_subject(subject=subject, path=path)

        subject_id = self.convert_subject_to_subject_id(subject)

        files = os.listdir(dataset_path / f"sub-{subject_id}" / "eeg")

        paths = []
        for file in files:
            if file.endswith(".vhdr"):
                paths.append(dataset_path / f"sub-{subject_id}" / "eeg" / file)

        paths = sorted(paths, key=_extract_run_number)

        return paths

    def download_by_subject(self, subject, path=None):
        """
        Download and extract the dataset.

        Parameters
        ----------
        subject : int
            The subject number to download the dataset for.

        path : str | None
            The path to the directory where the dataset should be downloaded.
            If None, the default directory is used.


        Returns
        -------
        path : str
            The dataset path.
        """

        path = Path(dl.get_dataset_path(self.code, path)) / (f"MNE-{self.code}-data")

        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(path / "kojima2024_manifest.json", _manifest_link)

        with open(path / "kojima2024_manifest.json", "r") as f:
            manifest = json.load(f)

        files = self._get_files_list(subject, manifest)

        for file in tqdm(files):
            download_url = _api_base_url + str(file["file_id"])
            dl.download_if_missing(
                path / file["directory"] / file["fname"],
                download_url,
                warn_missing=False,
            )

        return path

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            A dictionary containing the raw data for the subject.
        """

        # Get the file path for the subject's data
        files_path = self.data_path(subject)
        runs = {}
        for file in files_path:

            for task in ["2stream", "4stream"]:

                fname = file.name

                run = int(fname.split("_")[2].split("-")[1])

                run = _get_run_num_for_task(run, task)

                if run == -1:
                    continue

                raw = mne.io.read_raw_brainvision(file, eog=["vEOG", "hEOG"])
                raw = raw.load_data()

                raw = raw.set_montage("standard_1020")

                # Get events from annotations and create a stimulus channel
                events, _ = mne.events_from_annotations(raw)

                # Create stimulus channel data from events
                stim_data = np.zeros(raw.n_times)

                # Set stimulus channel values directly from events
                event_samples = events[:, 0]
                event_codes = events[:, 2]
                stim_data[event_samples] = event_codes

                # Create stimulus channel info
                stim_info = mne.create_info(
                    ch_names=["STI"], sfreq=raw.info["sfreq"], ch_types=["stim"]
                )

                # Create RawArray for stimulus channel
                stim_raw = mne.io.RawArray(stim_data[np.newaxis, :], stim_info)

                # Add stimulus channel to raw data
                raw.add_channels([stim_raw], force_update_info=True)

                runs.update({f"{run}{task}": raw})

        sessions = {"0": runs}
        return sessions


def _get_run_num_for_task(run, task):
    """
    Get the sequential run number for a given task.

    In this dataset, experimental runs were conducted in the order:
    run-1, run-2, run-3, run-4, run-5, run-6, run-7, ...
    where different tasks (e.g., "2stream", "4stream") alternated across runs.
    For example:
        - Task "2stream" corresponds to runs 1, 3, 5, 8, 10, 12
        - Task "4stream" corresponds to runs 2, 4, 6, 7, 9, 11

    This function converts the original run index into a task-specific
    sequential run number (starting from 1), so that each task has its own
    independent ascending run numbering.

    Parameters
    ----------
    run : int
        The original run index in the experiment (e.g., 1, 2, 3, ...).
    task : {"2stream", "4stream"}
        The task name.

    Returns
    -------
    int
        The sequential run number for the specified task.

    Examples
    --------
    >>> _get_run_num_for_task(1, "2stream")
    1
    >>> _get_run_num_for_task(3, "2stream")
    2
    >>> _get_run_num_for_task(4, "4stream")
    2
    """

    mapping = {
        "2stream": {1: 1, 3: 2, 5: 3, 8: 4, 10: 5, 12: 6},
        "4stream": {2: 1, 4: 2, 6: 3, 7: 4, 9: 5, 11: 6},
    }

    if run not in mapping[task]:
        return -1
    return mapping[task][run]
