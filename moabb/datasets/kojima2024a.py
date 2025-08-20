import json
import os
import string
import warnings
from pathlib import Path

import mne
from tqdm import tqdm

from moabb.datasets import download as dl

from .base import BaseDataset


_manifest_link = "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.7910/DVN/MQOVEY"
_api_base_url = "https://dataverse.harvard.edu/api/access/datafile/"


class Kojima2024A(BaseDataset):
    """Class for Kojima2024A dataset management. P300 dataset.

    **Dataset description**

    This dataset [1]_ originates from a study investigating a three-class auditory BCI
    based on auditory stream segregation (ASME-BCI) [2]_.

    In the experiment, participants focused on one of three auditory streams, leveraging
    auditory stream segregation to selectively attend to stimuli in the target stream.
    Each stream contained a two-stimulus oddball sequence composed of one deviant
    stimulus and one standard stimulus.

    The sequence below illustrates an example trial. For instance, when D2 is the target
    stimulus, the participant attended to Stream2 and selectively listened for D2.
    In this case, D2 is the target, and D1 and D3 are considered non-target stimuli.

    .. code-block:: text

        Stream3  ----- S3 -------- S3 -------- S3 -------- D3 -------- S3 -----
        Stream2  -- S2 -------- S2 -------- D2 -------- S2 -------- S2 --------
        Stream1  S1 -------- D1 -------- S1 -------- S1 -------- S1 -----------

    Each participant completed 1 session consisting of 6 runs.
    Each run lasted approximately 5 minutes.
    In each run, all deviant stimuli (D1--D4) were presented approximately 60 times.

    Recording Details:
        - EEG signals were recorded using a BrainAmp system (Brain Products, Germany)
          at a sampling rate of 1000 Hz.

        - Data were collected in Tokyo, Japan, where the power line frequency is 50 Hz.

        - EEG was recorded from 64 scalp electrodes according to the international 10--20 system:
          Fp1, Fp2, AF7, AF3, AFz, AF4, AF8, F7, F5, F3, F1, Fz, F2, F4, F6, F8,
          FT9, FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8, FT10, T7, C5, C3, C1,
          Cz, C2, C4, C6, T8, TP9, TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8,
          TP10, P7, P5, P3, P1, Pz, P2, P4, P6, P8, PO7, PO3, POz, PO4, PO8,
          O1, Oz, O2

          EEG signals were referenced to the right mastoid and grounded to the left mastoid.

        - EOG was recorded using 2 electrodes (vEOG and hEOG), placed above/below and
          lateral to one eye.

    References
    ----------

    .. [1] Kojima, S. (2024).
        Replication Data for: An auditory brain-computer interface based on selective attention to multiple tone streams.
        Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/MQOVEY
    .. [2] Kojima, S. & Kanoh, S. (2024).
        An auditory brain-computer interface based on selective attention to multiple tone streams.
        PLoS ONE 19(5): e0303565. DOI: https://doi.org/10.1371/journal.pone.0303565
    """

    def __init__(self):

        self.subject_list = list(range(1, 12))
        self.n_channels = 64

        super().__init__(
            self.subject_list,
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=0),
            code="Kojima2024A",
            interval=[-0.5, 1.2],
            paradigm="p300",
            doi="10.7910/DVN/MQOVEY",
        )

    def _get_files_list(self, subject, manifest):

        subject_id = self.convert_subject_to_subject_id(subject)

        manifest_files = manifest["datasetVersion"]["files"]

        files_to_load = []

        for file in manifest_files:

            if (f"sub-{subject_id}" not in file["label"]) or (
                "_eeg" not in file["label"]
            ):
                continue

            fname = file["label"]
            directory = "/".join(file["directoryLabel"].split("/")[1:])
            file_id = file["dataFile"]["id"]

            files_to_load.append(
                {"fname": fname, "directory": directory, "file_id": file_id}
            )

        return files_to_load

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

            fname = file.name

            run_id = fname.split("_")[2].split("-")[1]
            task = fname.split("_")[1].split("-")[1]

            annotations_map = {
                "Stimulus/S  2": "NonTarget",
                "Stimulus/S  8": "NonTarget",
                "Stimulus/S 32": "NonTarget",
            }

            if task == "low":
                annotations_map["Stimulus/S  2"] = "Target"
            elif task == "mid":
                annotations_map["Stimulus/S  8"] = "Target"
            elif task == "high":
                annotations_map["Stimulus/S 32"] = "Target"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                raw = mne.io.read_raw_brainvision(file, eog=["vEOG", "hEOG"])
                raw = raw.load_data()

                raw = raw.set_montage("standard_1020")

                raw.annotations.rename(annotations_map)

            runs.update({f"{run_id}{task}": raw})

        sessions = {"0": runs}

        return sessions

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
