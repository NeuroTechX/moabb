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


def _extract_run_number(path):
    match = re.search(r"run-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def markers_from_events(events, event_id):

    event_desc = {v: k for k, v in event_id.items()}

    samples = np.array(events)[:, 0]

    markers = list()
    for val in np.array(events)[:, 2]:
        """
        if "marker:" in str(event_desc[val]):
            markers.append(event_desc[val])
        else:
            markers.append(event_desc[val])
        """
        markers.append(int(event_desc[val]))

    return samples, markers


def split_trials(markers, events=[201, 202, 203, 204], n_events_trial=601, n_trials=4):
    trials = []
    for idx_trial in range(n_trials):

        idx_start = idx_trial * n_events_trial
        idx_end = idx_start + n_events_trial

        trial = markers[idx_start:idx_end]

        # idx 0 is events for trial start
        del trial[0]

        trials.append(trial)

    return trials


def encode_events(markers_trial, run, trial):
    """

    event encoding:

    {run}{trial}{D:1, S:0}{T:1, NT:0}{stim_num}

    e.g.,
    run-1, trial-1, D1, Target: 11111
    run-2, trial-3, S3, NonTarget: 23003

    """

    markers_trial = [run * 10000 + trial * 1000 + m for m in markers_trial]

    return markers_trial


def events_from_markers(samples, markers, offset=0):
    unique_markers = np.unique(markers)

    event_id = dict()
    events = list()
    for marker, sample in zip(markers, samples):
        id = np.argwhere(unique_markers == marker)[0][0] + 1 + offset
        events.append([sample, 0, int(id)])
        event_id[marker] = int(id)
    events = np.array(events)

    return events, event_id


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

    return mapping[task][run]


class _Kojima2024BBase(BaseDataset):
    """
    Parent class of Kojima2024_2 and Kojima2024_4
    Should not be instantiated.
    """

    def __init__(
        self,
        task,
        keep_trial_structure=False,
    ):

        self.task = task

        self.subject_list = list(range(1, 16))
        self.n_channels = 64
        self.keep_trial_structure = keep_trial_structure

        super().__init__(
            self.subject_list,
            sessions_per_subject=1,
            events=dict(Target=1, NonTarget=0),
            code=f"Kojima2024B-{self.task}",
            interval=[-0.5, 1.2],
            paradigm="p300",
            doi="10.7910/DVN/1UJDV6",
        )

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
            task = self.task

            run = int(fname.split("_")[2].split("-")[1])
            run = _get_run_num_for_task(run, task)

            raw = mne.io.read_raw_brainvision(file, eog=["vEOG", "hEOG"])
            raw = raw.load_data()

            raw = raw.set_montage("standard_1020")

            events, event_id = mne.events_from_annotations(raw)

            bv_to_marker_mapping = {}
            for key in list(event_id.keys()):
                bv_to_marker_mapping[key] = str(int(key.split("/")[1][1:]))

            raw.annotations.rename(bv_to_marker_mapping)

            events, event_id = mne.events_from_annotations(raw)
            samples, markers = markers_from_events(events, event_id)

            markers = split_trials(markers)

            new_markers = []
            for idx_trial, markers_trial in enumerate(markers):
                new_markers.append(
                    encode_events(markers[0], run=run, trial=idx_trial + 1)
                )

            markers = [f"{m}" for markers_trial in new_markers for m in markers_trial]
            events, event_id = events_from_markers(samples, markers)

            event_desc = {v: k for k, v in event_id.items()}

            annotations = mne.annotations_from_events(
                events=events, sfreq=raw.info["sfreq"], event_desc=event_desc
            )

            raw = raw.set_annotations(annotations)

            runs.update({f"{run}{task}": raw})

            """
            if self.keep_trial_structure:
                try:
                    import tag_mne as tm
                except ImportError:
                    raise ImportError(
                        "Package tag_mne is required when keep_trial_structure=True. Install it with pip install tag-mne"
                    )

                events, event_id = mne.events_from_annotations(raw)

                bv_to_marker_mapping = {}
                for key in list(event_id.keys()):
                    bv_to_marker_mapping[key] = str(int(key.split("/")[1][1:]))

                raw.annotations.rename(bv_to_marker_mapping)

                events, event_id = mne.events_from_annotations(raw)
                samples, markers = tm.markers_from_events(events, event_id)

                event_names = {
                    "D1": ["101", "111"],
                    "D2": ["102", "112"],
                    "D3": ["103", "113"],
                    "D4": ["104", "114"],
                    "S1": ["1"],
                    "S2": ["2"],
                }

                if self.task == "4stream":
                    event_names["S3"] = ["3"]
                    event_names["S4"] = ["4"]

                markers = tm.add_event_names(markers, event_names)
                markers = tm.add_tag(markers, f"task:{self.task}")
                markers = tm.add_tag(markers, f"run:{run}")

                markers = tm.split_trials(markers, trial=["201", "202", "203", "204"])

                markers = tm.add_tag_to_markers(
                    markers,
                    Target=["111", "112", "113", "114"],
                    NonTarget=["101", "102", "103", "104"],
                    Standard=["1", "2", "3", "4"],
                )

                events, event_id = tm.events_from_markers(samples, markers)

                event_desc = {v: k for k, v in event_id.items()}

                annotations = mne.annotations_from_events(
                    events=events, sfreq=raw.info["sfreq"], event_desc=event_desc
                )

                raw = raw.set_annotations(annotations)

            else:
                annotations_mapping = {
                    "Stimulus/S111": "Target",
                    "Stimulus/S112": "Target",
                    "Stimulus/S113": "Target",
                    "Stimulus/S114": "Target",
                    "Stimulus/S101": "NonTarget",
                    "Stimulus/S102": "NonTarget",
                    "Stimulus/S103": "NonTarget",
                    "Stimulus/S104": "NonTarget",
                }

                raw.annotations.rename(annotations_mapping)

            """

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
            if file.endswith(".vhdr") and f"task-{self.task}" in file:
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


class Kojima2024B_2stream(_Kojima2024BBase):
    """Class for Kojima2024B_2stream dataset management. P300 dataset.

    **Dataset description**

    This dataset [1]_ originates from a study investigating a four-class auditory BCI
    based on auditory stream segregation (ASME-BCI) [2]_.

    In the experiment, participants focused on one of two auditory streams, leveraging
    auditory stream segregation to selectively attend to stimuli in the target stream.
    Each stream contained a three-stimulus oddball sequence composed of two deviant
    stimuli and one standard stimulus.

    The current class corresponds to the "ASME-2stream" condition described in [2]_.
    For the "ASME-4stream" condition, see :class:`Kojima2024B_4stream`.

    The sequence below illustrates an example trial. For instance, when D4 is the target
    stimulus, the participant attended to Stream2 and selectively listened for D4.
    In this case, D4 is the target, and D1, D2, and D3 are considered non-target stimuli.

    .. code-block:: text

        Stream2  -- S2 --- D3 --- S2 --- S2 --- S2 --- S2 --- D4 ---
        Stream1  S1 --- S1 --- D1 --- S1 --- S1 --- D2 --- S1 --- S1

    Each participant completed 1 session consisting of 6 runs.
    Each run included 4 trials, each with a different target stimulus.
    In each trial, all deviant stimuli (D1--D4) were presented 15 times.


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

    Parameters
    ----------

    keep_trial_structure : bool, default=False
        In MOABB, all classification tasks are performed as binary classification problems for P300 datasets.
        If you want to perform 4-class classification for each trial, set ``keep_trial_structure=True``.

        Note that this is only compatible with the :meth:`base.BaseDataset.get_data` method.
        It cannot be used with :class:`moabb.paradigms.base.BaseParadigm` or :class:`moabb.paradigms.P300`.
        To make it compatible with these, you need to provide an appropriate ``process_pipelines`` and ``postprocess_pipeline``
        argument to the :meth:`moabb.paradigms.base.BaseProcessing.get_data`,
        :meth:`moabb.evaluations.base.BaseEvaluation.evaluate` or
        :meth:`moabb.evaluations.base.BaseEvaluation.process` etc...

        .. note::
            Using ``keep_trial_structure=True`` requires the external package
            :mod:`tag-mne` (available on PyPI).

    References
    ----------

    .. [1] Kojima, S. (2024).
        Replication Data for: Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/1UJDV6
    .. [2] Kojima, S. & Kanoh, S. (2024).
        Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Frontiers in Human Neuroscience 18:1461960. DOI: https://doi.org/10.3389/fnhum.2024.1461960
    """

    convert_subject_to_subject_id = _Kojima2024BBase.convert_subject_to_subject_id

    def __init__(self, keep_trial_structure=False):
        super().__init__(
            task="2stream",
            keep_trial_structure=keep_trial_structure,
        )


class Kojima2024B_4stream(_Kojima2024BBase):
    """Class for Kojima2024B_2stream dataset management. P300 dataset.

    **Dataset description**

    This dataset [1]_ originates from a study investigating a four-class auditory BCI
    based on auditory stream segregation (ASME-BCI) [2]_.

    In the experiment, participants focused on one of four auditory streams, leveraging
    auditory stream segregation to selectively attend to stimuli in the target stream.
    Each stream contained a two-stimulus oddball sequence composed of one deviant
    stimulus and one standard stimulus.

    The current class corresponds to the "ASME-4stream" condition described in [2]_.
    For the "ASME-2stream" condition, see :class:`Kojima2024B_2stream`.

    The sequence below illustrates an example trial. For instance, when D3 is the target
    stimulus, the participant attended to Stream3 and selectively listened for D3.
    In this case, D3 is the target, and D1, D2, and D4 are considered non-target stimuli.

    .. code-block:: text

        Stream4  -------- S4 -------- S4 -------- D4 -------- S4 -------- S4 --
        Stream3  ----- S3 -------- S3 -------- S3 -------- D3 -------- S3 -----
        Stream2  -- S2 -------- S2 -------- D2 -------- S2 -------- S2 --------
        Stream1  S1 -------- D1 -------- S1 -------- S1 -------- S1 -----------

    Each participant completed 1 session consisting of 6 runs.
    Each run included 4 trials, each with a different target stimulus.
    In each trial, all deviant stimuli (D1--D4) were presented 15 times.

    Recording Detailes:
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

    Parameters
    ----------

    keep_trial_structure : bool, default=False
        In MOABB, all classification tasks are performed as binary classification problems for P300 datasets.
        If you want to perform 4-class classification for each trial, set ``keep_trial_structure=True``.

        Note that this is only compatible with the :meth:`base.BaseDataset.get_data` method.
        It cannot be used with :class:`moabb.paradigms.base.BaseParadigm` or :class:`moabb.paradigms.P300`.
        To make it compatible with these, you need to provide an appropriate ``process_pipelines`` and ``postprocess_pipeline``
        argument to the :meth:`moabb.paradigms.base.BaseProcessing.get_data`,
        :meth:`moabb.evaluations.base.BaseEvaluation.evaluate` or
        :meth:`moabb.evaluations.base.BaseEvaluation.process` etc...

        .. note::
            Using ``keep_trial_structure=True`` requires the external package
            :mod:`tag-mne` (available on PyPI).

    References
    ----------

    .. [1] Kojima, S. (2024).
        Replication Data for: Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/1UJDV6
    .. [2] Kojima, S. & Kanoh, S. (2024).
        Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Frontiers in Human Neuroscience 18:1461960. DOI: https://doi.org/10.3389/fnhum.2024.1461960
    """

    convert_subject_to_subject_id = _Kojima2024BBase.convert_subject_to_subject_id

    def __init__(self, keep_trial_structure=False):
        super().__init__(
            task="4stream",
            keep_trial_structure=keep_trial_structure,
        )
