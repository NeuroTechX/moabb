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
        markers.append(f"marker:{event_desc[val]}")

    return samples, markers


def split_trials(markers, n_events_trial, n_trials):
    trials = []
    for idx_trial in range(n_trials):

        idx_start = idx_trial * n_events_trial
        idx_end = idx_start + n_events_trial

        trial = markers[idx_start:idx_end]

        # idx 0 is events for trial start
        del trial[0]

        trials.append(trial)

    return trials


def add_info_to_markers(markers, key, value, append=True):

    if append:
        new_markers = [f"{marker}/{key}:{value}" for marker in markers]
    else:
        new_markers = [f"{key}:{value}/{marker}" for marker in markers]

    return new_markers


def split_sequences(markers_trial, len_sequence, n_sequences):

    new_markers_trial = []
    for idx_sequence in range(n_sequences):
        sequence = idx_sequence + 1

        idx_start = idx_sequence * len_sequence
        idx_end = idx_start + len_sequence

        markers_sequence = markers_trial[idx_start:idx_end]
        markers_sequence = add_info_to_markers(markers_sequence, "sequence", sequence)

        for m in markers_sequence:
            new_markers_trial.append(m)

    return new_markers_trial


def add_events_names(markers, events_names):

    new_markers = []
    for m in markers:
        marker = int(m.split(":")[1])
        if marker in list(events_names.keys()):
            name = events_names[marker]
        else:
            name = "misc"

        new_markers.append(f"{m}/event:{name}")

    return new_markers


def get_marker(marker_desc):
    for m in marker_desc.split("/"):
        if "marker" == m.split(":")[0]:
            return int(m.split(":")[1])
    return None


def add_events_types(markers, events_types):

    new_markers = []
    for m in markers:

        marker = get_marker(m)
        if marker is None:
            raise RuntimeError(f'marker could not be parsed: "{m}"')

        if marker in list(events_types.keys()):
            name = events_types[marker]
        else:
            name = "misc"

        new_markers.append(f"{m}/{name}")

    return new_markers


def convert_marker_to_dict(marker):
    marker_dict = {}
    for m in marker.split("/"):
        marker_dict[m.split(":")[0]] = m.split(":")[1]

    return marker_dict


def reorder_markers(markers, order):

    new_markers = []
    for marker in markers:
        marker_dict = convert_marker_to_dict(marker)

        new_marker = [f"{k}:{marker_dict[k]}" for k in order]

        new_markers.append("/".join(new_marker))

    return new_markers


def add_events_info(markers_trials, len_sequence, n_sequences, events_types, order):

    new_markers_trials = []
    for idx_trial, markers_trial in enumerate(markers_trials):
        trial = idx_trial + 1
        markers_trial = add_info_to_markers(markers_trial, "trial", trial)

        markers_trial = split_sequences(markers_trial, len_sequence, n_sequences)

        markers_trial = reorder_markers(markers_trial, order)

        markers_trial = add_events_types(markers_trial, events_types)

        new_markers_trials.append(markers_trial)

    return new_markers_trials


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
    ):

        self.task = task

        if self.task == "2stream":
            self.len_sequence = 20
        elif self.task == "4stream":
            self.len_sequence = 40
        else:
            raise ValueError(f"Unknown task: {self.task}")

        self.n_sequences = 15
        self.subject_list = list(range(1, 16))
        self.n_channels = 64

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

            runs.update({f"{run}{task}": raw})

        sessions = {"0": runs}

        return sessions

    def _get_single_subject_data_with_events_info(self, subject):
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

            events_names = {
                101: "D1",
                102: "D2",
                103: "D3",
                104: "D4",
                111: "D1",
                112: "D2",
                113: "D3",
                114: "D4",
                1: "S1",
                2: "S2",
                3: "S3",
                4: "S4",
            }

            markers = add_events_names(markers, events_names)

            markers = add_info_to_markers(markers, "task", self.task, False)
            markers = add_info_to_markers(markers, "run", run, False)
            markers = add_info_to_markers(markers, "subject", subject, False)

            markers_trials = split_trials(
                markers=markers,
                n_events_trial=self.len_sequence * self.n_sequences + 1,
                n_trials=4,
            )

            events_types = {
                101: "NonTarget",
                102: "NonTarget",
                103: "NonTarget",
                104: "NonTarget",
                111: "Target",
                112: "Target",
                113: "Target",
                114: "Target",
                1: "Standard",
                2: "Standard",
                3: "Standard",
                4: "Standard",
            }
            markers_trials = add_events_info(
                markers_trials,
                len_sequence=self.len_sequence,
                n_sequences=self.n_sequences,
                events_types=events_types,
                order=["subject", "run", "task", "trial", "sequence", "event", "marker"],
            )

            markers = [m for markers_trial in markers_trials for m in markers_trial]

            events, event_id = events_from_markers(samples, markers)

            event_desc = {v: k for k, v in event_id.items()}

            annotations = mne.annotations_from_events(
                events=events, sfreq=raw.info["sfreq"], event_desc=event_desc
            )

            raw = raw.set_annotations(annotations)

            runs.update({f"{run}{task}": raw})

        sessions = {"0": runs}

        return sessions

    def get_data_with_events_info(self, subjects):
        """
        Retrieve MNE Raw objects with detailed event annotations for the given subjects.

        Parameters
        ----------
        subjects : list of int
            A list of subject numbers to load data for.

        Returns
        -------
        dict
            A dictionary mapping each subject ID to its corresponding
            :class:`mne.io.Raw` instance.

            Each Raw object contains EEG recordings with detailed event
            metadata stored in its ``annotations`` attribute.
            The annotation descriptions encode hierarchical metadata in the
            following format:

            ``subject:<id>/run:<id>/task:<name>/trial:<id>/sequence:<id>/event:<label>/marker:<id>/<Target|NonTarget|Standard>``

            where
              - **subject** : subject ID
              - **run** : run number
              - **task** : experimental task ("2stream" or "4stream")
              - **trial** : trial number within the run
              - **sequence** : stimulus sequence number
              - **event** : stimulus/event label (e.g., "D1", "D2", "D3", "D4", "S1", "S2")
              - **marker** : event marker ID
              - **Target | NonTarget | Standard** : stimulus type classification

            Example
            -------
            >>> dataset = Kojima2024B_2stream
            >>> sessions = dataset.get_data_with_events_info([1])
            >>> raw = sessions[1]["0"]["12stream"]
            >>> raw.annotations[0]["description"]
            subject:1/run:1/task:2stream/trial:1/sequence:1/event:S1/marker:1/Standard
            >>> raw.annotations[3]["description"]
            subject:1/run:1/task:2stream/trial:1/sequence:1/event:D4/marker:114/Target
        """

        if not isinstance(subjects, list):
            raise ValueError("subjects must be a list")

        return {
            subject: self._get_single_subject_data_with_events_info(subject)
            for subject in subjects
        }

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
    get_data_with_events_info = _Kojima2024BBase.get_data_with_events_info

    def __init__(self):
        super().__init__(
            task="2stream",
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
    get_data_with_events_info = _Kojima2024BBase.get_data_with_events_info

    def __init__(self):
        super().__init__(
            task="4stream",
        )
