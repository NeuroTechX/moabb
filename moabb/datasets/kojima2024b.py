import json
import os
import re
import string
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)

from .base import BaseDataset


_manifest_link = "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi%3A10.7910/DVN/1UJDV6"
_api_base_url = "https://dataverse.harvard.edu/api/access/datafile/"

# The description of the events is available in https://dataverse.harvard.edu/file.xhtml?fileId=10649473&version=1.0.
# This dataset contains the data of an oddball experiment where
# S1-S4 are the standard stimuli and D1-D4 denotes the deviant stimuli.
# During one sequence, the user is instructed to look at one of the deviant stimulus (D1-D4),
# then called Target.
# By opposition, the other deviant stimuli are called NonTarget.
EVENTS = {
    "D1": [101, 111],
    "D2": [102, 112],
    "D3": [103, 113],
    "D4": [104, 114],
    "S1": [1],
    "S2": [2],
    "S3": [3],
    "S4": [4],
    "Deviant": [101, 111, 102, 112, 103, 113, 104, 114],
    "Standard": [1, 2, 3, 4],
    "Target": [111, 112, 113, 114],
    "NonTarget": [101, 102, 103, 104],
}


def _extract_run_number(path):
    match = re.search(r"run-(\d+)", path.name)
    return int(match.group(1)) if match else -1


class Kojima2024B(BaseDataset):
    """Class for Kojima2024B dataset management. P300 dataset.

    **Dataset description**

    This dataset [1]_ originates from a study investigating a four-class auditory BCI
    based on auditory stream segregation (ASME-BCI) [2]_.

    In the experiment, participants focused on auditory streams, leveraging
    auditory stream segregation to selectively attend to stimuli in the target stream.
    The dataset includes both 2-stream and 4-stream conditions:

    **4-stream condition:**
    Participants focused on one of four auditory streams. Each stream contained
    a two-stimulus oddball sequence composed of one deviant stimulus and one standard stimulus.

    The sequence below illustrates an example trial. For instance, when D3 is the target
    stimulus, the participant attended to Stream3 and selectively listened for D3.
    In this case, D3 is the target, and D1, D2, and D4 are considered non-target stimuli.

    .. code-block:: text

        Stream4  -------- S4 -------- S4 -------- D4 -------- S4 -------- S4 --
        Stream3  ----- S3 -------- S3 -------- S3 -------- D3 -------- S3 -----
        Stream2  -- S2 -------- S2 -------- D2 -------- S2 -------- S2 --------
        Stream1  S1 -------- D1 -------- S1 -------- S1 -------- S1 -----------

    **2-stream condition:**
    Participants focused on one of two auditory streams. Each stream contained
    a three-stimulus oddball sequence composed of two deviant stimuli and one standard stimulus.

    The sequence below illustrates an example trial. For instance, when D4 is the target
    stimulus, the participant attended to Stream2 and selectively listened for D4.
    In this case, D4 is the target, and D1, D2, and D3 are considered non-target stimuli.

    .. code-block:: text

        Stream2  -- S2 --- D3 --- S2 --- S2 --- S2 --- S2 --- D4 ---
        Stream1  S1 --- S1 --- D1 --- S1 --- S1 --- D2 --- S1 --- S1

    Each participant completed 1 session consisting of 6 runs.
    Each run included 4 trials, each with a different target stimulus.
    In each trial, all deviant stimuli (D1--D4) were presented 15 times.

    Notes
    -----
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
    events : dict
        Event mapping for the dataset.
    task : {"all", "2stream", "4stream"}, optional
            Which task condition to include:

            - ``"all"``: load both 2-stream and 4-stream conditions (default).
            - ``"2stream"``: load only the 2-stream condition.
            - ``"4stream"``: load only the 4-stream condition.

            For each task condition, the total number of trials per class is:

        - ``"2stream"``: 1080 NT / 360 T
        - ``"4stream"``: 1080 NT / 360 T

    References
    ----------

    .. [1] Kojima, S. (2024).
        Replication Data for: Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Harvard Dataverse, V1. DOI: https://doi.org/10.7910/DVN/1UJDV6
    .. [2] Kojima, S. & Kanoh, S. (2024).
        Four-class ASME BCI: investigation of the feasibility and comparison of two strategies for multiclassing.
        Frontiers in Human Neuroscience 18:1461960. DOI: https://doi.org/10.3389/fnhum.2024.1461960
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64, "eog": 2},
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "AFz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPz",
                "Cz",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FT10",
                "FT7",
                "FT8",
                "FT9",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP10",
                "TP7",
                "TP8",
                "TP9",
                "hEOG",
                "vEOG",
            ],
            sensor_type="EEG",
            reference="right mastoid",
            ground="left mastoid",
            hardware="BrainAmp",
            software=None,
            filters=None,
            line_freq=50.0,
            montage="standard_1020",
            impedance_threshold_kohm=None,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["vertical", "horizontal"],
                has_emg=False,
                emg_channels=0,
                other_physiological=None,
            ),
            cap_manufacturer="EasyCap",
            cap_model=None,
            electrode_type="passive Ag/AgCl",
            electrode_material="Ag/AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            gender={"male": 13, "female": 2},
            age_mean=22.8,
            age_std=None,
            age_min=21.0,
            age_max=24.0,
            ages=None,
            handedness=None,
            clinical_population=None,
            bci_experience=None,
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="auditory stream segregation with oddball",
            events={
                "ASME-4stream_S1": 1,
                "ASME-4stream_D1": 1,
                "ASME-4stream_S2": 1,
                "ASME-4stream_D2": 1,
                "ASME-4stream_S3": 1,
                "ASME-4stream_D3": 1,
                "ASME-4stream_S4": 1,
                "ASME-4stream_D4": 1,
                "ASME-2stream_S1": 1,
                "ASME-2stream_D1": 1,
                "ASME-2stream_D2": 1,
                "ASME-2stream_S2": 1,
                "ASME-2stream_D3": 1,
                "ASME-2stream_D4": 1,
            },
            n_classes=4,
            class_labels=["D1", "D2", "D3", "D4"],
            trials_per_class=None,
            trial_duration=90.0,
            tasks=["ASME-4stream", "ASME-2stream"],
            study_design="within-subject comparison",
            study_domain="auditory BCI",
            feedback_type="none",
            stimulus_type="auditory tones",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="focus selectively on deviant stimuli in one of the streams and count target deviant stimuli",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation=None,
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2024.1461960",
            description="Four-class ASME BCI investigation comparing two strategies for multiclassing: ASME-4stream (four streams with single target stimulus each) vs ASME-2stream (two streams with two target stimuli each)",
            investigators=["Simon Kojima", "Shin'ichiro Kanoh"],
            institution="Shibaura Institute of Technology",
            country="JP",
            repository=None,
            data_url="https://doi.org/10.7910/DVN/1UJDV6",
            license="CC BY",
            publication_year=2024,
            senior_author="Shin'ichiro Kanoh",
            contact_info=["simon.kojima@ieee.org"],
            associated_paper_doi="10.3389/fnhum.2024.1461960",
            funding=["JSPS KAKENHI (Grant Number JP23K11811 to Shin'ichiro Kanoh)"],
            institution_address="Tokyo, Japan",
            institution_department="Graduate School of Engineering and Science (Simon Kojima); College of Engineering (Shin'ichiro Kanoh)",
            ethics_approval=[
                "Review Board on Bioengineering Research Ethics of the Shibaura Institute of Technology"
            ],
            acknowledgements=None,
            how_to_acknowledge=None,
            keywords=[
                "brain-computer interface",
                "electroencephalogram",
                "event-related potential",
                "auditory scene analysis",
                "stream segregation",
                "machine learning",
                "NASA-TLX",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=12,
        sessions=None,
        contributing_labs=None,
        n_contributing_labs=1,
        data_processed=False,
        file_format="BrainVision",
        external_links=None,
        tags=Tags(pathology=["Healthy"], modality=["auditory"], type=["ERP", "P300"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            preprocessing_steps=None,
            highpass_hz=None,
            lowpass_hz=None,
            bandpass=None,
            notch_hz=None,
            filter_type=None,
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            downsampled_to_hz=None,
            epoch_window=None,
            notes=None,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Linear Discriminant Analysis (LDA)", "shrinkage-LDA"],
            feature_extraction=[
                "mean amplitudes in 10 intervals (0.1s non-overlapping, 0-1.0s)"
            ],
            frequency_bands={
                "analyzed_range": [0.1, 8.0],
            },
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="3-fold chronological cross-validation (BCI simulation); 4-fold chronological cross-validation (binary classification)",
            cv_folds=None,
            evaluation_type=["offline simulation"],
        ),
        performance={
            "ASME-4stream_accuracy": 0.83,
            "ASME-2stream_accuracy": 0.86,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type=None,
            code_length=None,
            n_targets=4,
            n_repetitions=15,
            isi_ms=None,
            soa_ms={
                "ASME-4stream_overall": 150.0,
                "ASME-2stream_overall": 300.0,
                "within_stream": 600.0,
            },
            imagery_tasks=None,
            cue_duration_s=None,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials={
                "ASME-4stream": "600 stimuli per trial (4 trials per run, 6 runs)",
                "ASME-2stream": "300 stimuli per trial (4 trials per run, 6 runs)",
            },
            n_trials_per_class=None,
            n_blocks=12,
            block_duration_s=90.0,
            trials_context="12 runs alternating between ASME-4stream and ASME-2stream, 4 trials per run",
        ),
        abstract="The ASME (Auditory Stream segregation Multiclass ERP) paradigm is used for an auditory brain-computer interface (BCI). Two approaches for achieving four-class ASME were investigated: ASME-4stream (four streams with a single target stimulus each) and ASME-2stream (two streams with two target stimuli each). Fifteen healthy subjects participated. ERPs were analyzed, and binary classification and BCI simulation were conducted offline using linear discriminant analysis. Average accuracies were 0.83 (ASME-4stream) and 0.86 (ASME-2stream). The ASME-2stream paradigm showed shorter latency and larger amplitude of P300, higher binary classification accuracy, and smaller workload. Both paradigms achieved sufficiently high accuracy (over 80%) for practical auditory BCI.",
        methodology="Subjects performed 12 runs alternating between ASME-4stream and ASME-2stream paradigms. Each run contained 4 trials with ~90s duration. ASME-4stream presented 4 streams (SOA=0.15s, 600 stimuli/trial, ratio 9:1 standard:deviant). ASME-2stream presented 2 streams with 2 deviant stimuli each (SOA=0.3s, 300 stimuli/trial, ratio 8:1:1). EEG recorded at 1000 Hz from 64 channels. EOG artifacts removed using ICA on 15 PCs. Data filtered (1-40 Hz for ERP, 0.1-8 Hz for classification), epoched (-0.1 to 1.2s), downsampled to 250 Hz. Classification used shrinkage-LDA with mean amplitudes from 10 intervals (0-1.0s) as features. Performance evaluated using 4-fold chronological cross-validation. Usability assessed via NASA-TLX questionnaire.",
    )

    def __init__(
        self,
        events={"Target": EVENTS["Target"], "NonTarget": EVENTS["NonTarget"]},
        task="all",
        subjects=None,
        sessions=None,
    ):
        self.n_channels = 64

        if task == "all":
            self.tasks = ["2stream", "4stream"]
        elif task == "2stream":
            self.tasks = ["2stream"]
        elif task == "4stream":
            self.tasks = ["4stream"]
        else:
            raise ValueError(
                f"Task '{task}' is not implemented. "
                "Please choose from {'all', '2stream', '4stream'}."
            )

        super().__init__(
            list(range(1, 16)),
            sessions_per_subject=1,
            events=events,
            code="Kojima2024B",
            interval=[-0.5, 1.2],
            paradigm="p300",
            doi="10.7910/DVN/1UJDV6",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            raise TypeError("Type of subjects must be either int or list.")

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

            for task in self.tasks:

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
