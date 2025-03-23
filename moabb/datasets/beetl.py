import logging
import os
import zipfile
from pathlib import Path

import mne
import numpy as np
import pooch

import moabb.datasets.download as dl

from .base import BaseDataset
from .download import get_dataset_path


LOGGER = logging.getLogger(__name__)
BASE_URL = "https://ndownloader.figshare.com/files/"

LEADERBOARD_ARTICLE_ID = 14839650
FINAL_EVALUATION_ARTICLE_ID = 16586213
FINAL_LABEL_TXT_ARTICLE_ID = 21602622


class Beetl2021_A(BaseDataset):
    """Motor Imagery dataset from BEETL Competition - Dataset A.

    **Dataset description**

    Dataset A contains data from subjects with 500 Hz sampling rate and 63 EEG channels.
    In the leaderboard phase, this includes subjects 1-2, while in the final phase it includes
    subjects 1-3.

    Note: for the BEETL competition, there was a leaderboard phase and a final phase. Both phases
    contained data from two datasets, A and B. However, during leaderboard phase, dataset A contained
    data from subjects 1-2, while dataset B contained data from subjects 3-5. During the final phase,
    dataset A contained data from subjects 1-3, while dataset B contained data from subjects 4-5.

    Note: for the competition the data is cut into 4 second trials, here the data is concatenated
    into one session! In order to get the data as provided in the competition, the data has to be
    cut into 4 second trials.

    For the leaderboard phase, the dataset contains only training data, while for the final phase it
    includes both training and testing data. To learn more about the datasets in detail see [1]_.
    To learn more about the competition see [2]_.

    For benchmarking the BEETL competition use phase "final", train on training data benchmark on testing data.

    Data is sampled at 500 Hz and contains 63 EEG channels. The data underwent frequency-domain preprocessing
    using a bandpass filter (1-100 Hz) and a 50 Hz notch filter to attenuate power line interference.

    Motor imagery tasks include:
    - Rest (label 0)
    - Left hand (label 1)
    - Right hand (label 2)
    - Feet (label 3)

    Attributes
    ----------
    phase : str
        Either "leaderboard" or "final"

    References
    ----------
    .. [1] Wei, X., Faisal, A. A., Grosse-Wentrup, M., Gramfort, A.,
        Chevallier, S., Jayaram, V., ... & Tempczyk, P. (2022, July). 2021
        BEETL competition: Advancing transfer learning for subject independence
        and heterogeneous EEG data sets. In NeurIPS 2021 Competitions and
        Demonstrations Track (pp. 205-219). PMLR.
    .. [2] Competition: https://beetl.ai/introduction
    """

    def __init__(self, phase="final"):
        """Initialize BEETL Dataset A.

        Parameters
        ----------
        phase : str
            Either "leaderboard" (subjects 1-2) or "final" (subjects 1-3)
        """
        if phase not in ["leaderboard", "final"]:
            raise ValueError("Phase must be either 'leaderboard' or 'final'")

        self.phase = phase
        subjects = list(range(1, 3)) if phase == "leaderboard" else list(range(1, 4))

        # Channel setup
        self.ch_names = [
            "Fp1",
            "Fz",
            "F3",
            "F7",
            "FT9",
            "FC5",
            "FC1",
            "C3",
            "T7",
            "TP9",
            "CP5",
            "CP1",
            "Pz",
            "P3",
            "P7",
            "O1",
            "Oz",
            "O2",
            "P4",
            "P8",
            "TP10",
            "CP6",
            "CP2",
            "C4",
            "T8",
            "FT10",
            "FC6",
            "FC2",
            "F4",
            "F8",
            "Fp2",
            "AF7",
            "AF3",
            "AFz",
            "F1",
            "F5",
            "FT7",
            "FC3",
            "FCz",
            "C1",
            "C5",
            "TP7",
            "CP3",
            "P1",
            "P5",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "P6",
            "P2",
            "CPz",
            "CP4",
            "TP8",
            "C6",
            "C2",
            "FC4",
            "FT8",
            "F6",
            "F2",
            "AF4",
            "AF8",
        ]

        self.sfreq = 500
        self.phase = phase

        super().__init__(
            subjects=subjects,
            sessions_per_subject=1,  # Data is concatenated into one session
            events=dict(rest=0, left_hand=1, right_hand=2, feet=3),
            code="Beetl2021-A",
            interval=[0, 4],  # 4s trial window
            paradigm="imagery",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)

        # Create MNE info
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=["eeg"] * len(self.ch_names),
        )

        phase_str = "leaderboardMI" if self.phase == "leaderboard" else "finalMI"
        subject_dir = Path(file_paths[0]) / phase_str / phase_str / f"S{subject}"

        data_list = []
        labels_list = []

        # Load training data
        for race in range(1, 6):
            data_file = subject_dir / "training" / f"race{race}_padsData.npy"
            label_file = subject_dir / "training" / f"race{race}_padsLabel.npy"
            if data_file.exists() and label_file.exists():
                data_list.append(np.load(data_file, allow_pickle=True))
                labels_list.append(np.load(label_file, allow_pickle=True))

        data = np.concatenate(data_list)
        labels = np.concatenate(labels_list)

        # Create events array
        events = np.column_stack(
            (
                np.arange(0, len(labels) * data.shape[-1], data.shape[-1]),
                np.zeros(len(labels), dtype=int),
                labels,
            )
        )

        # Create Raw object
        event_desc = {int(code): name for name, code in self.event_id.items()}
        raw = mne.io.RawArray(np.hstack(data), info)
        raw.set_annotations(
            mne.annotations_from_events(
                events=events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        # Load test data
        test_data_list = []
        for race in range(6, 16):
            data_file = subject_dir / "testing" / f"race{race}_padsData.npy"
            if data_file.exists():
                test_data_list.append(np.load(data_file, allow_pickle=True))

        test_data = np.concatenate(test_data_list)

        # load labels from .txt
        test_labels = np.loadtxt(Path(file_paths[0]) / "final_MI_label.txt", dtype=int)
        subject_labels = test_labels[
            (subject - 1) * test_data.shape[0] : (subject) * test_data.shape[0]
        ]

        test_events = np.column_stack(
            (
                np.arange(
                    0, len(subject_labels) * test_data.shape[-1], test_data.shape[-1]
                ),
                np.zeros(len(subject_labels), dtype=int),
                subject_labels,
            )
        )

        # Create Raw object
        test_raw = mne.io.RawArray(np.hstack(test_data), info)
        test_raw.set_annotations(
            mne.annotations_from_events(
                events=test_events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        return {"0": {f"0{self.phase}train": raw, f"1{self.phase}test": test_raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return path to the data files."""
        if subject not in self.subject_list:
            raise ValueError(f"Subject {subject} not in {self.subject_list}")

        path = get_dataset_path("BEETL", path)
        base_path = Path(os.path.join(path, f"MNE-{self.code:s}-data"))
        # Create the directory if it doesn't exist
        base_path.mkdir(parents=True, exist_ok=True)

        # Download data if needed
        for article_id in [LEADERBOARD_ARTICLE_ID, FINAL_EVALUATION_ARTICLE_ID]:
            file_list = dl.fs_get_file_list(article_id)
            hash_file_list = dl.fs_get_file_hash(file_list)
            id_file_list = dl.fs_get_file_id(file_list)

            for file_name in id_file_list.keys():
                file_path = os.path.join(base_path, file_name)
                extract_dir = os.path.join(base_path, os.path.splitext(file_name)[0])

                # Step 1: Download the zip file if not already downloaded
                if not os.path.exists(file_path):
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

                # Step 2: Unzip the file if not already extracted
                if not os.path.exists(extract_dir):
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)

        # Download labels for final phase
        file_list = dl.fs_get_file_list(FINAL_LABEL_TXT_ARTICLE_ID)
        hash_file_list = dl.fs_get_file_hash(file_list)
        id_file_list = dl.fs_get_file_id(file_list)

        for file_name in id_file_list.keys():
            fpath = base_path / file_name
            if (not fpath.exists() or force_update) and file_name == "final_MI_label.txt":
                fpath = base_path / file_name
                if not fpath.exists() or force_update:
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

        return [str(base_path)]


class Beetl2021_B(BaseDataset):
    """Motor Imagery dataset from BEETL Competition - Dataset B.

    **Dataset description**

    Dataset B contains data from subjects with 200 Hz sampling rate and 32 EEG channels.
    In the leaderboard phase, this includes subjects 3-5, while in the final phase it includes
    subjects 4-5.

    Note: for the BEETL competition, there was a leaderboard phase and a final phase. Both phases
    contained data from two datasets, A and B. However, during leaderboard phase, dataset A contained
    data from subjects 1-2, while dataset B contained data from subjects 3-5. During the final phase,
    dataset A contained data from subjects 1-3, while dataset B contained data from subjects 4-5.

    Note: for the competition the data is cut into 4 second trials, here the data is concatenated
    into one session! In order to get the data as provided in the competition, the data has to be
    cut into 4 second trials.

    For the leaderboard phase, the dataset contains only training data, while for the final phase it
    includes both training and testing data. To learn more about the datasets in detail see [1]_.
    To learn more about the competition see [2]_.

    For benchmarking the BEETL competition use phase "final", train on training data benchmark on testing data.

    The data was filtered using a highpass filter with a cutoff frequency of 1 Hz and a
    lowpass filter with a cutoff frequency of 100 Hz.

    Motor imagery tasks include:
    - Left hand (label 0)
    - Right hand (label 1)
    - Feet (label 2)
    - Rest (label 3)

    Attributes
    ----------
    phase : str
        Either "leaderboard" or "final"

    References
    ----------
    .. [1] Wei, X., Faisal, A. A., Grosse-Wentrup, M., Gramfort, A.,
        Chevallier, S., Jayaram, V., ... & Tempczyk, P. (2022, July). 2021
        BEETL competition: Advancing transfer learning for subject independence
        and heterogeneous EEG data sets. In NeurIPS 2021 Competitions and
        Demonstrations Track (pp. 205-219). PMLR.
    .. [2] Competition: https://beetl.ai/introduction
    """

    def __init__(self, phase="final"):
        """Initialize BEETL Dataset B.

        Parameters
        ----------
        phase : str
            Either "leaderboard" (subjects 3-5) or "final" (subjects 4-5)
        """
        if phase not in ["leaderboard", "final"]:
            raise ValueError("Phase must be either 'leaderboard' or 'final'")

        self.phase = phase
        subjects = list(range(3, 6)) if phase == "leaderboard" else list(range(4, 6))

        super().__init__(
            subjects=subjects,
            sessions_per_subject=1,  # Data is concatenated into one session
            events=dict(left_hand=0, right_hand=1, feet=2, rest=3),
            code="Beetl2021-B",
            interval=[0, 4],  # 4s trial window
            paradigm="imagery",
        )

        self.ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "Fz",
            "F4",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
        ]
        self.sfreq = 200
        self.phase = phase

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)

        # Create MNE info
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=["eeg"] * len(self.ch_names),
        )

        # Load data
        phase_str = "leaderboardMI" if self.phase == "leaderboard" else "finalMI"
        subject_dir = Path(file_paths[0]) / phase_str / phase_str / f"S{subject}"

        # Load training data
        train_data = np.load(
            subject_dir / "training" / f"training_s{subject}X.npy", allow_pickle=True
        )
        train_labels = np.load(
            subject_dir / "training" / f"training_s{subject}y.npy", allow_pickle=True
        )

        # Create events array
        events = np.column_stack(
            (
                np.arange(
                    0, len(train_labels) * train_data.shape[-1], train_data.shape[-1]
                ),
                np.zeros(len(train_labels), dtype=int),
                train_labels,
            )
        )

        # Create Raw object
        event_desc = {int(code): name for name, code in self.event_id.items()}
        raw = mne.io.RawArray(np.hstack(train_data * 1e-6), info)
        raw.set_annotations(
            mne.annotations_from_events(
                events=events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        # Load test data
        test_data = np.load(
            subject_dir / "testing" / f"testing_s{subject}X.npy", allow_pickle=True
        )
        # load labels from .txt
        test_labels = np.loadtxt(Path(file_paths[0]) / "final_MI_label.txt", dtype=int)
        subject_labels = test_labels[
            (subject - 1) * test_data.shape[0] : (subject) * test_data.shape[0]
        ]

        test_events = np.column_stack(
            (
                np.arange(
                    0, len(subject_labels) * test_data.shape[-1], test_data.shape[-1]
                ),
                np.zeros(len(subject_labels), dtype=int),
                subject_labels,
            )
        )

        # Create Raw object
        test_raw = mne.io.RawArray(np.hstack(test_data * 1e-6), info)
        test_raw.set_annotations(
            mne.annotations_from_events(
                events=test_events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        return {"0": {f"0{self.phase}train": raw, f"1{self.phase}test": test_raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return path to the data files."""
        if subject not in self.subject_list:
            raise ValueError(f"Subject {subject} not in {self.subject_list}")

        path = get_dataset_path("BEETL", path)
        base_path = Path(os.path.join(path, f"MNE-{self.code:s}-data"))

        # Create the directory if it doesn't exist
        base_path.mkdir(parents=True, exist_ok=True)

        # Download data if needed
        for article_id in [LEADERBOARD_ARTICLE_ID, FINAL_EVALUATION_ARTICLE_ID]:
            file_list = dl.fs_get_file_list(article_id)
            hash_file_list = dl.fs_get_file_hash(file_list)
            id_file_list = dl.fs_get_file_id(file_list)

            for file_name in id_file_list.keys():
                file_path = os.path.join(base_path, file_name)
                extract_dir = os.path.join(base_path, os.path.splitext(file_name)[0])

                # Step 1: Download the zip file if not already downloaded
                if not os.path.exists(file_path):
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

                # Step 2: Unzip the file if not already extracted
                if not os.path.exists(extract_dir):
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)

        # Download labels for final phase
        file_list = dl.fs_get_file_list(FINAL_LABEL_TXT_ARTICLE_ID)
        hash_file_list = dl.fs_get_file_hash(file_list)
        id_file_list = dl.fs_get_file_id(file_list)

        for file_name in id_file_list.keys():
            fpath = base_path / file_name
            if (not fpath.exists() or force_update) and file_name == "final_MI_label.txt":
                fpath = base_path / file_name
                if not fpath.exists() or force_update:
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

        return [str(base_path)]
