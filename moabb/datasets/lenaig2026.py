"""An enhanced experimental paradigm for auditory BCIs by addressing both acoustic and human factors.

Lenaïg Guého, et al. (2026), IEEE 14th International Conference on Brain-Computer Interface (BCI)
DOI: 10.1109/BCI69045.2026.11435072
Data DOI: 10.5281/zenodo.21156618
"""

import logging
from pathlib import Path

import mne
import numpy as np

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)
from .utils import extract_rar


log = logging.getLogger(__name__)

_DOI = "10.1109/BCI69045.2026.11435072"
_DATA_DOI = "10.5281/zenodo.21156618"
_SIGN = "Lenaig2026"

_ZENODO_RECORD = "21156618"
_ZENODO_BASE = f"https://zenodo.org/records/{_ZENODO_RECORD}/files"

# Event codes for experiment triggers
_EVENT_DICT = {
    "ExperimentStart": 1,
    "ExperimentStop": 2,
    "TrialStop": 3,
    "Sinus": 4,
    "BrownNoise": 5,
    "Cicada": 6,
    "Cat": 7,
    "Silence": 8,
}

_CH_TYPES = {
    "Channel 1": "eeg",
    "Channel 2": "eeg",
    "Channel 3": "eeg",
    "Channel 4": "eeg",
    "Channel 5": "eeg",
    "Channel 6": "eeg",
    "Channel 7": "eeg",
    "Channel 8": "eeg",
    "Channel 9": "eeg",
    "Channel 10": "eeg",
    "Channel 11": "eeg",
    "Channel 12": "eeg",
    "Channel 13": "eeg",
    "Channel 14": "eeg",
    "Channel 15": "eeg",
    "Channel 16": "eeg",
    "Channel 17": "eeg",
    "Channel 18": "eeg",
    "Channel 19": "eeg",
    "Channel 20": "eeg",
    "Channel 21": "eeg",
    "Channel 22": "eeg",
    "Channel 23": "eeg",
    "Channel 24": "eeg",
    "Gyro 1": "misc",
    "Gyro 2": "misc",
    "Gyro 3": "misc",
}

# Mapping of channel names to standard EEG names
_CH_NAMES_EEG = {
    "Channel 1": "Fp1",
    "Channel 2": "Fp2",
    "Channel 3": "F3",
    "Channel 4": "F4",
    "Channel 5": "C3",
    "Channel 6": "C4",
    "Channel 7": "P3",
    "Channel 8": "P4",
    "Channel 9": "O1",
    "Channel 10": "O2",
    "Channel 11": "F7",
    "Channel 12": "F8",
    "Channel 13": "T7",
    "Channel 14": "T8",
    "Channel 15": "P7",
    "Channel 16": "P8",
    "Channel 17": "Fz",
    "Channel 18": "Cz",
    "Channel 19": "Pz",
    "Channel 20": "M1",
    "Channel 21": "M2",
    "Channel 22": "AFz",
    "Channel 23": "CPz",
    "Channel 24": "POz",
}

# Subject indices for each experiment
SUBJECTS_EXP1 = list(range(1, 25))
SUBJECTS_EXP2 = list(range(25, 49))
# Valid run options
VALID_RUNS = [1, 2, "both"]
# Default time interval for epochs
DEFAULT_INTERVAL = [1, 9]
# Default paradigm type
DEFAULT_PARADIGM = "ssvep"

# Default dataset code and session
DEFAULT_CODE = "Lenaig2026"
DEFAULT_SESSION = "0"
# Default events always include 'Silence'
DEFAULT_EVENTS = {"Silence": 8}
# Default data directory
DATA_DIR = Path("../data_zenodo")


class Lenaig2026(BaseDataset):
    """
    SSAEP BCI dataset from Guého et al. (2026).

    Dataset from the paper [1]_.

    **Dataset Description**

    This study investigates the EEG activity of 48 participants in response to a set of four auditory stimuli: a pure tone (used as a reference), cicada song and cat's purr (natural sounds), and brownian noise (an artificial signal with a spectral content similar to that of certain natural sounds, such as water noise). In addition, a silent condition, corresponding to the absence of auditory stimulation, was included for subsequent analyses. The stimuli are amplitude-modulated by a 40 Hz sinusoid (modulation index = 1) and have a duration of 10 seconds. The experiment is conducted at two loudness levels (60 and 66 phons, diotic presentation), with 24 participants each.

    EEG acquisition is performed using a 24-channel mBrainTrain headset (international 10-20 system, passive electrodes, impedance maintained below 10 kΩ), coupled with a Smarting module, at a sampling rate of 500 Hz.

    The measurement consists in one session of two 10-minute runs (separated by a 5-minute break), each including 50 trials (10 repetitions per condition), spaced by intervals ranging from 6 to 10 seconds. Stimuli are presented in a pseudo-random order to avoid immediate repetitions of the same stimulus and to prevent a stimulus ending a sequence of five trials from starting the next sequence.

    Parameters
    ----------
    exp : int, optional
        Experiment number (1 or 2). Default is 1.
    run : int, str, optional
        Which run(s) to use (1, 2, or 'both'). Default is 'both'.
    stim : str, optional
        Stimulus type ('Sinus', 'BrownNoise', 'Cat', 'Cicada'). Default is 'Sinus'.

    References
    ----------
    .. [1] L. Guého, L. Bougrain, C. Plapous, P. Hénaff and R. Nicol, "An enhanced experimental paradigm for auditory BCIs by addressing both acoustic and human factors," 2026 14th International Conference on Brain-Computer Interface (BCI), Gangwon Province, Korea, Republic of, 2026, pp. 1-7, doi: 10.1109/BCI69045.2026.11435072.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500,
            n_channels=28,
            channel_types={"eeg": 24, "misc": 3},
            sensor_type="EEG",
            electrode_material="Ag/AgCl",
            electrode_type="Wet - Passive",
            software="OpenVIBE",
            line_freq=50,
            montage="standard_1020",
            impedance_threshold_kohm=10,
        ),
        participants=ParticipantMetadata(
            n_subjects=48, gender={"male": 24, "female": 24}, age_max=63, age_min=18
        ),
        experiment=ExperimentMetadata(
            events=_EVENT_DICT,
            paradigm="ssvep",
            n_classes=2,
            class_labels=["Stimulus", "Silence"],
            trials_per_class=10,
            trial_duration=10,
            feedback_type="auditory",
            mode="offline",
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            license="CC-BY-NC-ND-4.0",
            country="FR",
            institution="Orange Labs",
            description="An enhanced experimental paradigm for auditory BCIs by addressing both acoustic and human factors.",
            investigators=[
                "Lenaïg Guého",
                "Laurent Bougrain",
                "Cyril Plapous",
                "Patrick Hénaff",
                "Rozenn Nicol",
            ],
            publication_year=2026,
        ),
    )

    def __init__(self, exp=1, run="both", stim="Sinus"):
        """
        Initialize the Lenaig2026 dataset object.

        Parameters
        ----------
        exp : int, optional
            Experiment number (1 or 2). Default is 1.
        run : int, str, optional
            Which run(s) to use (1, 2, or 'both'). Default is 'both'.
        stim : str, optional
            Stimulus type ('Sinus', 'BrownNoise', 'Cat', 'Cicada'). Default is 'Sinus'.

        Raises
        ------
        ValueError
            If an invalid experiment or run is provided.
        """
        self.exp = exp
        self.stim = stim

        # Select subject list based on experiment
        if exp == 1:
            self.subject_list = list(SUBJECTS_EXP1)
        elif exp == 2:
            self.subject_list = list(SUBJECTS_EXP2)
        else:
            raise ValueError(f"Invalid experiment: {exp}. Use 1 or 2.")

        # Validate and set runs
        if run not in VALID_RUNS:
            raise ValueError(f"Invalid run: {run}. Use 1, 2, or 'both'.")
        self.runs = [1, 2] if run == "both" else [int(run)]

        # Map stimulus to event code, always include 'Silence'
        self.event_stim = {self.stim: _EVENT_DICT[self.stim], **DEFAULT_EVENTS}

        # Call parent constructor with experiment/session/event info
        super().__init__(
            subjects=self.subject_list,
            sessions_per_subject=1,
            events=self.event_stim,
            code=DEFAULT_CODE,
            interval=DEFAULT_INTERVAL,
            paradigm=DEFAULT_PARADIGM,
            doi=_DOI,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """
        Get the list of file paths for a given subject and selected runs.

        Parameters
        ----------
        subject : int
            Subject number.
        path : str or Path, optional
            Base directory for data. If None, uses DATA_DIR.
        force_update : bool, optional
            Redownload the data.
        update_path : str, optional
            Unused, for compatibility.
        verbose : bool, optional
            Unused, for compatibility.

        Returns
        -------
        list of Path
            List of file paths for the subject and selected runs.

        Raises
        ------
        ValueError
            If the subject is not in the subject list.
        FileNotFoundError
            If any of the required data files are missing.
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        base_path = (
            Path(path)
            if path
            else Path(dl.get_dataset_path(_SIGN, None)) / f"MNE-{_SIGN}-data"
        )
        paths = [
            base_path
            / "EEG_24Chan_AudioStim"
            / f"EXP{self.exp}/S1/R{run}/{subject:02}_R{run}.gdf"
            for run in self.runs
        ]

        if all(p.exists() for p in paths) and not force_update:
            return (
                paths  # Return existing paths if all files exist and no update is forced
            )

        url = f"{_ZENODO_BASE}/EEG_24Chan_AudioStim.rar"
        rar_path = Path(dl.data_dl(url, sign=_SIGN, path=base_path, verbose=verbose))
        extract_rar(rar_path, dest_dir=base_path)

        if not all(p.exists() for p in paths) or force_update:
            raise FileNotFoundError("Some data files are missing.")

        return paths

    def _get_single_subject_data(self, subject):
        """
        Load and preprocess raw EEG data for a single subject.

        Parameters
        ----------
        subject : int
            Subject ID.

        Returns
        -------
        dict
            Session dictionary compatible with MOABB, containing preprocessed MNE Raw objects for each run.
        """
        file_paths = self.data_path(subject)
        session = {DEFAULT_SESSION: {}}

        for idx, filepath in enumerate(file_paths):
            # Load raw GDF file
            raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)

            # Rename channels to standard names
            raw.set_channel_types(_CH_TYPES, on_unit_change="ignore", verbose=False)
            raw.rename_channels(_CH_NAMES_EEG, verbose=False)

            # Set standard 10-20 montage
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, verbose=0)

            # Extract events from annotations
            events, _ = mne.events_from_annotations(raw, verbose=False)

            # Create stimulus channel
            stim_data = np.zeros(raw.n_times, dtype=int)
            stim_data[events[:, 0]] = events[:, 2]
            stim_info = mne.create_info(
                ["STI 014"], raw.info["sfreq"], ["stim"], verbose=False
            )
            stim_raw = mne.io.RawArray(stim_data[np.newaxis, :], stim_info, verbose=False)
            raw.add_channels([stim_raw])

            # Store the run in the session dictionary
            session[DEFAULT_SESSION][str(self.runs[idx])] = raw

        return session
