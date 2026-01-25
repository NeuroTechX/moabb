"""BNCI 2025-002 Continuous 2D Trajectory Decoding dataset."""

from datetime import datetime, timezone

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"


def _data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    """Download data file from URL."""
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


@verbose
def _load_data_002_2025(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 002-2025 dataset (Continuous 2D Trajectory Decoding).

    Parameters
    ----------
    subject : int
        Subject number (1-20).
    path : None | str
        Location of where to look for the BNCI data storing location.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI_PATH in mne-python config.
    base_url : str
        Base URL for the dataset.
    only_filenames : bool
        If True, return only the local path of the files.
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    sessions : dict
        Dictionary containing sessions with raw data for each run.
    """
    if (subject < 1) or (subject > 20):
        raise ValueError("Subject must be between 1 and 20. Got %d." % subject)

    # Subject IDs in the dataset (as listed on BNCI website)
    # fmt: off
    subject_ids = [
        "fe3", "fg4", "fg6", "fh0", "fh4", "fh5", "fh7", "fi3", "fi6", "fj0",
        "fj4", "fj5", "fj7", "fk3", "fk6", "fl0", "fl3", "fl4", "fl6", "fm0",
    ]
    # fmt: on

    subj_id = subject_ids[subject - 1]

    # 60 EEG channels following the 10-10 system
    # Fp1, Fp2, FT9, FT10 were reallocated as EOG (VEOG1, VEOG2, HEOG1, HEOG2)
    # TP9, TP10 were relocated to PPO1h, PPO2h
    # fmt: off
    ch_names_eeg = [
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POz", "PO4", "PO8",
        "O1", "Oz", "O2",
        "PPO1h", "PPO2h",
    ]
    # fmt: on

    ch_names_eog = ["VEOG1", "VEOG2", "HEOG1", "HEOG2"]
    ch_names = ch_names_eeg + ch_names_eog
    ch_types = ["eeg"] * 60 + ["eog"] * 4

    sessions = {}
    filenames = []

    # 3 sessions, each with 3 perception levels
    perception_levels = ["perc0", "perc50", "perc100"]

    for session_idx in range(1, 4):
        session_runs = {}
        for run_idx, perc in enumerate(perception_levels):
            filename_part = f"{subj_id}_ses{session_idx}_{perc}.mat"
            url = f"{base_url}002-2025/{filename_part}"
            filename = _data_path(url, path, force_update, update_path)[0]
            filenames.append(filename)

            if only_filenames:
                continue

            # Load the MAT file
            raw = _convert_run_002_2025(
                filename,
                ch_names,
                ch_types,
                subject_id=subject,
                session_idx=session_idx,
                perception=perc,
            )
            session_runs[f"{run_idx}{perc}"] = raw

        if not only_filenames:
            sessions[f"{session_idx - 1}ses{session_idx}"] = session_runs

    if only_filenames:
        return filenames

    return sessions


@verbose
def _convert_run_002_2025(
    filename,
    ch_names,
    ch_types,
    subject_id=None,
    session_idx=None,
    perception=None,
    verbose=None,
):
    """Convert one run from 002-2025 dataset to MNE Raw object.

    Parameters
    ----------
    filename : str
        Path to MAT file.
    ch_names : list
        Channel names.
    ch_types : list
        Channel types.
    subject_id : int
        Subject number.
    session_idx : int
        Session index.
    perception : str
        Perception level (perc0, perc50, perc100).
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    raw : mne.io.RawArray
        MNE Raw object containing the EEG data.
    """
    # Load the MAT file
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    # The dataset uses LSL format, with EEG, EOG, and movement data
    # Extract EEG data - try different possible keys
    eeg_data = None
    sfreq = 200.0  # Default sampling rate for this dataset

    if "cnt" in data:
        eeg_data = data["cnt"]
        if "fs" in data:
            sfreq = float(data["fs"])
    elif "X" in data:
        eeg_data = data["X"]
        if "fs" in data:
            sfreq = float(data["fs"])
    elif "data" in data:
        run_data = data["data"]
        if hasattr(run_data, "X"):
            eeg_data = run_data.X
            if hasattr(run_data, "fs"):
                sfreq = float(run_data.fs)
        else:
            eeg_data = run_data
    else:
        # Try to find the data in the structure
        for key in data.keys():
            if not key.startswith("__"):
                val = data[key]
                if hasattr(val, "X"):
                    eeg_data = val.X
                    if hasattr(val, "fs"):
                        sfreq = float(val.fs)
                    break
                elif isinstance(val, np.ndarray) and val.ndim == 2:
                    if val.shape[0] >= 60 or val.shape[1] >= 60:
                        eeg_data = val
                        break

    if eeg_data is None:
        raise ValueError(
            f"Could not find EEG data in MAT file. Keys: {list(data.keys())}"
        )

    # Ensure data is in correct shape (n_channels, n_samples)
    # Data should be (n_samples, n_channels) or (n_channels, n_samples)
    if eeg_data.shape[0] > eeg_data.shape[1]:
        # Data is (n_samples, n_channels), transpose to (n_channels, n_samples)
        eeg_data = eeg_data.T

    # Convert to Volts (MNE standard) if data is in microvolts
    if np.abs(eeg_data).max() > 1:  # Likely in microvolts
        eeg_data = eeg_data * 1e-6

    # Check number of channels
    n_channels_data = eeg_data.shape[0]
    n_channels_expected = len(ch_names)

    if n_channels_data != n_channels_expected:
        # If we have more channels, they might include trajectory/marker data
        # Take only the first 64 channels (60 EEG + 4 EOG)
        if n_channels_data > n_channels_expected:
            eeg_data = eeg_data[:n_channels_expected, :]
        else:
            # If we have fewer channels, adjust channel names
            ch_names = ch_names[:n_channels_data]
            ch_types = ch_types[:n_channels_data]

    # Create MNE info structure
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # Create Raw object
    raw = RawArray(data=eeg_data, info=info, verbose=verbose)

    # Set montage for EEG channels
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")

    # Set line frequency (European dataset - 50 Hz)
    raw.info["line_freq"] = 50.0

    # Set measurement date (dataset recorded ~2021-2022)
    raw.set_meas_date(datetime(2022, 1, 1, tzinfo=timezone.utc))

    # Add description
    desc = f"Session {session_idx}, Perception: {perception}"
    raw.info["description"] = desc

    return raw


class BNCI2025_002(BaseDataset):
    """BNCI 2025-002 Continuous 2D Trajectory Decoding dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 20 able-bodied participants
    performing a continuous 2D trajectory decoding task using attempted
    movement. The study investigates continuous decoding of hand movement
    trajectories from EEG signals, with participants tracking a moving target
    on screen while their dominant arm is strapped to restrict actual motor
    output (simulating attempted movement conditions similar to paralyzed
    individuals).

    The experimental paradigm includes both calibration and online decoding
    phases, with varying levels of EEG feedback (0%, 50%, 100%) to evaluate
    the impact of feedback on decoding performance.

    **Participants**

    - 20 able-bodied subjects (10 male, mean age 24 +/- 5 years)
    - All right-handed
    - 4 had prior EEG experience
    - Location: Institute of Neural Engineering, Graz University of
      Technology, Austria

    **Recording Details**

    - Equipment: 64-channel actiCAP system (Brain Products GmbH)
    - Channels: 60 EEG + 4 EOG electrodes
    - Original sampling rate: 200 Hz
    - Electrode positions: 10-10 system with modifications
      (Fp1, Fp2, FT9, FT10 used as EOG; TP9, TP10 relocated to PPO1h, PPO2h)
    - Reference: Common average
    - Data synchronized using Lab Streaming Layer (LSL)

    **Experimental Procedure**

    Each session consists of:

    - Calibration phase: 2 eye runs (38 trials, 8s each) + 4 snake runs
      (48 trials, 23s each)
    - Online phase with 3 perception conditions:
      - perc0: No EEG feedback (baseline)
      - perc50: 50% EEG feedback
      - perc100: 100% EEG feedback

    Trial types:

    - Snake runs: Tracking a moving white target with decorrelated x/y
      coordinates
    - Free runs: Tracing static shapes (diagonal/circle) at self-paced speed

    **Data Organization**

    - 3 sessions per subject (recorded over 5 days)
    - 3 perception levels per session (perc0, perc50, perc100)
    - Files named: {subject_id}_ses{session}_perc{level}.mat

    References
    ----------
    .. [1] Kobler, R. J., Almeida, I., Sburlea, A. I., & Muller-Putz, G. R.
           (2022). Continuous 2D trajectory decoding from attempted movement:
           across-session performance in able-bodied and feasibility in a
           spinal cord injured participant. Journal of Neural Engineering,
           19(3), 036005. https://doi.org/10.1088/1741-2552/ac689f

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is designed for continuous decoding research, specifically
    for predicting 2D hand movement trajectories from EEG. Unlike
    classification-based motor imagery datasets, this dataset contains
    continuous trajectory labels suitable for regression-based decoders.

    The paradigm "imagery" is used for compatibility with MOABB's motor
    imagery processing pipelines, though the actual task involves attempted
    (rather than imagined) movements.

    See Also
    --------
    BNCI2014_001 : 4-class motor imagery dataset
    BNCI2014_004 : 2-class motor imagery dataset
    """

    _participant_demographics = {
        "n_subjects": 20,
        "gender": {"male": 10, "female": 10},
        "age_mean": 24,
        "age_std": 5,
        "handedness": "all right-handed",
        "health_status": "able-bodied participants",
        "bci_experience": "4 had prior EEG experience",
        "location": "Institute of Neural Engineering, Graz University of Technology, Austria",
    }

    ARTICLE_METADATA = {
        "n_subjects": 20,
        "sessions_per_subject": 3,
        "sampling_rate": 200,
        "n_channels": 64,
        "channel_types": {"eeg": 60, "eog": 4},
        "paradigm": "imagery",
        "events": {"snakerun": 1, "freerun": 2, "eyerun": 3},
        "doi": "10.1088/1741-2552/ac689f",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "reference": "common average",
        "filters": "0.18-3 Hz (final processing)",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/002-2025/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 21)),
            sessions_per_subject=3,
            events={"snakerun": 1, "freerun": 2, "eyerun": 3},
            code="BNCI2025-002",
            interval=[0, 8],  # Trial length varies but 8s is a common window
            paradigm="imagery",
            doi="10.1088/1741-2552/ac689f",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        sessions = _load_data_002_2025(
            subject=subject,
            path=None,
            force_update=False,
            update_path=None,
            base_url=BNCI_URL,
            only_filenames=False,
            verbose=False,
        )
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return paths to data files for a single subject."""
        return _load_data_002_2025(
            subject=subject,
            path=path,
            force_update=force_update,
            update_path=update_path,
            base_url=BNCI_URL,
            only_filenames=True,
            verbose=verbose,
        )
