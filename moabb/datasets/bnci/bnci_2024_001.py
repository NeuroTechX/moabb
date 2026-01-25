"""BNCI 2024-001 Handwritten Character Classification dataset implementation.

This module provides the implementation for the BNCI 2024-001 dataset,
which contains EEG data from handwritten character classification tasks.

To integrate into bnci.py:
1. Add to dataset_list: "BNCI2024-001": _load_data_001_2024
2. Add to baseurl_list: "BNCI2024-001": BNCI_URL
3. Add to _dataset_years: "BNCI2024-001": 2024
4. Copy _load_data_001_2024 and _convert_run_001_2024 functions
5. Copy BNCI2024_001 class
6. Add BNCI2024_001 to __init__.py exports
"""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


@verbose
def _load_data_001_2024(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2024 dataset (Handwritten Character Classification).

    This dataset contains EEG data from 20 healthy subjects performing
    handwritten character (letter) writing tasks. The data was collected
    for research on handwritten character classification from EEG through
    continuous kinematic decoding.

    Parameters
    ----------
    subject : int
        Subject number (1-20).
    path : None | str
        Location for data storage.
    force_update : bool
        Force update of the dataset.
    update_path : bool | None
        If True, set the data path in config.
    base_url : str
        Base URL for data download.
    only_filenames : bool
        If True, return only the local path of the files without loading.
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    sessions : dict
        Dictionary of sessions with raw data.
    """
    if (subject < 1) or (subject > 20):
        raise ValueError("Subject must be between 1 and 20. Got %d." % subject)

    # Download the MAT file for this subject
    url = "{u}001-2024/sub{s:02d}.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]

    if only_filenames:
        return [filename]

    # Load the MAT file
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Channel names: 60 EEG + 4 EOG channels
    # Based on standard 10-20 extended montage for EEG channels
    # fmt: off
    ch_names_eeg = [
        "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
        "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
        "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2",
        "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
        "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5", "C1",
        "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5",
        "P1", "P2", "P6", "PO3",
    ]
    ch_names_eog = ["EOG1", "EOG2", "EOG3", "EOG4"]
    # fmt: on
    ch_names = ch_names_eeg + ch_names_eog
    ch_types = ["eeg"] * 60 + ["eog"] * 4

    # Process the data - assuming similar structure to other BNCI datasets
    runs = []

    if isinstance(data["data"], np.ndarray):
        run_array = data["data"]
    else:
        run_array = [data["data"]]

    for run in run_array:
        raw, evd = _convert_run_001_2024(run, ch_names, ch_types, verbose)
        if raw is not None:
            runs.append(raw)

    # Return in sessions format
    sessions = {"0": {str(ii): run for ii, run in enumerate(runs)}}
    return sessions


@verbose
def _convert_run_001_2024(run, ch_names, ch_types, verbose=None):
    """Convert one run from 001-2024 dataset to raw.

    Parameters
    ----------
    run : object
        Run data from MAT file.
    ch_names : list of str
        List of channel names.
    ch_types : list of str
        List of channel types.
    verbose : bool, str, int, or None
        Verbosity level.

    Returns
    -------
    raw : instance of RawArray
        Raw MNE object.
    event_id : dict
        Dictionary containing event codes.
    """
    # Parse EEG data
    n_chan = run.X.shape[1]
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X  # Convert from microvolts to volts
    sfreq = run.fs

    # Adjust channel names/types if necessary
    if n_chan != len(ch_names):
        # Fall back to generic channel names if mismatch
        ch_names = ["EEG%d" % ch for ch in range(1, n_chan + 1)]
        ch_types = ["eeg"] * n_chan
        montage = None

    # Create trigger channel
    trigger = np.zeros((len(eeg_data), 1))

    # Some runs may not contain trials (baseline runs)
    if hasattr(run, "trial") and len(run.trial) > 0:
        trigger[run.trial - 1, 0] = run.y
    else:
        return None, None

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = list(ch_names) + ["STI"]
    ch_types = list(ch_types) + ["stim"]

    # Create event_id from classes
    if hasattr(run, "classes"):
        event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    else:
        # Default event IDs for the 10 letters
        event_id = {
            "letter_a": 1,
            "letter_d": 2,
            "letter_e": 3,
            "letter_f": 4,
            "letter_j": 5,
            "letter_n": 6,
            "letter_o": 7,
            "letter_s": 8,
            "letter_t": 9,
            "letter_v": 10,
        }

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)

    if montage is not None:
        raw.set_montage(montage, on_missing="ignore")

    # Set line frequency (50 Hz for European datasets)
    raw.info["line_freq"] = 50.0

    return raw, event_id


class BNCI2024_001(BaseDataset):
    """BNCI 2024-001 Handwritten Character Classification dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 20 healthy subjects performing
    handwritten character (letter) writing tasks. Participants wrote 10
    different letters (a, d, e, f, j, n, o, s, t, v) while EEG was recorded.
    The study investigates the classification of handwritten characters from
    non-invasive EEG through continuous kinematic decoding.

    **Participants**

    - 20 healthy subjects
    - Location: Institute of Neural Engineering, Graz University of Technology,
      Austria

    **Recording Details**

    - Equipment: EEG system with 60 EEG + 4 EOG channels
    - Channels: 60 EEG electrodes + 4 EOG electrodes = 64 total
    - Electrode montage: Extended 10-20 system
    - Sampling rate: 512 Hz (estimated from similar Graz datasets)

    **Experimental Procedure**

    - 10 letter classes: a, d, e, f, j, n, o, s, t, v
    - Participants wrote letters inside a box while fixating on the screen
    - No visual feedback of the writing was provided during the task

    **Event Codes**

    The events correspond to the 10 different letters written by participants:

    - letter_a (1): Letter 'a'
    - letter_d (2): Letter 'd'
    - letter_e (3): Letter 'e'
    - letter_f (4): Letter 'f'
    - letter_j (5): Letter 'j'
    - letter_n (6): Letter 'n'
    - letter_o (7): Letter 'o'
    - letter_s (8): Letter 's'
    - letter_t (9): Letter 't'
    - letter_v (10): Letter 'v'

    References
    ----------
    .. [1] Crell, M. R., & Muller-Putz, G. R. (2024). Handwritten character
           classification from EEG through continuous kinematic decoding.
           Computers in Biology and Medicine, 182, 109132.
           https://doi.org/10.1016/j.compbiomed.2024.109132

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is notable for exploring non-invasive EEG-based handwritten
    character classification, which could enable communication for individuals
    with limited movement capacity. The study demonstrated that handwritten
    characters can be classified from non-invasive EEG and that decoding
    movement kinematics prior to classification improves performance.
    """

    _participant_demographics = {
        "n_subjects": 20,
        "health_status": "healthy subjects",
        "location": "Graz University of Technology, Austria",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 21)),
            sessions_per_subject=1,
            events={
                "letter_a": 1,
                "letter_d": 2,
                "letter_e": 3,
                "letter_f": 4,
                "letter_j": 5,
                "letter_n": 6,
                "letter_o": 7,
                "letter_s": 8,
                "letter_t": 9,
                "letter_v": 10,
            },
            code="BNCI2024-001",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.1016/j.compbiomed.2024.109132",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        sessions = _load_data_001_2024(subject=subject, verbose=False)
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of the dataset."""
        return _load_data_001_2024(
            subject=subject,
            verbose=verbose,
            update_path=update_path,
            path=path,
            force_update=force_update,
            only_filenames=True,
        )
