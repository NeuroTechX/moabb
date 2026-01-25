"""BNCI 2020-001 Reach-and-Grasp Electrode Comparison dataset."""

from datetime import datetime, timezone

import numpy as np
from mne import Annotations, create_info
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
def _load_data_001_2020(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2020 dataset (Reach-and-Grasp electrode comparison).

    This dataset contains EEG data from 45 subjects (15 per electrode type)
    performing natural reach-and-grasp movements. Three electrode types were
    used: gel-based (G), water-based (V), and dry electrodes (H).

    Parameters
    ----------
    subject : int
        Subject number (1-15). Each subject ID maps to one subject from
        each electrode type (gel, water, dry).
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
        Dictionary of sessions with raw data. Sessions are organized by
        electrode type: '0gel', '1water', '2dry'.
    """
    if (subject < 1) or (subject > 15):
        raise ValueError("Subject must be between 1 and 15. Got %d." % subject)

    # Electrode type prefixes and session names
    # G = gel (g.tec), V = versatile (water-based), H = hero (dry)
    electrode_types = [
        ("G", "0gel"),
        ("V", "1water"),
        ("H", "2dry"),
    ]

    sessions = {}
    filenames = []

    for prefix, session_name in electrode_types:
        url = f"{base_url}001-2020/{prefix}{subject:02d}.mat"
        filename = _data_path(url, path, force_update, update_path, verbose)[0]
        filenames.append(filename)

        if only_filenames:
            continue

        # Load the MAT file
        mat_data = loadmat(filename, struct_as_record=False, squeeze_me=True)

        header = mat_data["header"]
        events = mat_data["events"]
        signal = mat_data["signal"]

        # Get channel information
        sfreq = float(header.sample_rate)
        n_channels = signal.shape[0]

        # Only use channel labels that correspond to actual signal channels
        # Some files have extra labels (e.g., PTH channels) not in the signal
        all_labels = [ch.strip() for ch in header.channels_labels]
        ch_labels = all_labels[:n_channels]

        # Determine channel types based on header information
        eeg_idx = list(header.channels_eeg - 1)  # Convert to 0-indexed
        if hasattr(header.channels_eog, "__len__") and len(header.channels_eog) > 0:
            eog_idx = list(header.channels_eog - 1)
        else:
            eog_idx = []

        # Filter indices to only include channels within signal range
        eeg_idx = [idx for idx in eeg_idx if idx < n_channels]
        eog_idx = [idx for idx in eog_idx if idx < n_channels]

        ch_types = ["misc"] * n_channels
        for idx in eeg_idx:
            ch_types[idx] = "eeg"
        for idx in eog_idx:
            ch_types[idx] = "eog"

        # Clean up channel names
        ch_names = []
        for i, label in enumerate(ch_labels):
            # Standardize EOG channel names
            if "EOG" in label.upper():
                clean_name = label.replace("-", "").replace(" ", "")
                ch_names.append(clean_name)
            elif "PTH" in label.upper():
                ch_names.append(label.replace("-", "_"))
            else:
                ch_names.append(label)

        # Create MNE info structure
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Convert signal to volts (data is in microvolts)
        signal_scaled = signal.copy().astype(np.float64)
        for idx in eeg_idx + eog_idx:
            signal_scaled[idx, :] *= 1e-6

        # Create RawArray
        raw = RawArray(signal_scaled, info, verbose=verbose)

        # Set line frequency (European dataset)
        raw.info["line_freq"] = 50.0

        # Set measurement date (dataset recorded 2020)
        raw.set_meas_date(datetime(2020, 1, 1, tzinfo=timezone.utc))

        # Create annotations from events
        # Filter for movement onset and rest events only
        relevant_codes = [503587, 503588, 768]  # palmar onset, lateral onset, rest onset
        code_to_desc = {
            503587: "palmar_grasp",
            503588: "lateral_grasp",
            768: "rest",
        }

        onset_times = []
        descriptions = []
        for pos, code in zip(events.positions, events.codes):
            if code in relevant_codes:
                onset_times.append(pos / sfreq)
                descriptions.append(code_to_desc[code])

        if onset_times:
            annotations = Annotations(
                onset=onset_times,
                duration=[0.0] * len(onset_times),
                description=descriptions,
            )
            raw.set_annotations(annotations)

        # Set montage for EEG channels
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")

        sessions[session_name] = {"0": raw}

    if only_filenames:
        return filenames
    return sessions


class BNCI2020_001(BaseDataset):
    """BNCI 2020-001 Reach-and-Grasp Electrode Comparison dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 45 subjects (15 per electrode type)
    performing natural reach-and-grasp movements with different electrode
    systems. Three electrode types were compared:

    - **Gel-based electrodes** (g.tec g.USBamp system): 58 EEG + 6 EOG channels
    - **Water-based electrodes** (BitBrain EEG-Versatile): 32 EEG + 6 EOG channels
    - **Dry electrodes** (BitBrain EEG-Hero): 11 EEG channels (no EOG)

    The study investigates the feasibility of decoding natural reach-and-grasp
    movements from EEG signals recorded with different electrode technologies,
    including mobile systems suitable for real-world applications.

    **Participants**

    - 45 healthy able-bodied subjects (15 per electrode type)
    - All subjects performed the same experimental protocol
    - Each subject used only one electrode type
    - Location: Graz University of Technology, Austria (in collaboration with
      BitBrain, Spain)

    **Recording Details**

    - Sampling rate: 256 Hz (all systems)
    - Reference: Earlobe (right for gel, left for water/dry)
    - Ground: AFz (gel/water), left earlobe (dry)
    - Filters: 0.3-100 Hz bandpass (3rd-4th order Butterworth)

    **Experimental Procedure**

    - Self-paced reaching and grasping actions toward objects on a table
    - Two grasp types: palmar grasp (empty jar) and lateral grasp (spoon in jar)
    - Rest condition: Quiet sitting with fixation
    - 80 trials per grasp type distributed across 4 runs
    - Window of interest: [-2, 3] seconds relative to movement onset

    **Event Codes**

    - palmar_grasp: Movement onset for palmar grasp (reaching to empty jar)
    - lateral_grasp: Movement onset for lateral grasp (reaching to jar with spoon)
    - rest: Onset of rest period

    **Sessions**

    Data for each subject is organized into three sessions by electrode type:

    - Session '0gel': Gel-based electrode recording
    - Session '1water': Water-based electrode recording
    - Session '2dry': Dry electrode recording

    **Classification Results (from original paper)**

    Grand average peak accuracy on unseen test data:

    - Gel-based: 61.3% (8.6% STD)
    - Water-based: 62.3% (9.2% STD)
    - Dry electrodes: 56.4% (8.0% STD)

    References
    ----------
    .. [1] Schwarz, A., Escolano, C., Montesano, L., & Muller-Putz, G. R. (2020).
           Analyzing and Decoding Natural Reach-and-Grasp Actions Using Gel,
           Water and Dry EEG Systems. Frontiers in Neuroscience, 14, 849.
           https://doi.org/10.3389/fnins.2020.00849

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset is valuable for comparing electrode technologies in naturalistic
    movement paradigms. Data is available under CC BY 4.0 license.
    """

    _participant_demographics = {
        "n_subjects": 15,
        "n_subjects_total": 45,
        "health_status": "healthy able-bodied subjects",
        "location": "Graz University of Technology, Austria",
        "electrode_types": [
            "gel (g.tec)",
            "water (BitBrain Versatile)",
            "dry (BitBrain Hero)",
        ],
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=3,
            events={"palmar_grasp": 503587, "lateral_grasp": 503588, "rest": 768},
            code="BNCI2020-001",
            interval=[-2, 3],
            paradigm="imagery",
            doi="10.3389/fnins.2020.00849",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        sessions = _load_data_001_2020(
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
        """Return the data paths of the dataset."""
        return _load_data_001_2020(
            subject=subject,
            path=path,
            force_update=force_update,
            update_path=update_path,
            base_url=BNCI_URL,
            only_filenames=True,
            verbose=verbose,
        )
