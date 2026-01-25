"""BNCI 2020-001 Reach-and-Grasp Electrode Comparison dataset."""

from datetime import datetime, timezone

from mne import Annotations
from mne.utils import verbose
from scipy.io import loadmat

from .base import BNCIBaseDataset
from .utils import (
    BNCI_URL,
    bnci_data_path,
    convert_units,
    make_raw,
    validate_subject,
)


ELECTRODE_TYPES = [
    ("G", "gel"),
    ("V", "water"),
    ("H", "dry"),
]
SUBJECTS_PER_TYPE = 15
TOTAL_SUBJECTS = SUBJECTS_PER_TYPE * len(ELECTRODE_TYPES)


def _map_subject_to_electrode(subject):
    validate_subject(subject, TOTAL_SUBJECTS, "BNCI2020-001")
    type_idx, subj_idx = divmod(subject - 1, SUBJECTS_PER_TYPE)
    prefix, electrode_label = ELECTRODE_TYPES[type_idx]
    return prefix, electrode_label, subj_idx + 1


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
        Subject number (1-45). Subjects 1-15 are gel, 16-30 water, 31-45 dry.
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
        Dictionary of sessions with raw data. Each subject has one session.
    """
    prefix, electrode_label, subj_idx = _map_subject_to_electrode(subject)
    url = f"{base_url}001-2020/{prefix}{subj_idx:02d}.mat"
    filename = bnci_data_path(url, path, force_update, update_path, verbose)[0]

    if only_filenames:
        return [filename]

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
    for label in ch_labels:
        # Standardize EOG channel names
        if "EOG" in label.upper():
            clean_name = label.replace("-", "").replace(" ", "")
            ch_names.append(clean_name)
        elif "PTH" in label.upper():
            ch_names.append(label.replace("-", "_"))
        else:
            ch_names.append(label)

    # Convert signal to volts (data is in microvolts)
    eeg_eog_mask = eeg_idx + eog_idx
    signal_scaled = convert_units(
        signal.copy(), from_unit="uV", to_unit="V", channel_mask=eeg_eog_mask
    )

    raw = make_raw(
        signal_scaled,
        ch_names,
        ch_types,
        sfreq,
        verbose=verbose,
        montage="standard_1005",
        line_freq=50.0,
        meas_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        description=f"electrode_type={electrode_label}",
    )

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

    return {"0": {"0": raw}}


class BNCI2020_001(BNCIBaseDataset):
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

    **Electrode Types**

    Subjects are grouped by electrode type (15 per type). The subject index maps to:

    - 1-15: Gel-based electrode recording
    - 16-30: Water-based electrode recording
    - 31-45: Dry electrode recording

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
        "n_subjects": 45,
        "subjects_per_electrode_type": 15,
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
            subjects=list(range(1, TOTAL_SUBJECTS + 1)),
            sessions_per_subject=1,
            events={"palmar_grasp": 503587, "lateral_grasp": 503588, "rest": 768},
            code="BNCI2020-001",
            interval=[-2, 3],
            paradigm="imagery",
            doi="10.3389/fnins.2020.00849",
            load_fn=_load_data_001_2020,
            base_url=BNCI_URL,
        )
