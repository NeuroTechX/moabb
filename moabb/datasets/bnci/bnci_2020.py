"""BNCI 2020 datasets."""

from datetime import date, datetime, timezone

import numpy as np
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


# =============================================================================
# BNCI 2020-001: Reach-and-Grasp Electrode Comparison
# =============================================================================

ELECTRODE_TYPES = [
    ("G", "gel"),
    ("V", "water"),
    ("H", "dry"),
]
SUBJECTS_PER_TYPE = 15
TOTAL_SUBJECTS_001 = SUBJECTS_PER_TYPE * len(ELECTRODE_TYPES)


def _map_subject_to_electrode(subject):
    validate_subject(subject, TOTAL_SUBJECTS_001, "BNCI2020-001")
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

    def __init__(self):
        super().__init__(
            subjects=list(range(1, TOTAL_SUBJECTS_001 + 1)),
            sessions_per_subject=1,
            events={"palmar_grasp": 503587, "lateral_grasp": 503588, "rest": 768},
            code="BNCI2020-001",
            interval=[-2, 3],
            paradigm="imagery",
            doi="10.3389/fnins.2020.00849",
            load_fn=_load_data_001_2020,
            base_url=BNCI_URL,
        )


# =============================================================================
# BNCI 2020-002: Attention Shift (Covert Spatial Attention)
# =============================================================================


@verbose
def _load_data_002_2020(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 002-2020 Attention Shift dataset.

    This dataset contains EEG recordings from 18 subjects performing a
    covert spatial attention task. Subjects attended to colored stimuli
    (red/green crosses) in the left or right visual field to communicate
    yes/no responses.

    Parameters
    ----------
    subject : int
        Subject number (1-18).
    path : str | None
        Path to download/load data.
    force_update : bool
        Force download of data.
    update_path : bool | None
        If True, set the path in config.
    base_url : str
        Base URL for downloading.
    only_filenames : bool
        If True, return only filenames.
    verbose : bool | str | int | None
        Verbosity level.

    Returns
    -------
    sessions : dict
        Dictionary with session data.
    """
    validate_subject(subject, 18, "BNCI2020-002")

    url = "{u}002-2020/P{s:02d}.mat".format(u=base_url, s=subject)
    filename = bnci_data_path(url, path, force_update, update_path)[0]

    if only_filenames:
        return [filename]

    raw, event_id = _convert_attention_shift(filename, verbose=verbose)

    # Extract subject metadata and enrich raw object
    mat_data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    if "subject" in mat_data:
        subj_info = mat_data["subject"]
        subject_info = {}

        # Extract age
        if hasattr(subj_info, "age"):
            age = int(subj_info.age)
            # Recording year is 2020 based on dataset code
            rec_year = 2020
            birth_year = rec_year - age
            subject_info["birthday"] = date(birth_year, 1, 1)

        # Extract sex
        if hasattr(subj_info, "sex"):
            sex_str = str(subj_info.sex).lower()
            if sex_str in ["male", "m"]:
                subject_info["sex"] = 1
            elif sex_str in ["female", "f"]:
                subject_info["sex"] = 2
            else:
                subject_info["sex"] = 0

        # Extract handedness
        if hasattr(subj_info, "handedness"):
            hand_str = str(subj_info.handedness).lower()
            if hand_str in ["right", "r"]:
                subject_info["hand"] = 1
            elif hand_str in ["left", "l"]:
                subject_info["hand"] = 2
            else:
                subject_info["hand"] = 0

        if subject_info:
            raw.info["subject_info"] = subject_info

    sessions = {"0": {"0": raw}}
    return sessions


@verbose
def _convert_attention_shift(filename, verbose=None):
    """Convert attention shift data from MAT file to MNE Raw.

    The data is organized as trials with shape (channels, samples, trials).
    We concatenate all trials into a continuous recording with event markers.

    Parameters
    ----------
    filename : str
        Path to the MAT file.
    verbose : bool | str | int | None
        Verbosity level.

    Returns
    -------
    raw : mne.io.RawArray
        The MNE Raw object.
    event_id : dict
        Dictionary mapping event names to codes.
    """
    mat_data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    bciexp = mat_data["bciexp"]

    sfreq = float(bciexp.srate)
    n_channels, n_samples, n_trials = bciexp.data.shape

    # Get channel names - these are EEG channels
    ch_names = list(bciexp.label)

    # Channel types: all EEG
    ch_types = ["eeg"] * n_channels

    # Add EOG channels from separate fields
    ch_names_full = ch_names + ["HEOG", "VEOG", "STI"]
    ch_types_full = ch_types + ["eog", "eog", "stim"]

    # Reshape data: concatenate trials
    # Original: (channels, samples, trials) -> (channels, samples * trials)
    eeg_data = bciexp.data.reshape(n_channels, -1)

    # Get EOG data: (samples, trials) -> (samples * trials)
    heog_data = bciexp.heog.T.reshape(1, -1)
    veog_data = bciexp.veog.T.reshape(1, -1)

    # Create stimulus channel with trial onset markers
    # For P300 paradigm compatibility, we use Target/NonTarget naming:
    # - Target (2): Right attention (yes response) - the attended stimulus
    # - NonTarget (1): Left attention (no response)
    stim_data = np.zeros((1, n_samples * n_trials))

    # Get intentions for each trial
    intentions = np.asarray(bciexp.intention)
    event_id = {"NonTarget": 1, "Target": 2}

    value_map = None
    try:
        numeric_vals = intentions.astype(int)
    except (ValueError, TypeError):
        numeric_vals = None
    if numeric_vals is not None:
        unique_vals = set(np.unique(numeric_vals).tolist())
        if unique_vals <= {0, 1}:
            value_map = {0: 1, 1: 2}
        elif unique_vals <= {1, 2}:
            value_map = {1: 1, 2: 2}

    target_tokens = {"yes", "y", "right", "r", "target", "true"}
    nontarget_tokens = {"no", "n", "left", "l", "nontarget", "false"}

    for trial_idx in range(n_trials):
        trial_start = trial_idx * n_samples
        # Map intention to event code
        # 'yes' response is associated with attending right (green cross) -> Target
        # 'no' response is associated with attending left (red cross) -> NonTarget
        intention = intentions[trial_idx]
        if value_map is not None:
            code = value_map.get(int(intention), 1)
        else:
            token = str(intention).strip().lower()
            if token in target_tokens:
                code = 2
            elif token in nontarget_tokens:
                code = 1
            else:
                code = 1
        stim_data[0, trial_start] = code

    # Combine all data
    # Scale EEG data to Volts (data is in microvolts)
    all_data = np.vstack([eeg_data * 1e-6, heog_data * 1e-6, veog_data * 1e-6, stim_data])

    raw = make_raw(
        all_data,
        ch_names_full,
        ch_types_full,
        sfreq,
        verbose=verbose,
        montage="standard_1005",
        line_freq=50.0,
        meas_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    return raw, event_id


class BNCI2020_002(BNCIBaseDataset):
    """BNCI 2020-002 Attention Shift (Covert Spatial Attention) dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 18 healthy subjects performing
    a covert spatial attention task for brain-computer interface (BCI) control.
    The paradigm decodes binary decisions based on the N2pc component - a
    neurological marker reflecting attention to visual targets in specific
    hemispheres.

    Subjects were presented with colored stimuli (red and green crosses) in
    left and right visual hemifields simultaneously. By covertly shifting
    attention to one side (left or right), subjects could indicate "yes" or
    "no" responses without any overt movement, enabling gaze-independent
    communication.

    **Participants**

    - 18 healthy subjects (10 female)
    - Age range: 19-38 years (mean 27 years)
    - All right-handed
    - Normal or corrected-to-normal vision
    - Location: Otto-von-Guericke University Magdeburg, Germany

    **Recording Details**

    - Equipment: BrainAmp DC Amplifier (Brain Products GmbH)
    - Channels: 29 EEG + 2 EOG (horizontal and vertical)
    - Electrode positions: Standard 10-20 system
    - Reference: Right mastoid
    - Sampling rate: 250 Hz
    - Hardware filter: 0.1 Hz high-pass
    - Display: 24" TFT, 70 cm viewing distance

    **Experimental Procedure**

    - Binary communication task: attend left (red cross) for "no",
      attend right (green cross) for "yes"
    - 120 statements presented, subjects respond by covert attention shift
    - Each trial: 10 visual stimuli presentations
    - Stimulus parameters tested:
        - Four symbol sizes: 0.45, 0.90, 1.36, 1.81 degrees visual angle
        - Five eccentricities: 4, 5.5, 7, 8.5, 10 degrees visual angle
    - Inter-stimulus interval: ~175 ms
    - Online accuracy: 88.5% (+/- 7.8%)

    **Event Codes**

    For P300 paradigm compatibility, events are named Target/NonTarget:

    - NonTarget (1): Left attention (no response)
    - Target (2): Right attention (yes response)

    **Data Organization**

    - 1 session per subject
    - 120 trials per subject, each with 10 stimulus presentations
    - Trial duration: 16 seconds (4000 samples at 250 Hz)
    - Data stored in MAT format with fields:
        - bciexp.data: EEG data (channels x samples x trials)
        - bciexp.heog, bciexp.veog: EOG data
        - bciexp.intention: subject's intended response (yes/no)
        - subject: demographic information

    References
    ----------
    .. [1] Reichert, C., Tellez-Ceja, I. F., Schwenker, F., Rusnac, A.-L.,
           Curio, G., Aust, L., & Hinrichs, H. (2020). Impact of Stimulus Features
           on the Performance of a Gaze-Independent Brain-Computer Interface
           Based on Covert Spatial Attention Shifts. Frontiers in Neuroscience,
           14, 591777. https://doi.org/10.3389/fnins.2020.591777

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset uses a covert spatial attention paradigm with N2pc ERP
    detection, which is different from traditional P300 or motor imagery
    paradigms. The paradigm is designed for gaze-independent BCI control,
    making it suitable for users who cannot control eye movements.

    See Also
    --------
    BNCI2015_009 : AMUSE auditory spatial P300 paradigm
    BNCI2015_010 : RSVP visual P300 paradigm

    Examples
    --------
    >>> from moabb.datasets import BNCI2020_002
    >>> dataset = BNCI2020_002()
    >>> dataset.subject_list
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 19)),
            sessions_per_subject=1,
            events={"NonTarget": 1, "Target": 2},
            code="BNCI2020-002",
            interval=[0, 16],  # 16 seconds per trial (4000 samples at 250 Hz)
            paradigm="p300",  # ERP-based paradigm for compatibility
            doi="10.3389/fnins.2020.591777",
            load_fn=_load_data_002_2020,
            base_url=BNCI_URL,
        )
