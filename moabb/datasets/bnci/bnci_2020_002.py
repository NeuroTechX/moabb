"""BNCI 2020-002 Attention Shift (Covert Spatial Attention) dataset."""

from datetime import date, datetime, timezone

import numpy as np
from mne.utils import verbose
from scipy.io import loadmat

from .base import BNCIBaseDataset
from .utils import (
    BNCI_URL,
    bnci_data_path,
    make_raw,
    standardize_channel_names,
    validate_subject,
)


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

    # Standardize channel names for montage compatibility
    ch_names = standardize_channel_names(ch_names)

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
        # 'yes' response is associated with attending right (green cross on right) -> Target
        # 'no' response is associated with attending left (red cross on left) -> NonTarget
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

    - NonTarget (1): Left attention (no response) - covert attention to left visual field
    - Target (2): Right attention (yes response) - covert attention to right visual field

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

    The classification approach focuses on detecting small attention-based
    differences between hemispheres using 14 posterior channels, which are
    expected to reflect correlates of visual spatial attention.

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

    _participant_demographics = {
        "n_subjects": 18,
        "gender": {"female": 10, "male": 8},
        "age_range": (19, 38),
        "age_mean": 27,
        "handedness": "all right-handed",
        "health_status": "healthy subjects with normal/corrected vision",
        "location": "Otto-von-Guericke University Magdeburg, Germany",
    }

    ARTICLE_METADATA = {
        "n_subjects": 18,
        "sessions_per_subject": 1,
        "sampling_rate": 250,
        "n_channels": 31,
        "channel_types": {"eeg": 29, "eog": 2},
        "paradigm": "p300",  # ERP-based paradigm
        "events": {"NonTarget": 1, "Target": 2},
        "doi": "10.3389/fnins.2020.591777",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "reference": "right mastoid",
        "equipment": "BrainAmp DC (Brain Products GmbH)",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/002-2020/",
    }

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
