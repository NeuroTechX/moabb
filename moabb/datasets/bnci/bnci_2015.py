"""BNCI 2015 datasets."""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.utils import depreciated_alias

from .base import (
    BBCI_URL,
    BNCI_URL,
    MNEBNCI,
    _convert_bbci,
    _convert_mi,
    _convert_run_epfl,
    _finalize_raw,
    data_path,
    load_data,
)
from .utils import convert_units, validate_subject


@verbose
def _load_data_001_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2015 dataset."""
    validate_subject(subject, 12, "BNCI2015-001")

    if subject in [8, 9, 10, 11]:
        ses = [(0, "A"), (1, "B"), (2, "C")]  # 3 sessions for those subjects
    else:
        ses = [(0, "A"), (1, "B")]

    # fmt: off
    ch_names = [
        "FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CPz", "CP4",
    ]
    # fmt: on
    ch_types = ["eeg"] * 13

    sessions = {}
    filenames = []
    for session_idx, r in ses:
        url = "{u}001-2015/S{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        filenames += filename
        if only_filenames:
            continue
        runs, ev = _convert_mi(filename[0], ch_names, ch_types, subject_id=subject)
        sessions[f"{session_idx}{r}"] = {str(ii): run for ii, run in enumerate(runs)}
    if only_filenames:
        return filenames
    return sessions


@verbose
def _load_data_003_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 003-2015 dataset."""
    validate_subject(subject, 10, "BNCI2015-003")

    url = "{u}003-2015/s{s:d}.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = data["s%d" % subject]
    sfreq = 256.0

    ch_names = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "Oz", "PO8", "Target", "Flash"]

    ch_types = ["eeg"] * 8 + ["stim"] * 2

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    sessions = {}
    sessions["0"] = {}
    for r_name, run in [("0train", data.train), ("1test", data.test)]:
        # flash events on the channel 9
        flashs = run[9:10]
        ix_flash = flashs[0] > 0
        flashs[0, ix_flash] += 2  # add 2 to avoid overlap on event id
        flash_code = np.unique(flashs[0, ix_flash])

        if len(flash_code) == 36:
            # char mode
            evd = {"Char%d" % ii: (ii + 2) for ii in range(1, 37)}
        else:
            # row / column mode
            evd = {"Col%d" % ii: (ii + 2) for ii in range(1, 7)}
            evd.update({"Row%d" % ii: (ii + 8) for ii in range(1, 7)})

        # target events are on channel 10
        targets = np.zeros_like(flashs)
        targets[0, ix_flash] = run[10, ix_flash] + 1

        eeg_channels = convert_units(run[1:-2], from_unit="uV", to_unit="V")
        eeg_data = np.r_[eeg_channels, targets, flashs]
        raw = RawArray(data=eeg_data, info=info, verbose=verbose)
        # Enrich raw object with additional metadata
        raw.info["line_freq"] = 50.0
        _finalize_raw(raw, "BNCI2015-003", subject)

        sessions["0"][r_name] = raw

    return sessions


@verbose
def _load_data_004_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 004-2015 dataset."""
    validate_subject(subject, 9, "BNCI2015-004")

    subjects = ["A", "C", "D", "E", "F", "G", "H", "J", "L"]

    url = "{u}004-2015/{s}.mat".format(u=base_url, s=subjects[subject - 1])
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    # fmt: off
    ch_names = [
        "AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T3", "C3",
        "Cz", "C4", "T4", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1", "Pz",
        "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2",
    ]
    # fmt: on
    ch_types = ["eeg"] * 30
    raws, ev = _convert_mi(
        filename, ch_names, ch_types, dataset_code="BNCI2015-004", subject_id=subject
    )
    sessions = {str(ii): {"0": run} for ii, run in enumerate(raws)}
    return sessions


@verbose
def _load_data_007_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 007-2015 dataset (Motion VEP Speller).

    This dataset contains motion-onset visual evoked potentials (mVEPs)
    for gaze-independent BCI communication. Uses BBCI data format.
    """
    validate_subject(subject, 16, "BNCI2015-007")

    # Subject codes for the 16 subjects
    # fmt: off
    subjects = [
        "fat", "gdf", "gdg", "iac", "iba", "ibe", "ibq", "ibs",
        "ibt", "ibu", "ibv", "ibw", "ibx", "iby", "ice", "icv",
    ]
    # fmt: on

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-MVEP/MVEP_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 63

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-007", subject)
    return raws, event_id


@verbose
def _load_data_008_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 008-2015 dataset (Center Speller).

    This dataset contains P300 evoked potentials recorded during a gaze-independent
    two-stage visual speller paradigm called the "Center Speller".
    """
    validate_subject(subject, 13, "BNCI2015-008")

    # fmt: off
    subjects = [
        "iac", "iba", "ibb", "ibc", "ibd", "ibe", "ibf",
        "ibg", "ibh", "ibi", "ibj", "ica", "saf",
    ]
    # fmt: on

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-CenterSpeller/CenterSpeller_VP{s}.mat".format(
        u=base_url, s=s
    )
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 63

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-008", subject)
    return raws, event_id


@verbose
def _load_data_006_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 006-2015 dataset (Music BCI)."""
    validate_subject(subject, 11, "BNCI2015-006")
    # Subject codes from BNCI website (not sequential vp1-vp11)
    subjects = [
        "VPaak",
        "VPaan",
        "VPgcc",
        "VPaap",
        "VPaaq",
        "VPjaq",
        "VPaar",
        "VPjat",
        "VPgeo",
        "VPaas",
        "VPaat",
    ]
    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-MusicBCI/musicbci_{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]
    mat_data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = mat_data["data"]
    eeg_data = data.X
    sfreq = float(data.fs)
    ch_names = [str(ch).strip() for ch in data.clab]
    ch_types = ["eeg"] * len(ch_names)
    trigger = np.zeros((len(eeg_data), 1))
    if hasattr(data, "trial") and len(data.trial) > 0:
        trial_indices = np.array(data.trial).flatten().astype(int) - 1
        trial_labels = np.array(data.y).flatten().astype(int)
        valid_mask = (trial_indices >= 0) & (trial_indices < len(eeg_data))
        trigger[trial_indices[valid_mask], 0] = trial_labels[valid_mask]
    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ["STI"]
    ch_types = ch_types + ["stim"]
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    raw.info["line_freq"] = 50.0
    _finalize_raw(raw, "BNCI2015-006", subject)
    sessions = {"0": {"0": raw}}
    return sessions


@verbose
def _load_data_009_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 009-2015 dataset."""
    validate_subject(subject, 21, "BNCI2015-009")

    # fmt: off
    subjects = [
        "fce", "kw", "faz", "fcj", "fcg", "far", "faw", "fax", "fcc", "fcm", "fas",
        "fch", "fcd", "fca", "fcb", "fau", "fci", "fav", "fat", "fcl", "fck",
    ]
    # fmt: on
    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-AMUSE/AMUSE_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 60 + ["eog"] * 2

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-009", subject)
    return raws, event_id


@verbose
def _load_data_010_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 010-2015 dataset."""
    validate_subject(subject, 12, "BNCI2015-010")

    # fmt: off
    subjects = [
        "fat", "gcb", "gcc", "gcd", "gce", "gcf",
        "gcg", "gch", "iay", "icn", "icr", "pia",
    ]
    # fmt: on

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-RSVP/RSVP_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    # Pass None for ch_types to allow dynamic detection based on actual channel count
    # Most subjects have 63 channels, but some (e.g., subject 5/VPgce) have 61
    ch_types = None

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-010", subject)
    return raws, event_id


@verbose
def _load_data_012_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 012-2015 dataset."""
    validate_subject(subject, 10, "BNCI2015-012")

    # Subject codes - removed "nx" (original subject 3) and "mg" (original subject 6)
    # as their data files are not available on the BNCI server (HTTP 404)
    subjects = ["nv", "nw", "ny", "nz", "oa", "ob", "oc", "od", "ja", "oe"]

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-PASS2D/PASS2D_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 63

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-012", subject)
    return raws, event_id


@verbose
def _load_data_013_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 013-2015 dataset."""
    validate_subject(subject, 6, "BNCI2015-013")

    data_paths = []
    for r in ["s1", "s2"]:
        url = "{u}013-2015/Subject{s:02d}_{r}.mat".format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))
    if only_filenames:
        return data_paths

    raws = []
    event_id = {}

    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        for run in data["run"]:
            raw, evd = _convert_run_epfl(run, verbose=verbose)
            _finalize_raw(raw, "BNCI2015-013", subject)
            raws.append(raw)
            event_id.update(evd)
    return raws, event_id


@depreciated_alias("BNCI2015001", "1.1")
class BNCI2015_001(MNEBNCI):
    """BNCI 2015-001 Motor Imagery dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 12 subjects performing two-class motor
    imagery tasks (left vs right hand). Each subject participated in multiple
    sessions, with some subjects having three sessions.

    **Participants**

    - 12 healthy subjects
    - Gender: not specified
    - Age: not specified

    **Recording Details**

    - Channels: 13 EEG electrodes
    - Sampling rate: 512 Hz
    - Reference: not specified

    References
    ----------
    .. [1] Faller, J., Scherer, R., Costa, U., Opisso, E., Medina, J., Muller-Putz,
           G. R. (2014). A co-adaptive brain-computer interface for end users with
           severe motor impairment. PLOS ONE, 9(7), e101168.
           https://doi.org/10.1371/journal.pone.0101168

    Notes
    -----
    .. versionadded:: 0.4.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=2,
            events={"right_hand": 1, "feet": 2},
            code="BNCI2015-001",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1109/tnsre.2012.2189584",
        )


@depreciated_alias("BNCI2015003", "1.1")
class BNCI2015_003(MNEBNCI):
    """BNCI 2015-003 P300 dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 10 subjects using a P300 speller
    system. The dataset includes target and non-target responses during a
    visual P300 paradigm.

    References
    ----------
    .. [1] Schreuder, M., Rost, T., & Tangermann, M. (2011). Listen, you are
           writing! Speeding up online spelling with a dynamic auditory BCI.
           Frontiers in neuroscience, 5, 112.
           https://doi.org/10.3389/fnins.2011.00112

    Notes
    -----
    .. versionadded:: 0.4.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2015-003",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1016/j.neulet.2009.06.045",
        )


@depreciated_alias("BNCI2015004", "1.1")
class BNCI2015_004(MNEBNCI):
    """BNCI 2015-004 Mental tasks dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 9 subjects performing five different
    mental tasks: mental multiplication, mental letter composing, mental
    rotation, mental counting, and a baseline task.

    References
    ----------
    .. [1] Zhang, X., Yao, L., Zhang, Q., Kanhere, S., Sheng, M., & Liu, Y.
           (2017). A survey on deep learning based brain computer interface:
           Recent advances and new frontiers. IEEE Transactions on Cognitive
           and Developmental Systems, 10(2), 145-163.

    Notes
    -----
    .. versionadded:: 0.4.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events={
                "math": 1,
                "letter": 2,
                "rotation": 3,
                "count": 4,
                "baseline": 5,
            },
            code="BNCI2015-004",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1109/TCDS.2017.2688350",
        )


class BNCI2015_006(MNEBNCI):
    """BNCI 2015-006 Music BCI dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset investigates the suitability of musical stimuli for use in a P300
    paradigm. 11 subjects listened to polyphonic music clips featuring three
    instruments playing together. A multi-streamed oddball paradigm was used.

    References
    ----------
    .. [1] Treder, M. S., Purwins, H., Miklody, D., Sturm, I., & Blankertz, B.
           (2014). Decoding auditory attention to instruments in polyphonic music
           using single-trial EEG classification. Journal of Neural Engineering,
           11(2), 026009. https://doi.org/10.1088/1741-2560/11/2/026009

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 12)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-006",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.1088/1741-2560/11/2/026009",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        sessions = load_data(subject=subject, dataset=self.code, verbose=False)
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of the dataset."""
        return load_data(
            subject=subject,
            dataset=self.code,
            verbose=verbose,
            update_path=update_path,
            path=path,
            force_update=force_update,
            only_filenames=True,
        )


class BNCI2015_007(MNEBNCI):
    """BNCI 2015-007 Motion VEP (mVEP) Speller dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_007  16      63      ~1800 NT / ~360 T   0.7s            100Hz           1
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset implements a motion-onset visual evoked potential (mVEP) based
    brain-computer interface for gaze-independent spelling. Unlike conventional
    flash-based P300 spellers that use luminance changes, this paradigm uses
    motion onset (moving bar stimuli) to elicit visual evoked potentials,
    specifically the N200 component. This approach has advantages including
    lower visual fatigue, reduced luminance and contrast requirements, and
    potential for use in bright environments.

    The motion VEP (mVEP) speller operates by presenting moving bar stimuli at
    different positions in a matrix layout. When the user attends to a target
    position, the motion onset at that location elicits a characteristic N200
    response that can be detected to determine the user's intended selection.

    **Participants**

    - 16 healthy subjects
    - Gender: Not specified in metadata
    - Age: Not specified in metadata
    - BCI experience: Not specified
    - Health status: Healthy volunteers
    - Location: Neurotechnology Group, Technische Universitat Berlin, Germany

    **Recording Details**

    - Equipment: BrainProducts actiCap active electrode system
    - Channels: 63 EEG electrodes (standard 10-10 system)
    - Sampling rate: 100 Hz (downsampled from original recording)
    - Reference: Nose reference
    - Montage: standard_1005
    - Filters: Bandpass filtered during preprocessing
    - Units: uV (converted to V during loading)

    **Experimental Procedure**

    - 6x6 matrix speller layout (36 possible targets)
    - Motion onset stimulation (moving bars)
    - 6 stimulus positions per row/column
    - Overt attention paradigm (gaze-dependent) and covert attention modes
    - One recording session per subject with multiple runs (typically 2)
    - Each run contains multiple spelling sequences

    **Data Organization**

    - Subject codes: fat, gdf, gdg, iac, iba, ibe, ibq, ibs, ibt, ibu, ibv, ibw,
      ibx, iby, ice, icv
    - Data URL: http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-MVEP/

    **Event Codes**

    - Target (1): Target stimulus presented (attended)
    - NonTarget (2): Non-target stimulus presented (not attended)

    References
    ----------
    .. [1] Treder, M. S., Purwins, H., Miklody, D., Sturm, I., & Blankertz, B.
           (2012). Decoding auditory attention to instruments in polyphonic music
           using single-trial EEG classification. Journal of Neural Engineering,
           11(2), 026009. https://doi.org/10.1088/1741-2560/11/2/026009

    Notes
    -----
    .. versionadded:: 1.2.0

    See Also
    --------
    BNCI2015_008 : Center Speller P300 dataset (gaze-independent)
    BNCI2015_009 : AMUSE auditory spatial P300 dataset
    BNCI2015_010 : RSVP visual speller (gaze-independent visual paradigm)
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 17)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-007",
            interval=[0, 0.7],
            paradigm="p300",  # Oddball-like paradigm with Target/NonTarget
            doi="10.1088/1741-2560/9/4/045006",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject.

        This dataset returns raws and event_id from load_data, so we need
        to convert to the standard sessions format.
        """
        raws, event_id = load_data(subject=subject, dataset=self.code, verbose=False)
        # Convert list of raws to sessions format
        # Single session with multiple runs
        sessions = {"0": {str(ii): raw for ii, raw in enumerate(raws)}}
        return sessions


class BNCI2015_008(MNEBNCI):
    """BNCI 2015-008 Center Speller P300 dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_008  13      63      ~1180 T / ~5900 NT  1.0s            250Hz           2
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_, also known as Treder2011.

    **Dataset Description**

    This dataset contains P300 evoked potentials recorded during a gaze-independent
    two-stage visual speller paradigm called the "Center Speller". Unlike traditional
    matrix spellers that require gaze fixation on target cells, the Center Speller
    allows users to focus on the screen center while covertly attending to peripheral
    stimuli.

    The paradigm uses a two-stage selection process where users first select a group
    of characters, then select individual characters within that group. This design
    enables efficient spelling without requiring eye movements, making it suitable
    for users with severe motor disabilities affecting eye control.

    **Participants**

    - 13 healthy subjects
    - BCI experience: Previous experience with P300-based BCIs
    - Location: Machine Learning Laboratory, TU Berlin, Germany

    **Recording Details**

    - Channels: 63 EEG electrodes (standard 10-10 system)
    - Sampling rate: 250 Hz
    - Reference: Nose reference

    **Data Organization**

    - Subject codes: iac, iba, ibb, ibc, ibd, ibe, ibf, ibg, ibh, ibi, ibj, ica, saf
    - Two runs per subject (calibration + online)
    - Data URL: http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-CenterSpeller/

    **Event Codes**

    - Target (1): Target stimulus presented (attended)
    - NonTarget (2): Non-target stimulus presented (not attended)

    References
    ----------
    .. [1] Treder, M. S., Schmidt, N. M., & Blankertz, B. (2011). Gaze-independent
           brain-computer interfaces based on covert attention and feature attention.
           Journal of Neural Engineering, 8(6), 066003.
           https://doi.org/10.1088/1741-2560/8/6/066003

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 14)),
            sessions_per_subject=2,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-008",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.1088/1741-2560/8/6/066003",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, event_id = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {"0": {str(ii): raw for ii, raw in enumerate(raws)}}
        return sessions


class BNCI2015_009(MNEBNCI):
    """BNCI 2015-009 AMUSE (Auditory Multi-class Spatial ERP) dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_009  21      62      Variable T/NT       0.8s            1000Hz          varies
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 21 subjects performing an
    auditory spatial attention task for brain-computer interface (BCI) control.
    The AMUSE (Auditory Multi-class Spatial ERP) paradigm uses auditory stimuli
    from different spatial locations to elicit P300-like event-related potentials.

    Subjects were presented with auditory stimuli (75 ms bandpass filtered white
    noise, 150-8000 Hz) from 8 loudspeakers arranged at ear height in a circle
    around the subject, with 45 degree spacing at approximately 1 meter distance.
    By attending to stimuli from a specific spatial location, subjects could
    select one of multiple targets, enabling multi-class BCI control without
    relying on visual stimulation.

    **Participants**

    - 21 healthy subjects
    - Location: Berlin Institute of Technology, Germany

    **Recording Details**

    - Equipment: 128-channel Brain Products amplifier
    - Channels: 60 EEG + 2 EOG (62 total)
    - Electrode type: Ag/AgCl electrodes
    - Sampling rate: 1000 Hz (downsampled to 100 Hz for analysis in original paper)
    - Auditory stimuli: 75 ms bandpass filtered white noise (150-8000 Hz), 58 dB
    - Speaker setup: 8 speakers at ear height, 45 degree spacing, ~1 meter distance

    **Data Organization**

    - Subject codes: fce, kw, faz, fcj, fcg, far, faw, fax, fcc, fcm, fas, fch,
      fcd, fca, fcb, fau, fci, fav, fat, fcl, fck
    - Data URL: http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-AMUSE/

    **Event Codes**

    - Target (1): Attended stimulus
    - NonTarget (2): Unattended stimulus

    References
    ----------
    .. [1] Schreuder, M., Rost, T., & Tangermann, M. (2011). Listen, you are
           writing! Speeding up online spelling with a dynamic auditory BCI.
           Frontiers in neuroscience, 5, 112.
           https://doi.org/10.3389/fnins.2011.00112

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 22)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-009",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.3389/fnins.2011.00112",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, event_id = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {"0": {str(ii): raw for ii, raw in enumerate(raws)}}
        return sessions


class BNCI2015_010(MNEBNCI):
    """BNCI 2015-010 RSVP P300 dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 12 subjects during a rapid serial
    visual presentation (RSVP) task. Subjects were instructed to attend to target
    images in a continuous stream of stimuli, eliciting P300 responses.

    References
    ----------
    .. [1] Acqualagna, L., & Blankertz, B. (2013). Gaze-independent BCI-spelling
           using rapid serial visual presentation (RSVP). Clinical Neurophysiology,
           124(5), 901-908.
           https://doi.org/10.1016/j.clinph.2012.12.050

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-010",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1016/j.clinph.2012.12.050",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, event_id = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {"0": {str(ii): raw for ii, raw in enumerate(raws)}}
        return sessions


class BNCI2015_012(MNEBNCI):
    """BNCI 2015-012 PASS2D P300 dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 10 subjects performing a P300
    speller task with a two-dimensional pseudo-random sequence (PASS2D) paradigm.

    Note: Only 10 of the original 12 participants' data is available on the BNCI
    server. Subjects 3 (VPnx) and 6 (VPmg) return HTTP 404 errors.

    References
    ----------
    .. [1] Schreuder, M., Rost, T., & Tangermann, M. (2011). Listen, you are
           writing! Speeding up online spelling with a dynamic auditory BCI.
           Frontiers in neuroscience, 5, 112.
           https://doi.org/10.3389/fnins.2011.00112

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-012",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.3389/fnins.2011.00112",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, event_id = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {"0": {str(ii): raw for ii, raw in enumerate(raws)}}
        return sessions


class BNCI2015_013(MNEBNCI):
    """BNCI 2015-013 Error-Related Potentials dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 6 subjects performing a
    cursor control task with error-related potentials. The dataset includes
    both correct and error responses.

    References
    ----------
    .. [1] Ferrez, P. W., & Millan, J. D. R. (2008). Error-related EEG potentials
           in brain-computer interfaces. Journal of Neural Engineering, 5(1), 62.
           https://doi.org/10.1088/1741-2560/5/1/007

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 7)),
            sessions_per_subject=2,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-013",
            interval=[0, 0.6],
            paradigm="p300",
            doi="10.1109/TNSRE.2010.2053387",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, _ = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {}
        for ii, raw in enumerate(raws):
            sessions[str(ii)] = {"0": raw}
        return sessions
