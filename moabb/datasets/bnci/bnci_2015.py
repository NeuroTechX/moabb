"""BNCI 2015 datasets."""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FilterDetails,
    FrequencyBands,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PerformanceMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
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
    eeg_data = convert_units(data.X, from_unit="uV", to_unit="V")
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
    imagery tasks (right hand vs feet). Each subject participated in multiple
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
    .. [1] Faller, J., Vidaurre, C., Solis-Escalante, T., Neuper, C., & Scherer, R.
           (2012). Autocalibration and recurrent adaptation: Towards a plug and play
           online ERD-BCI. IEEE Transactions on Neural Systems and Rehabilitation
           Engineering, 20(3), 313-319.
           https://doi.org/10.1109/tnsre.2012.2189584

    Notes
    -----
    .. versionadded:: 0.4.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=13,
            channel_types={"eeg": 13},
            montage="10-20",
            hardware="g.tec",
            sensor_type="active electrode",
            reference="Car",
            software="Matlab",
            filters="50 Hz notch",
            sensors=[
                "FC3",
                "C5",
                "CP3",
                "C1",
                "C3",
                "FCz",
                "C2",
                "CPz",
                "Cz",
                "FC4",
                "CP4",
                "C6",
                "C4",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"male": 7, "female": 5},
            age_mean=24.8,
            handedness="all right-handed",
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "feet"],
            trial_duration=11,
            study_design="Two-class motor imagery: sustained right hand movement imagery (palmar grip) versus both feet movement imagery (plantar extension)",
            feedback_type="none",
            stimulus_type="cursor_feedback",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/tnsre.2012.2189584",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filter", "notch filter"],
            filter_details=FilterDetails(
                highpass_hz=0.5,
                lowpass_hz=100,
                bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
                notch_hz=[50],
            ),
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["CSP", "Bandpower", "ERD", "ERS"],
            frequency_bands=FrequencyBands(
                alpha=[8, 13],
                mu=[8, 12],
                analyzed_range=[26.0, 31.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-out",
            evaluation_type=["cross_session"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=80.0,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "gaming", "vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=10,
            trials_context="per_class",
        ),
        data_processed=True,
    )

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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=56,
            channel_types={"eeg": 56},
            montage="fixed set of 56 electrodes",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nose",
            software="Matlab",
            filters={"bandpass": [0.1, 250], "unit": "Hz"},
            impedance_threshold_kohm=15,
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["horizontal", "vertical"],
                other_physiological=["respiration", "gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=21,
            health_status="healthy",
            gender={"male": 13, "female": 8},
            age_mean=34.1,
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=34,
            study_design="Auditory spatial attention BCI spelling task using AMUSE paradigm. Subjects attended to tones from one of six spatial directions while mentally counting target appearances to spell characters via a tw...",
            feedback_type="visual",
            stimulus_type="rsvp",
            stimulus_modalities=["visual", "auditory", "tactile", "multisensory"],
            primary_modality="multisensory",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neulet.2009.06.045",
            associated_paper_doi="10.3389/fnins.2011.00112",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="processed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "low-pass filtering below 40 Hz",
                "down sampling to 100 Hz",
                "baseline correction using 150 ms pre-stimulus data",
            ],
            filter_details=FilterDetails(
                highpass_hz=0.1,
                lowpass_hz=40,
                bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 250.0},
                filter_type="hardware analog band-pass (acquisition), software low-pass (online)",
            ),
            artifact_methods=["ICA"],
            re_reference="nose",
            downsampled_to_hz=100,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["Covariance/Riemannian"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_subject", "cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "speller",
                "wheelchair/navigation",
                "prosthetic",
                "vr_ar",
                "communication",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_trials={"calibration_trials_per_session": 48, "trials_per_direction": 8},
        ),
        data_processed=True,
    )

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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=30,
            channel_types={"eeg": 30},
            montage="10-20",
            hardware="g.tec",
            sensor_type="active electrodes",
            reference="left mastoid",
            ground="right mastoid",
            software="Matlab",
            filters="0.5-100.0 Hz bandpass, 50 Hz notch",
            sensors=[
                "Fp1",
                "Fp2",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "FC5",
                "FC1",
                "FC2",
                "FC6",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "CP5",
                "CP1",
                "CP2",
                "CP6",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "PO9",
                "O1",
                "Oz",
                "O2",
                "PO10",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["vertical"],
                has_emg=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="SCI and stroke",
            gender={"female": 7, "male": 2},
            age_mean=38.5,
            age_min=20,
            age_max=57,
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=5,
            class_labels=["right_hand", "feet", "word", "subtraction", "navigation"],
            trial_duration=10,
            study_design="Five mental tasks: word association, mental subtraction, spatial navigation, right hand motor imagery, both feet motor imagery",
            feedback_type="cue-guided, no online feedback during screening",
            stimulus_type="avatar",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="asynchronous",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0123727",
            repository="BNCI Horizon 2020",
            data_url="https://bnci-horizon-2020.eu/database/data-sets",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw filtered",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filter", "notch filter"],
            filter_details=FilterDetails(
                highpass_hz=0.5,
                lowpass_hz=100,
                bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
                notch_hz=[50],
            ),
            artifact_methods=["ICA"],
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["CSP", "ERD", "ERS", "Covariance/Riemannian"],
            frequency_bands=FrequencyBands(
                mu=[8, 12],
                analyzed_range=[0.5, 100.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["cross_subject", "transfer_learning"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "wheelchair/navigation",
                "prosthetic",
                "drone",
                "gaming",
                "smart_home",
                "vr_ar",
                "communication",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=40,
        ),
        file_format="MAT",
        data_processed=True,
    )

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
            doi="10.1371/journal.pone.0123727",
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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-10",
            hardware="Brain Products",
            sensor_type="active electrode",
            reference="left mastoid",
            software="EEGLab",
            filters={"bandpass": [0.016, 250]},
            impedance_threshold_kohm=20,
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=11,
            health_status="locked_in_syndrome",
            gender={"male": 7, "female": 4},
            age_mean=35.5,
            age_min=21,
            age_max=50,
            handedness="all but one right-handed",
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=35.0,
            study_design="Subjects listened to polyphonic music clips featuring three instruments. They were cued to attend to one particular instrument and mentally count the number of deviants for the cued instrument while i...",
            feedback_type="none (offline analysis)",
            stimulus_type="oddball",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="multisensory",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/11/2/026009",
            repository="GitHub",
            data_url="https://github.com/bbci/bbci_public/blob/master/doc/index.markdown",
            funding=["Grant Nos s"],
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling",
                "lowpass filtering",
                "epoching",
                "baseline correction",
                "artifact rejection",
            ],
            filter_details=FilterDetails(
                lowpass_hz=42,
                filter_type="Chebyshev",
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=250,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["Covariance/Riemannian"],
            frequency_bands=FrequencyBands(
                alpha=[8, 13],
            ),
        ),
        performance=PerformanceMetadata(
            accuracy_percent=91.0,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "vr_ar", "communication"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_blocks=10,
        ),
        data_processed=True,
    )

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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-10",
            hardware="BrainAmp",
            sensor_type="active electrode",
            reference="linked mastoids",
            filters={"bandpass": [0.016, 250]},
            impedance_threshold_kohm=10,
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=16,
            health_status="healthy",
            gender={"male": 10, "female": 6},
            age_mean=23.8,
            bci_experience="experienced",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=9.0,
            study_design="strictly\nremain on the fixation points; this was particularly emphasized\nfor the participants that were not controlled by the eyetracker.",
            feedback_type="online visual feedback (selected symbols displayed in gray at top of screen)",
            stimulus_type="rsvp",
            stimulus_modalities=["visual", "auditory", "tactile"],
            primary_modality="multisensory",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/9/4/045006",
            associated_paper_doi="10.1088/1741-2560/11/2/026009",
            funding=["DFG grant", "grant nos s", "BMBF grant", "grant no MU MU"],
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling",
                "low-pass filtering",
                "epoching",
                "baseline correction",
                "artifact rejection",
                "re-referencing",
            ],
            filter_details=FilterDetails(
                lowpass_hz=42,
                bandpass={"low_cutoff_hz": 0.016, "high_cutoff_hz": 250.0},
                filter_type="Chebyshev",
            ),
            artifact_methods=["ICA"],
            re_reference="linked\nmastoid",
            downsampled_to_hz=200,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["Covariance/Riemannian"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "prosthetic", "vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_processed=True,
    )

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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-10",
            hardware="Brain Products",
            sensor_type="active electrode",
            reference="Car",
            software="EEGLab",
            filters={"bandpass": [0.016, 250]},
            impedance_threshold_kohm=20,
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=13,
            health_status="healthy",
            gender={"male": 8, "female": 5},
            age_mean=30.5,
            age_min=16,
            age_max=45,
            handedness="12 right-handed, 1 left-handed",
            bci_experience="2 experienced (1 author, 1 from precursor study), 11 naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=13.0,
            study_design="Visual speller using geometric shapes with unique colors presented centrally in sequential fashion. Two-level selection: first selecting letter group, then selecting desired letter from group. Non-spa...",
            feedback_type="none",
            stimulus_type="rsvp",
            stimulus_modalities=["visual", "auditory", "tactile", "multisensory"],
            primary_modality="multisensory",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/8/6/066003",
            repository="GitHub",
            data_url="https://github.com/bbci/bbci_public/blob/master/doc/index.markdown",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="continuous EEG with event markers",
            preprocessing_applied=True,
            preprocessing_steps=["BBCI Matlab toolbox preprocessing"],
            filter_details=FilterDetails(
                lowpass_hz=49,
                filter_type="Chebyshev",
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=250,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "SVM", "Shrinkage LDA"],
            feature_extraction=["Covariance/Riemannian"],
            frequency_bands=FrequencyBands(
                alpha=[8, 13],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=71.0,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "prosthetic", "gaming", "vr_ar", "communication"],
            environment="home",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_blocks=10,
        ),
        data_processed=True,
    )

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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=20,
            channel_types={"eeg": 20},
            hardware="Brain Products",
            sensor_type="Ag/AgCl",
            reference="nose",
            software="Matlab",
            filters={"bandpass_hz": [0.1, 250]},
            sensors=[
                "Fz",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "P1",
                "Pz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["horizontal"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=21,
            health_status="healthy",
            gender={"male": 5, "female": 2, "total": 7},
            age_mean=29.5,
            age_min=25,
            age_max=34,
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=33.0,
            study_design="Subjects attended to spatial auditory cues from speakers arranged around them, identifying target directions in an oddball paradigm",
            feedback_type="visual",
            stimulus_type="oddball",
            stimulus_modalities=["visual", "auditory", "tactile"],
            primary_modality="multisensory",
            synchronicity="synchronous",
            mode="both",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0009813",
            associated_paper_doi="10.3389/fnins.2011.00112",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            preprocessing_steps=[
                "low-pass filtering",
                "downsampling",
                "epoching",
                "baseline correction",
                "artifact rejection",
            ],
            filter_details=FilterDetails(
                lowpass_hz=30,
                bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 250.0},
                filter_type="Chebyshev",
                filter_order=8,
            ),
            artifact_methods=["ICA", "trial rejection"],
            re_reference="nose",
            downsampled_to_hz=100,
        ),
        signal_processing=SignalProcessingMetadata(
            feature_extraction=["Covariance/Riemannian"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=70.0,
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "speller",
                "wheelchair/navigation",
                "gaming",
                "vr_ar",
                "communication",
            ],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_trials={"C1000": 2560, "C300": 3750, "C175": 3000, "C300s": 1500},
        ),
        data_processed=True,
    )

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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=63,
            channel_types={"eeg": 63},
            montage="fp12 af34 fz f1-10 fcz fc1-6 ft78 cz c1-6 t78 cpz cp1-6 tp78 pz p1-10 poz po347-10 oz o12",
            hardware="BrainAmp",
            sensor_type="active electrode",
            reference="left mastoid",
            software="Python, MATLAB",
            impedance_threshold_kohm=10,
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"male": 6, "female": 6},
            age_mean=29.17,
            bci_experience="mixed: 3 experienced, 9 naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["counting"],
            trial_duration=1.3,
            study_design="Participants attended to target letters in a rapid serial visual presentation stream of 30 symbols, silently counting occurrences of the target letter to select symbols for spelling",
            feedback_type="visual feedback - classifier selected symbol displayed after each trial",
            stimulus_type="rsvp",
            stimulus_modalities=["visual", "auditory", "tactile"],
            primary_modality="multisensory",
            synchronicity="asynchronous",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.clinph.2012.12.050",
            funding=["BMBF Grant", "Grant Nos s", "Grant No. MU MU", "DFG Grant"],
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            preprocessing_steps=[
                "lowpass filter",
                "downsampling",
                "epoching",
                "baseline correction",
                "artifact rejection",
            ],
            filter_details=FilterDetails(
                lowpass_hz=40,
                filter_type="Chebyshev",
            ),
            artifact_methods=["ICA"],
            re_reference="linked mastoid",
            downsampled_to_hz=200,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["ERD", "ERS", "Covariance/Riemannian"],
            frequency_bands=FrequencyBands(
                alpha=[8, 13],
            ),
        ),
        performance=PerformanceMetadata(
            accuracy_percent=94.8,
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "speller",
                "wheelchair/navigation",
                "prosthetic",
                "gaming",
                "vr_ar",
                "communication",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_repetitions=10,
        ),
        data_processed=True,
    )

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

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=708.0,
            n_channels=63,
            channel_types={"eeg": 63},
            montage="10-20",
            hardware="Brain Products",
            sensor_type="Ag/AgCl",
            reference="nose",
            software="Matlab",
            filters={"bandpass_hz": [0.1, 250]},
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=135,
            health_status="healthy",
            gender={"male": 9, "female": 3},
            age_mean=25.1,
            bci_experience="2 subjects had previous BCI experience",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=1.0,
            study_design="Nine-class auditory oddball task using spatial auditory stimuli varying in pitch (high/medium/low) and direction (left/middle/right). Subjects focused on target stimuli while ignoring non-targets. Spe...",
            feedback_type="visual",
            stimulus_type="rc_speller",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="multisensory",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnins.2011.00099",
            associated_paper_doi="10.3389/fnins.2011.00112",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="processed/epoched for analysis",
            preprocessing_applied=True,
            preprocessing_steps=[
                "analog bandpass filter (0.1-250 Hz)",
                "low-pass filter to 40 Hz",
                "downsampling to 100 Hz",
                "epoching (-150 to 800 ms relative to stimulus onset)",
                "baseline correction (first 150 ms)",
                "artifact rejection (peak-to-peak > 100 μV)",
            ],
            filter_details=FilterDetails(
                highpass_hz=0.1,
                lowpass_hz=40,
                bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 250.0},
                filter_type="analog (initial), then digital low-pass",
            ),
            artifact_methods=["EMG removal", "ICA"],
            re_reference="nose",
            downsampled_to_hz=100,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "speller",
                "wheelchair/navigation",
                "prosthetic",
                "vr_ar",
                "communication",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_trials={"calibration_per_run": 9, "total_calibration": 27},
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-012",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.3389/fnins.2011.00099",
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
    .. [1] Chavarriaga, R., & Millán, J. D. R. (2010). Learning from EEG
           error-related potentials in noninvasive brain-computer interfaces.
           IEEE Trans. Neural Syst. Rehabil. Eng., 18(4), 381-388.
           https://doi.org/10.1109/TNSRE.2010.2053387

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-20",
            hardware="Biosemi ActiveTwo",
            sensor_type="active",
            reference="Car",
            software="EEGLAB",
            filters="1.0-10.0 Hz bandpass",
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=7,
                eog_type=["horizontal"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=6,
            gender={"male": 5, "female": 1},
            age_mean=27.5,
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=2.0,
            study_design="Subjects monitored an autonomous cursor moving toward a target location. The cursor moved correctly 80% of the time and erroneously (opposite direction) 20% of the time. Subjects had no control over t...",
            feedback_type="visual",
            stimulus_type="oddball",
            stimulus_modalities=["visual", "multisensory"],
            primary_modality="multisensory",
            synchronicity="asynchronous",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TNSRE.2010.2053387",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            preprocessing_steps=[
                "spatial filtering (CAR)",
                "bandpass filtering",
                "epoch extraction",
            ],
            filter_details=FilterDetails(
                highpass_hz=1,
                lowpass_hz=10,
                bandpass={"low_cutoff_hz": 1.0, "high_cutoff_hz": 10.0},
            ),
            artifact_methods=["EOG correction", "ICA"],
            re_reference="car",
            downsampled_to_hz=64,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "SVM"],
            feature_extraction=["Covariance/Riemannian"],
            frequency_bands=FrequencyBands(
                analyzed_range=[1.0, 10.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["cross_subject", "cross_session"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=75.81,
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "wheelchair/navigation",
                "prosthetic",
                "gaming",
                "smart_home",
                "vr_ar",
                "communication",
            ],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
        ),
        data_structure=DataStructureMetadata(
            n_trials=64,
            n_blocks=10,
            trials_context="per_run",
        ),
        file_format="MAT",
        data_processed=True,
    )

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
