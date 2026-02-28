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
    ParadigmSpecificMetadata,
    ParticipantMetadata,
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
                "FCz",
                "FC4",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP3",
                "CPz",
                "CP4",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr"],
            ),
            cap_manufacturer="g.tec",
            cap_model="g.GAMMAsys",
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"male": 7, "female": 5},
            age_mean=24.8,
            handedness="all right-handed",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"right_hand": 1, "feet": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "feet"],
            trial_duration=11.0,
            study_design="Two-class motor imagery: sustained right hand movement imagery (palmar grip) versus both feet movement imagery (plantar extension)",
            feedback_type="visual",
            stimulus_type="cursor_feedback",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="training",
            instructions="Relax during reference period (3s), perform sustained kinesthetic movement imagery during activity period. Condition 1 (arrow right): imagine palmar grip with right hand. Condition 2 (arrow down): imagine plantar extension of both feet.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/tnsre.2012.2189584",
            investigators=[
                "Josef Faller",
                "Carmen Vidaurre",
                "Teodoro Solis-Escalante",
                "Christa Neuper",
                "Reinhold Scherer",
            ],
            institution="Graz University of Technology",
            country="Austria",
            publication_year=2012,
            senior_author="Reinhold Scherer",
            contact_info=[
                "josef.faller@tugraz.at",
                "christa.neuper@uni-graz.at",
                "carmen.vidaurre@tu-berlin.de",
            ],
            funding=["FP7 Framework EU Research Project BrainAble (No. 247447)"],
            institution_address="8010 Graz, Austria",
            institution_department="Institute of Knowledge Discovery",
            associated_paper_doi=None,
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
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
            highpass_hz=0.5,
            lowpass_hz=100.0,
            bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
            notch_hz=[50.0],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["logarithmic bandpower", "CSP"],
            frequency_bands={
                "alpha": [10, 13],
                "beta": [16, 24],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-out",
            evaluation_type=["cross_session"],
        ),
        performance={"accuracy_percent": 80.0},
        bci_application=BCIApplicationMetadata(
            applications=["communication", "control"],
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand_palmar_grip", "both_feet_plantar_extension"],
            cue_duration_s=1.25,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=200,
            trials_context="per_session",
            n_trials_per_class={"right_hand": 100, "feet": 100},
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        data_processed=True,
        file_format="gdf",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=2,
            events={"right_hand": 1, "feet": 2},
            code="BNCI2015-001",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1109/tnsre.2012.2189584",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            sampling_rate=256.0,
            n_channels=8,
            channel_types={"eeg": 8},
            montage="standard_1005",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl electrodes",
            reference="nose",
            ground=None,
            software="Matlab",
            filters="hardware analog band-pass filter between 0.1 and 250 Hz",
            line_freq=50.0,
            sensors=[
                "Fz",
                "Cz",
                "P3",
                "Pz",
                "P4",
                "PO7",
                "Oz",
                "PO8",
            ],
            impedance_threshold_kohm=15.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["bipolar"],
                has_emg=None,
                emg_channels=None,
                other_physiological=None,
            ),
            cap_manufacturer="Brain Products",
            cap_model=None,
            electrode_type="Ag/AgCl",
            electrode_material="silver/silver chloride",
        ),
        participants=ParticipantMetadata(
            n_subjects=21,
            health_status="Healthy",
            gender=None,
            age_mean=34.1,
            age_std=11.4,
            age_min=20,
            age_max=57,
            ages=None,
            handedness=None,
            clinical_population=None,
            bci_experience="naive",
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            task_type="auditory_oddball",
            n_classes=6,
            class_labels=[
                "direction_1",
                "direction_2",
                "direction_3",
                "direction_4",
                "direction_5",
                "direction_6",
            ],
            trials_per_class=None,
            trial_duration=None,
            tasks=["spelling", "auditory_attention"],
            study_design="Auditory Multi-class Spatial ERP (AMUSE) paradigm using spatial auditory cues from six speaker locations in azimuth plane. Two-step hex-o-spell like interface for character selection. Subjects mentally count target stimuli from one of six spatial directions.",
            study_domain="communication",
            feedback_type="auditory",
            stimulus_type="spatial_auditory",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            instructions="Focus attention to one target direction and mentally count the number of appearances",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation={
                "soa_ms": "175",
                "stimulus_duration_ms": "40",
                "stimulus_intensity_db": "58",
                "speaker_arrangement": "6 speakers at ear height, evenly distributed in circle with 60° distance, radius 65 cm",
            },
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neulet.2009.06.045",
            description="Auditory BCI speller using spatial cues (AMUSE paradigm) allowing purely auditory communication interface",
            investigators=["Martijn Schreuder", "Thomas Rost", "Michael Tangermann"],
            institution="Berlin Institute of Technology",
            country="Germany",
            repository="BNCI Horizon",
            license="CC-BY-NC-ND-4.0",
            publication_year=2011,
            senior_author="Michael Tangermann",
            contact_info=["schreuder@tu-berlin.de"],
            associated_paper_doi="10.3389/fnins.2011.00112",
            funding=[
                "European ICT Programme Project FP7-224631",
                "European ICT Programme Project FP7-216886",
                "Deutsche Forschungsgemeinschaft (DFG MU 987/3-2)",
                "Bundesministerium fur Bildung und Forschung (BMBF FKZ 01IB001A, 01GQ0850)",
                "FP7-ICT PASCAL2 Network of Excellence ICT-216886",
            ],
            institution_address="Machine Learning Laboratory, Berlin Institute of Technology, FR6-9, Franklinstraße 28/29, 10587 Berlin, Germany",
            institution_department="Machine Learning Laboratory",
            ethics_approval=["Ethics Committee of the Charité University Hospital"],
            acknowledgements="Thomas Denck, David List and Larissa Queda for help with experiments. Klaus-Robert Müller and Benjamin Blankertz for fruitful discussions.",
            keywords=[
                "brain-computer interface",
                "directional hearing",
                "auditory event-related potentials",
                "P300",
                "N200",
                "dynamic subtrials",
            ],
        ),
        sessions_per_subject=2,
        runs_per_session=2,
        sessions=["Session 1", "Session 2"],
        contributing_labs=None,
        n_contributing_labs=1,
        data_processed=True,
        file_format="gdf",
        external_links={
            "source": "http://www.frontiersin.org/neuroprosthetics/10.3389/fnins.2011.00112/abstract"
        },
        tags=Tags(
            pathology=["Healthy"],
            modality=["Auditory"],
            type=["ERP", "P300"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=["low-pass filter", "downsampling", "baselining"],
            highpass_hz=0.1,
            lowpass_hz=40.0,
            bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 40.0},
            filter_type="analog hardware filter for acquisition; low-pass for online",
            artifact_methods=["variance criterium", "peak-to-peak difference criterium"],
            re_reference="nose",
            downsampled_to_hz=100.0,
            epoch_window=[-0.15, None],
            notes="For online use signal was low-pass filtered below 40 Hz and downsampled to 100 Hz. Data baselined using 150 ms pre-stimulus data as reference.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "linear binary classifier"],
            feature_extraction=[
                "spatio-temporal features",
                "r2 coefficient",
                "interval averaging",
            ],
            frequency_bands=None,
            spatial_filters=["shrinkage regularization (Ledoit-Wolf)"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="online",
            cv_folds=None,
            evaluation_type=["online"],
        ),
        performance={
            "accuracy_percent": 77.4,
            "itr_bits_per_min": 2.84,
            "char_per_min_session1": 0.59,
            "char_per_min_session2_max": 1.41,
            "char_per_min_session2_avg": 0.94,
            "itr_session2_avg": 5.26,
            "itr_session2_max": 7.55,
            "success_rate_session1": 76.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type=None,
            code_length=None,
            n_targets=6,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=175.0,
            imagery_tasks=None,
            cue_duration_s=None,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=48,
            n_trials_per_class={"calibration_per_direction": 8},
            n_blocks=None,
            block_duration_s=None,
            trials_context="calibration_phase",
        ),
        abstract="This online study introduces an auditory spelling interface that eliminates the necessity for visual representation. In up to two sessions, a group of healthy subjects (N=21) was asked to use a text entry application, utilizing the spatial cues of the AMUSE paradigm (Auditory Multi-class Spatial ERP). The speller relies on the auditory sense both for stimulation and the core feedback. Without prior BCI experience, 76% of the participants were able to write a full sentence during the first session. By exploiting the advantages of a newly introduced dynamic stopping method, a maximum writing speed of 1.41 char/min (7.55 bits/min) could be reached during the second session (average: 0.94 char/min, 5.26 bits/min).",
        methodology="Participants surrounded by six speakers at ear height in circle (60° spacing, 65 cm radius). Each direction associated with unique combination of tone (base frequency + harmonics) and band-pass filtered noise. Two-step hex-o-spell interface for character selection. Session 1: calibration (48 trials, 8 per direction, 15 iterations each) followed by online spelling with 15 fixed iterations. Session 2: calibration followed by online spelling with dynamic stopping method (4-15 iterations). Spatio-temporal feature extraction using r2 coefficient and interval selection (2-4 intervals for early and late components, 112-224 features total). Linear binary classifier with shrinkage regularization (Ledoit-Wolf). Decision making based on median classifier scores across iterations.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2015-003",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1016/j.neulet.2009.06.045",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            sensor_type="active electrode",
            reference="left and right mastoid",
            ground="left and right mastoid",
            software=None,
            filters="0.5-100 Hz bandpass, 50 Hz notch",
            sensors=[
                "AFz",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "FC3",
                "FCz",
                "FC4",
                "T3",
                "C3",
                "Cz",
                "C4",
                "T4",
                "CP3",
                "CPz",
                "CP4",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO3",
                "PO4",
                "O1",
                "O2",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["horizontal", "vertical"],
            ),
            cap_manufacturer="g.tec",
            electrode_type="g.LADYbird active electrodes",
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="CNS tissue damage",
            gender={"male": 2, "female": 7},
            age_mean=38.0,
            age_std=10.0,
            age_min=20,
            age_max=57,
            handedness="not specified",
            clinical_population="stroke and spinal cord injury",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"math": 1, "letter": 2, "rotation": 3, "count": 4, "baseline": 5},
            paradigm="imagery",
            n_classes=5,
            class_labels=[
                "word_association",
                "mental_subtraction",
                "spatial_navigation",
                "right_hand",
                "feet",
            ],
            trial_duration=11.0,
            study_design="Five mental tasks: word association (WORD), mental subtraction (SUB), spatial navigation (NAV), motor imagery of right hand (HAND), and motor imagery of both feet (FEET). Cue-guided paradigm with 7 seconds of continuous mental imagery per trial.",
            feedback_type="none",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="screening",
            instructions="Participants were asked to continuously perform the specified mental imagery task for 7 seconds. For MI: kinesthetic imagination of movement (e.g., squeezing a rubber ball for hand, dorsiflexion for feet). For WORD: generate words beginning with presented letter. For SUB: successive elementary subtractions. For NAV: spatial navigation.",
            tasks=[
                "word_association",
                "mental_subtraction",
                "spatial_navigation",
                "right_hand_imagery",
                "feet_imagery",
            ],
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0123727",
            investigators=[
                "Reinhold Scherer",
                "Josef Faller",
                "Elisabeth V. C. Friedrich",
                "Eloy Opisso",
                "Ursula Costa",
                "Andrea Kübler",
                "Gernot R. Müller-Putz",
            ],
            institution="Institut Guttmann",
            country="Spain",
            publication_year=2015,
            senior_author="Reinhold Scherer",
            contact_info=["reinhold.scherer@tugraz.at"],
            funding=[
                "FP7 EU Research Projects BrainAble (No. 247447)",
                "ABC (No. 287774)",
                "BackHome (No. 288566)",
            ],
            institution_address="08916 Badalona, Barcelona, Spain",
            institution_department="Institut Universitari de Neurorehabilitació adscrit a la UAB",
            ethics_approval=["Comitè d'Ètica Assistencial de l'Institut Guttman"],
            keywords=[
                "brain-computer interface",
                "motor imagery",
                "mental tasks",
                "EEG",
                "CNS tissue damage",
                "stroke",
                "spinal cord injury",
                "binary classification",
            ],
            repository="BNCI Horizon 2020",
            data_url="https://bnci-horizon-2020.eu/database/data-sets",
            license="CC-BY-NC-ND-4.0",
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        tags=Tags(
            pathology=["Stroke", "Spinal Cord Injury", "CNS Damage"],
            modality=["Motor", "Cognitive"],
            type=["Motor", "Cognitive"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filter", "notch filter", "artifact rejection"],
            highpass_hz=0.5,
            lowpass_hz=100.0,
            bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
            notch_hz=[50],
            artifact_methods=["manual artifact rejection based on EOG"],
            re_reference="left and right mastoid",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["bandpower", "temporal features"],
            frequency_bands={
                "mu": [8, 12],
                "beta": [13, 30],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold cross-validation",
            cv_folds=10,
            evaluation_type=["within_session", "cross_session"],
        ),
        performance={
            "accuracy_percent": 77.0,
            "best_task_pair_GMAC": 77.0,
            "SUB_vs_FEET_GMAC": 77.0,
            "WORD_vs_HAND_GMAC": 70.0,
            "HAND_vs_FEET_GMAC": 64.0,
            "between_day_WORD_vs_HAND_GMAC": 82.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication", "motor_function_restoration"],
            environment="rehabilitation center",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "right_hand",
                "feet",
                "word_association",
                "mental_subtraction",
                "spatial_navigation",
            ],
            imagery_duration_s=7.0,
            cue_duration_s=1.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=40,
            trials_context="per_class_per_day",
            n_blocks=8,
        ),
        data_processed=True,
        file_format="gdf",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
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
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            sampling_rate=200.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-10",
            hardware="Brain Products",
            sensor_type="active electrode",
            reference="left mastoid",
            ground="forehead",
            software=None,
            filters={"bandpass": [0.016, 250]},
            line_freq=50.0,
            impedance_threshold_kohm=20.0,
            auxiliary_channels=None,
            cap_manufacturer="Brain Products",
            cap_model="actiCAP",
            electrode_type="active",
            electrode_material=None,
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPz",
                "Cz",
                "EOGvu",
                "F1",
                "F10",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FT7",
                "FT8",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P10",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP7",
                "TP8",
            ],
        ),
        participants=ParticipantMetadata(
            n_subjects=11,
            health_status="Healthy",
            gender={"male": 7, "female": 4},
            age_mean=28.0,
            age_std=None,
            age_min=21,
            age_max=50,
            ages=None,
            handedness="all but one right-handed",
            clinical_population=None,
            bci_experience="naive",
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="auditory oddball",
            events={"attended_deviant": 1, "unattended_deviant": 2},
            n_classes=3,
            class_labels=["drums/flute", "bass", "keyboard/piano"],
            trials_per_class=None,
            trial_duration=40.0,
            tasks=["selective auditory attention", "deviant counting"],
            study_design="Multi-streamed musical oddball paradigm with three concurrent instruments. Participants attended to one instrument and counted deviants while ignoring the other two instruments. Two music conditions tested: Synth-Pop (bass, drums, keyboard) and Jazz (double-bass, piano, flute).",
            study_domain="auditory BCI",
            feedback_type="none",
            stimulus_type="musical oddball",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="auditory",
            synchronicity="asynchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Attend to cued instrument, count the number of deviants in that instrument, ignore other two instruments, maintain fixation on cross, minimize eye movements",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation={
                "visual_cue": "instrument indication",
                "fixation_cross": "continuous during music playback",
                "music_clips": "40-second polyphonic music",
            },
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/11/2/026009",
            description="Multi-streamed musical oddball paradigm for auditory BCI. Each of three concurrent instruments has its own standard and deviant patterns. Participants selectively attend to one instrument to detect deviants.",
            investigators=[
                "M S Treder",
                "H Purwins",
                "D Miklody",
                "I Sturm",
                "B Blankertz",
            ],
            institution="Technische Universität Berlin",
            country="Germany",
            publication_year=2014,
            senior_author="B Blankertz",
            contact_info=["matthias.treder@tu-berlin.de"],
            funding=[
                "German Bundesministerium für Bildung und Forschung (Grant Nos. 16SV5839 and 01GQ0850)"
            ],
            institution_address="Berlin, Germany",
            institution_department="Neurotechnology Group; Bernstein Focus: Neurotechnology",
            ethics_approval=["Declaration of Helsinki"],
            acknowledgements="We acknowledge financial support by the German Bundesministerium für Bildung und Forschung (Grant Nos. 16SV5839 and 01GQ0850).",
            how_to_acknowledge=None,
            keywords=[
                "brain–computer interface",
                "EEG",
                "auditory",
                "music",
                "attention",
                "oddball paradigm",
                "P300",
            ],
            associated_paper_doi="10.1088/1741-2560/11/2/026009",
            repository="GitHub",
            data_url="https://github.com/bbci/bbci_public/blob/master/doc/index.markdown",
            license="CC-BY-NC-ND-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        sessions=None,
        contributing_labs=[
            "Neurotechnology Group TU Berlin",
            "Bernstein Focus Neurotechnology",
            "Aalborg University Copenhagen",
            "Berlin School of Mind and Brain",
        ],
        n_contributing_labs=4,
        data_processed=True,
        file_format="gdf",
        external_links=None,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Auditory"],
            type=["Perception", "Attention"],
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
            lowpass_hz=42.0,
            filter_type="Chebyshev",
            artifact_methods=["min-max criterion (100 μV threshold on Fp1 or Fp2)"],
            re_reference=None,
            downsampled_to_hz=250.0,
            epoch_window=[-0.2, 1.2],
            notes="Artifact rejection applied only to training set, preserved in test set. Passbands: 42 Hz, stopbands: 49 Hz for Chebyshev filter.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA with shrinkage covariance"],
            feature_extraction=[
                "spatio-temporal features",
                "voltage averaging in time windows",
            ],
            frequency_bands={
                "alpha": [8, 13],
            },
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-clip-out",
            cv_folds=None,
            evaluation_type=["cross_trial"],
        ),
        performance={
            "accuracy_percent": 91.0,
            "binary_classifier_accuracy_synth_pop": 69.25,
            "binary_classifier_accuracy_jazz": 71.47,
            "posterior_probability_accuracy_synth_pop": 91.0,
            "posterior_probability_accuracy_jazz": 91.5,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication", "speller", "message selection"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type=None,
            code_length=None,
            n_targets=3,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=None,
            imagery_tasks=None,
            cue_duration_s=None,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials="3-7 deviants per instrument per clip",
            n_trials_per_class=None,
            n_blocks=10,
            block_duration_s=None,
            trials_context="per_instrument_per_clip",
        ),
        abstract="Polyphonic music (music consisting of several instruments playing in parallel) is an intuitive way of embedding multiple information streams. The different instruments in a musical piece form concurrent information streams that seamlessly integrate into a coherent and hedonistically appealing entity. Here, we explore polyphonic music as a novel stimulation approach for use in a brain–computer interface. In a multi-streamed oddball experiment, we had participants shift selective attention to one out of three different instruments in music audio clips. Each instrument formed an oddball stream with its own specific standard stimuli (a repetitive musical pattern) and oddballs (deviating musical pattern). Contrasting attended versus unattended instruments, ERP analysis shows subject- and instrument-specific responses including P300 and early auditory components. The attended instrument can be classified offline with a mean accuracy of 91% across 11 participants. This is a proof of concept that attention paid to a particular instrument in polyphonic music can be inferred from ongoing EEG, a finding that is potentially relevant for both brain–computer interface and music research.",
        methodology="Participants listened to 40-second polyphonic music clips with three concurrent instruments (Synth-Pop: bass, drums, keyboard; Jazz: double-bass, piano, flute). Each instrument had standard patterns and infrequent deviants (3-7 per clip). Participants were cued to attend to one instrument and count deviants. EEG recorded at 1000 Hz with 64 electrodes, downsampled to 250 Hz, lowpass filtered (Chebyshev, 42 Hz passband), epoched (-200 to 1200 ms), baseline corrected, and artifact rejected. Two classification approaches: (1) general binary classifier and (2) instrument-specific classifiers with posterior probabilities. Features: spatio-temporal (3 time intervals × 63 electrodes = 189 dimensions). LDA with shrinkage covariance. Leave-one-clip-out cross-validation. Main experiment: 10 blocks of 21 clips (7 clips per instrument as target). Total: 3 Synth-Pop mixed blocks, 3 Jazz mixed blocks, 2 Synth-Pop solo blocks, 2 Jazz solo blocks.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 12)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-006",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.1088/1741-2560/11/2/026009",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            sampling_rate=100.0,
            n_channels=63,
            channel_types={"eeg": 63},
            montage="10-10",
            hardware="BrainAmp EEG amplifier",
            sensor_type="active electrode",
            reference="linked mastoids",
            ground="forehead",
            software="Pyff, VisionEgg, MATLAB",
            filters="hardware bandpass filter 0.016–250 Hz",
            sensors=[
                "Fp1",
                "Fp2",
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "Fz",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "F10",
                "FCz",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FT7",
                "FT8",
                "T7",
                "T8",
                "Cz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "TP7",
                "TP8",
                "CPz",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "Pz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "P10",
                "POz",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "Oz",
                "O1",
                "O2",
            ],
            line_freq=50.0,
            impedance_threshold_kohm=10.0,
            cap_manufacturer="Brain Products",
            electrode_type="actiCap active electrode system",
        ),
        participants=ParticipantMetadata(
            n_subjects=16,
            health_status="Healthy",
            gender={"male": 10, "female": 6},
            age_mean=23.8,
            age_min=21,
            age_max=30,
            handedness="normal or corrected-to-normal vision",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 1, "NonTarget": 2},
            paradigm="p300",
            task_type="visual_speller",
            n_classes=2,
            class_labels=["target", "nontarget"],
            trial_duration=30.0,
            study_design="Three different Cake Speller modifications: Overt Cake Speller (gaze toward target), Covert Cake Speller (central fixation, covert attention), Motion Center Speller (foveal stimulation). Two-level selection (group-level and symbol-level) from 30 symbols.",
            study_domain="gaze-independent communication",
            feedback_type="visual",
            stimulus_type="motion VEP (mVEP)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            instructions="Copy-spelling and free-spelling with attention to target symbols. Participants counted moving bar/pattern presentations in target location.",
            stimulus_presentation={
                "soa_ms": "200 ms (Cake Spellers) or 266 ms (Motion Center Speller)",
                "stimulus_duration_ms": "100 ms",
                "isi_ms": "100 ms",
                "repetitions": "10 repetitions per level",
                "total_presentations": "120 per selection (2 levels × 10 repetitions × 6 groups/symbols)",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/9/4/045006",
            description="Exploring motion VEPs for gaze-independent communication",
            investigators=[
                "Sulamith Schaeff",
                "Matthias Sebastian Treder",
                "Bastian Venthur",
                "Benjamin Blankertz",
            ],
            institution="Berlin Institute of Technology",
            institution_department="Neurotechnology Group",
            country="Germany",
            publication_year=2012,
            senior_author="Benjamin Blankertz",
            contact_info=["benjamin.blankertz@tu-berlin.de"],
            keywords=[
                "motion visually evoked potentials",
                "mVEP",
                "BCI",
                "speller",
                "gaze-independent",
                "covert attention",
                "P300",
                "N200",
            ],
            ethics_approval=["Declaration of Helsinki"],
            associated_paper_doi="10.1088/1741-2560/11/2/026009",
            funding=["DFG grant", "grant nos s", "BMBF grant", "grant no MU MU"],
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["P300", "VEP"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling",
                "low-pass filter",
                "baseline correction",
                "artifact rejection",
            ],
            highpass_hz=0.016,
            lowpass_hz=250.0,
            bandpass={"low_cutoff_hz": 0.016, "high_cutoff_hz": 250.0},
            filter_type="hardware bandpass, Chebyshev low-pass for offline",
            artifact_methods=["min-max criterion (70 μV)", "variance criterion"],
            re_reference="linked mastoids",
            downsampled_to_hz=100.0,
            epoch_window=[-0.2, 1.0],
            notes="For offline analysis: downsampled to 200 Hz, low-pass filtered (42 Hz passband, 49 Hz stopband). For online: downsampled to 100 Hz. Artifact rejection: min-max ≥70 μV. Nontarget epochs filtered to avoid overlap with targets (3 preceding and 4 following stimuli must be nontargets).",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA with shrinkage of covariance matrix"],
            feature_extraction=[
                "signed square values of point-biserial correlation coefficients"
            ],
            frequency_bands={
                "analyzed_range": [100.0, 800.0],
            },
            spatial_filters=["LDA spatial filter"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="train on calibration, test on copy-spelling and free-spelling",
            evaluation_type=["within_session"],
        ),
        performance={
            "N200_latency_overt_ms": 164.0,
            "N200_latency_covert_ms": 180.0,
            "N200_latency_motion_center_ms": 198.0,
            "P300_latency_range_ms": "300-500",
            "N200_latency_range_ms": "100-250",
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=6,
            n_repetitions=10,
            isi_ms=100.0,
            soa_ms=200.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=120,
            trials_context="per_selection (2 levels × 10 repetitions × 6 groups/symbols)",
            n_blocks=4,
            block_duration_s=None,
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        sessions=["practice", "calibration", "copy_spelling", "free_spelling"],
        data_processed=True,
        file_format="gdf",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 17)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-007",
            interval=[0, 0.7],
            paradigm="p300",  # Oddball-like paradigm with Target/NonTarget
            doi="10.1088/1741-2560/9/4/045006",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            sampling_rate=250.0,
            n_channels=63,
            channel_types={"eeg": 63},
            montage="10-10",
            hardware="Brain Products actiCAP",
            sensor_type="active electrode",
            reference="left mastoid",
            ground="forehead",
            software=None,
            filters="0.016-250 Hz bandpass",
            sensors=[
                "Fp2",
                "AF3",
                "AF4",
                "Fz",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "F10",
                "FCz",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "T7",
                "T8",
                "Cz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "TP7",
                "TP8",
                "CPz",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "Pz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "P10",
                "POz",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "PO9",
                "PO10",
                "Oz",
                "O1",
                "O2",
                "Iz",
                "I1",
                "I2",
            ],
            line_freq=50.0,
            impedance_threshold_kohm=20.0,
            cap_manufacturer="Brain Products",
        ),
        participants=ParticipantMetadata(
            n_subjects=13,
            health_status="Healthy",
            gender={"male": 8, "female": 5},
            age_mean=27.0,
            age_min=16.0,
            age_max=45.0,
            handedness={"right": 12, "left": 1},
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 1, "NonTarget": 2},
            paradigm="p300",
            n_classes=30,
            class_labels=[
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
                ".",
                ",",
                "space",
                "backspace",
            ],
            trial_duration=30.0,
            study_design="Two-stage visual speller using covert spatial attention and non-spatial feature attention (color and form). Three speller variants tested: Hex-o-Spell (6 discs with size enhancement and unique colors), Cake Speller (6 triangular faces with unique colors), Center Speller (sequential presentation of 6 geometric shapes with unique colors and forms).",
            feedback_type="none",
            stimulus_type="visual_flash",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            instructions="Participants had to strictly fixate the center of the screen and covertly attend to the target symbol. They were instructed to silently count the number of intensifications of the target symbol.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/8/6/066003",
            investigators=["M S Treder", "N M Schmidt", "B Blankertz"],
            institution="Berlin Institute of Technology",
            country="Germany",
            publication_year=2011,
            institution_department="Machine Learning Laboratory",
            keywords=[
                "P300",
                "ERP",
                "BCI",
                "speller",
                "covert attention",
                "feature attention",
                "gaze-independent",
            ],
            repository="GitHub",
            data_url="https://github.com/bbci/bbci_public/blob/master/doc/index.markdown",
            license="CC-BY-NC-ND-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["ERP", "P300"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=["downsampling", "lowpass filter", "baseline correction"],
            highpass_hz=0.016,
            lowpass_hz=49.0,
            bandpass={"low_cutoff_hz": 0.016, "high_cutoff_hz": 250.0},
            filter_type="Chebyshev",
            re_reference="linked mastoids",
            downsampled_to_hz=250.0,
            epoch_window=[-200.0, 800.0],
            notes="For offline ERP analysis: downsampled to 250 Hz, lowpass filtered below 49 Hz using Chebyshev filter (passbands/stopbands: 42/49 Hz). For online classification: downsampled to 100 Hz, no software filter applied. Baseline correction using -200 ms prestimulus interval.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "SLDA"],
            feature_extraction=["ERP components", "P300", "P3"],
            spatial_filters=["shrinkage covariance"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="calibration-test split",
            evaluation_type=["within_session"],
        ),
        performance={
            "accuracy_percent": 92.0,
            "hex_o_spell_accuracy": 88.0,
            "cake_speller_accuracy": 90.0,
            "center_speller_accuracy": 97.0,
            "communication_rate_symbols_per_min": 2.3,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=30,
            n_repetitions=10,
            soa_ms=200.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="60 intensifications per stage (10 sequences × 6 elements)",
            trials_context="per_stage",
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        data_processed=True,
        file_format="gdf",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 14)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-008",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.1088/1741-2560/8/6/066003",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            n_channels=60,
            channel_types={"eeg": 60, "eog": 2},
            montage="10-20",
            hardware="Brain Products 128-channel amplifier",
            sensor_type="Ag/AgCl electrodes",
            reference="nose",
            software="Matlab",
            filters="0.1-250 Hz analog bandpass",
            sensors=[],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["bipolar"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="Healthy",
            gender={"male": 6, "female": 4},
            age_mean=30.3,
            age_min=22,
            age_max=55,
            handedness="unknown",
            bci_experience="mixed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 1, "NonTarget": 2},
            paradigm="p300",
            task_type="oddball",
            n_classes=5,
            class_labels=[
                "location_1",
                "location_2",
                "location_3",
                "location_7",
                "location_8",
            ],
            trial_duration=0.8,
            tasks=["spatial_auditory_oddball"],
            study_design="Offline auditory oddball task using spatial location of auditory stimuli as discriminating cue. Frontal five speakers used (speakers 1,2,3,7,8) with 45 degree spacing. Three conditions tested: C300 (300ms ISI), C175 (175ms ISI), C300s (300ms ISI, single speaker). Each stimulus was unique 40ms complex sound from bandpass filtered white noise with tone overlay.",
            study_domain="BCI",
            feedback_type="none",
            stimulus_type="auditory_spatial",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Subjects asked to mentally count target stimulations or respond by keypress (condition Cr). Minimize eye movements and muscle contractions. Target direction indicated prior to each block visually and by presenting stimulus from that location.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0009813",
            description="A new auditory multi-class brain-computer interface paradigm using spatial hearing as an informative cue",
            investigators=[
                "Martijn Schreuder",
                "Benjamin Blankertz",
                "Michael Tangermann",
            ],
            institution="Berlin Institute of Technology",
            country="Germany",
            publication_year=2010,
            senior_author="Michael Tangermann",
            contact_info=["martijn@cs.tu-berlin.de"],
            funding=[
                "European ICT Programme Project FP7-224631",
                "European ICT Programme Project FP7-216886",
                "Deutsche Forschungsgemeinschaft (DFG) MU 987/3-1",
                "Bundesministerium für Bildung und Forschung (BMBF) FKZ 01IB001A",
                "Bundesministerium für Bildung und Forschung (BMBF) FKZ 01GQ0850",
                "FP7-ICT PASCAL2 Network of Excellence ICT-216886",
            ],
            institution_address="Berlin, Germany",
            institution_department="Machine Learning Department",
            ethics_approval=[
                "Ethics Committee of the Charité University Hospital (number EA4/073/09)"
            ],
            keywords=[
                "auditory BCI",
                "P300",
                "spatial hearing",
                "multi-class",
                "oddball paradigm",
            ],
            associated_paper_doi="10.3389/fnins.2011.00112",
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Auditory"],
            type=["P300"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=[
                "bandpass filter",
                "notch filter",
                "downsampling",
                "artifact rejection",
            ],
            highpass_hz=0.1,
            lowpass_hz=250.0,
            bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 250.0},
            notch_hz=[50],
            filter_type="Chebyshev II order 8 (for visual inspection: 30 Hz pass, 42 Hz stop, 50 dB damping)",
            artifact_methods=["threshold-based artifact rejection"],
            re_reference="nose",
            downsampled_to_hz=100.0,
            epoch_window=[-0.15, 0.8],
            notes="Raw data acquired at 1000 Hz. For visual inspection: low-pass filtered with order 8 Chebyshev II filter (30 Hz pass, 42 Hz stop, 50 dB damping) applied forward and backward to minimize phase shifts, then downsampled to 100 Hz. For classification: same filter applied causally (forward only) for online portability. Artifact rejection used simple threshold method: subtrials with deflection >70 µV over ocular channels compared to baseline were rejected.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["ROC-separability-index"],
            frequency_bands={
                "analyzed_range": [0.1, 250.0],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross-validation",
            evaluation_type=["offline"],
        ),
        performance={
            "accuracy_percent": 90.0,
            "itr_bits_per_min": 17.39,
            "best_subject_itr_bits_per_min": 25.20,
            "best_subject_accuracy_percent": 100.0,
            "c300s_accuracy_percent": 70.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=5,
            n_repetitions=15,
            isi_ms=300.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="varied by condition",
            trials_context="BCI experiments: C300 (50 trials × 75 subtrials = 3750 subtrials), C175 (40 trials × 75 subtrials = 3000 subtrials), C300s (20 trials × 75 subtrials = 1500 subtrials). Physiological experiments: C1000 (32 trials × 80 subtrials = 2560 subtrials), Cr (576-768 subtrials)",
            n_blocks=50,
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        data_processed=True,
        file_format="gdf",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 22)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-009",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.3389/fnins.2011.00112",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            sampling_rate=200.0,
            n_channels=63,
            channel_types={"eeg": 63},
            montage="10-20",
            hardware="BrainAmp amplifiers",
            sensor_type="active electrode",
            reference="left mastoid",
            software="Python with Pyff framework",
            filters="lowpass Chebyshev filter up to 40 Hz",
            sensors=[
                "Fp1",
                "Fp2",
                "AF3",
                "AF4",
                "Fz",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "F10",
                "FCz",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FT7",
                "FT8",
                "Cz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "T7",
                "T8",
                "CPz",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "TP7",
                "TP8",
                "Pz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "P10",
                "POz",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "PO9",
                "PO10",
                "Oz",
                "O1",
                "O2",
            ],
            line_freq=50.0,
            impedance_threshold_kohm=10.0,
            cap_manufacturer="Brain Products",
            cap_model="actiCap",
            electrode_type="active electrode",
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="Healthy",
            gender={"male": 6, "female": 6},
            age_mean=29.17,
            age_std=8.4,
            age_min=24,
            age_max=55,
            handedness="all right-handed",
            bci_experience="mixed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="spelling",
            n_classes=2,
            class_labels=["target", "non_target"],
            trial_duration=46.5,
            study_design="RSVP (Rapid Serial Visual Presentation) BCI speller where 30 symbols are presented one-by-one in random order at the center of the screen. Three conditions tested: NoColor 116ms SOA, Color 116ms SOA, and Color 83ms SOA. Colors used to facilitate discrimination.",
            feedback_type="visual",
            stimulus_type="RSVP letters",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            instructions="Participants fixate center of screen, concentrate on target letter, silently count its occurrences. Avoid blinking during visual presentation.",
            events={"target": 1, "non_target": 29},
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.clinph.2012.12.050",
            investigators=["Laura Acqualagna", "Benjamin Blankertz"],
            institution="Berlin Institute of Technology",
            country="Germany",
            publication_year=2013,
            senior_author="Benjamin Blankertz",
            contact_info=[
                "laura.acqualagna@tu-berlin.de",
                "benjamin.blankertz@tu-berlin.de",
            ],
            institution_department="Machine Learning Laboratory; Neurotechnology Group",
            ethics_approval=[
                "Study performed in accordance with the declaration of Helsinki"
            ],
            keywords=[
                "Brain Computer Interfaces",
                "RSVP",
                "ERPs",
                "Speller",
                "P300",
                "N2",
                "gaze-independent",
            ],
            funding=["BMBF Grant", "Grant Nos s", "Grant No. MU MU", "DFG Grant"],
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["ERP"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=[
                "lowpass filter",
                "downsampling",
                "baseline correction",
                "artifact rejection",
            ],
            lowpass_hz=40.0,
            filter_type="Chebyshev",
            filter_order="passband up to 40 Hz, stopband starting at 49 Hz",
            artifact_methods=[
                "min-max criterion for eye movement rejection (75 µV on F9, Fz, F10, AF3, AF4)",
                "broadband power rejection (5-40 Hz)",
            ],
            re_reference="linked mastoids (offline)",
            downsampled_to_hz=200.0,
            epoch_window=[-0.1, 1.2],
            notes="Baseline correction on pre-stimulus interval (116ms for 116ms SOA, 83/2ms for 83ms SOA). Non-target epochs excluded if 3 preceding or following symbols were targets.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA with shrinkage"],
            feature_extraction=[
                "spatio-temporal features",
                "averaged voltages within time windows",
            ],
            frequency_bands={
                "alpha": [7, 13],
            },
            spatial_filters=[
                "55 channels used for classification (all except Fp1,2, AF3,4, F9,10, FT7,8)"
            ],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="calibration/test split",
            evaluation_type=["within_session"],
        ),
        performance={
            "accuracy_percent": 94.8,
            "mean_spelling_rate_symb_per_min": 1.43,
            "trial_duration_116ms_SOA_s": 46.5,
            "trial_duration_83ms_SOA_s": 36.6,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=30,
            n_repetitions=10,
            soa_ms=116.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="10 sequences of 30 symbols",
            trials_context="per sequence",
            n_blocks=3,
        ),
        sessions_per_subject=3,
        runs_per_session=2,
        sessions=["calibration", "copy-spelling", "free-spelling"],
        data_processed=True,
        file_format="EEG",
        abstract="A Brain Computer Interface (BCI) speller using rapid serial visual presentation (RSVP) paradigm for gaze-independent mental typewriting. Twelve healthy participants successfully operated the RSVP speller with mean online spelling rate of 1.43 symb/min and mean symbol selection accuracy of 94.8%. The RSVP speller does not require gaze shifts and can be operated by non-spatial visual attention, making it suitable for patients with impaired oculo-motor control.",
        methodology="Three experimental conditions tested (NoColor 116ms, Color 116ms, Color 83ms SOA). Each condition included calibration, copy-spelling, and free-spelling phases. Vocabulary of 30 symbols presented one-by-one at screen center in pseudo-random order. EEG recorded at 1000 Hz with 63 channels, downsampled to 200 Hz for ERP analysis. Classification using LDA with shrinkage on spatio-temporal features from 5 individually selected time windows. Symbol selection based on averaged classifier output across 10 sequences.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-010",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1016/j.clinph.2012.12.050",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            sampling_rate=250.0,
            n_channels=63,
            channel_types={"eeg": 63},
            montage="10-20",
            hardware="Brain Products",
            sensor_type="wet Ag/AgCl electrodes",
            reference="nose",
            ground=None,
            software="Matlab",
            filters="0.1-250 Hz analog bandpass, then 40 Hz lowpass",
            line_freq=50.0,
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPz",
                "Cz",
                "F1",
                "F10",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FT7",
                "FT8",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P10",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP7",
                "TP8",
            ],
            impedance_threshold_kohm=None,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=1,
                eog_type=None,
                has_emg=None,
                emg_channels=None,
                other_physiological=None,
            ),
            cap_manufacturer="EasyCap GmbH",
            cap_model="Fast'n Easy Cap",
            electrode_type="wet Ag/AgCl electrodes",
            electrode_material="Ag/AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="Healthy",
            gender={"male": 9, "female": 3},
            age_mean=25.1,
            age_std=None,
            age_min=21,
            age_max=34,
            ages=[26, 21, 25, 23, 34, 23, 23, 24, 24, 25, 29, 24],
            handedness=None,
            clinical_population=None,
            bci_experience="mostly naive",
            sexes=["m", "m", "w", "w", "m", "m", "m", "m", "m", "m", "m", "w"],
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 1, "NonTarget": 2},
            paradigm="p300",
            task_type="auditory ERP speller",
            n_classes=9,
            class_labels=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            trials_per_class=None,
            trial_duration=None,
            tasks=["text spelling", "counting task"],
            study_design="Nine-class auditory ERP paradigm with predictive text entry system (PASS2D). Users focus attention on two-dimensional auditory stimuli varying in pitch (high/medium/low) and direction (left/middle/right) presented via headphones.",
            study_domain="communication",
            feedback_type="visual",
            stimulus_type="auditory tones",
            stimulus_modalities=["auditory", "visual"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            instructions="Focus on target stimuli while ignoring all non-target stimuli. Minimize eye movements and muscle artifacts. Count targets during calibration. Spell sentences during online phase.",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation=None,
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnins.2011.00099",
            description="A novel 9-class auditory ERP paradigm driving a predictive text entry system",
            investigators=[
                "Johannes Höhne",
                "Martijn Schreuder",
                "Benjamin Blankertz",
                "Michael Tangermann",
            ],
            institution="Berlin Institute of Technology",
            country="Germany",
            data_url=None,
            publication_year=2011,
            senior_author="Johannes Höhne",
            contact_info=["j.hoehne@tu-berlin.de"],
            funding=None,
            institution_address="Franklinstr. 28/19, 10587 Berlin, Germany",
            institution_department="Machine Learning Laboratory",
            ethics_approval=None,
            acknowledgements=None,
            how_to_acknowledge=None,
            keywords=[
                "brain–computer interface",
                "BCI",
                "auditory ERP",
                "P300",
                "N200",
                "spatial auditory stimuli",
                "T9",
                "user-centered design",
            ],
            associated_paper_doi="10.3389/fnins.2011.00112",
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        sessions=["session_1"],
        contributing_labs=["Berlin Institute of Technology", "Fraunhofer FIRST"],
        n_contributing_labs=2,
        data_processed=True,
        file_format="gdf",
        external_links=None,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Auditory"],
            type=["ERP", "P300"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered and downsampled",
            preprocessing_applied=True,
            preprocessing_steps=[
                "analog bandpass filter",
                "lowpass filter",
                "downsampling",
                "artifact rejection",
            ],
            highpass_hz=0.1,
            lowpass_hz=40.0,
            bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 250.0},
            filter_type="analog bandpass then digital lowpass",
            artifact_methods=["threshold rejection"],
            re_reference="nose",
            downsampled_to_hz=100.0,
            epoch_window=[-0.15, 0.8],
            notes="Epochs with peak-to-peak voltage difference exceeding 100 μV in any channel were rejected during calibration. No artifact correction applied in online runs.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["FDA", "Fisher discriminant analysis"],
            feature_extraction=["mean amplitude in discriminative intervals"],
            frequency_bands=None,
            spatial_filters=["shrinkage regularization"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross-validation",
            cv_folds=None,
            evaluation_type=["within_session"],
        ),
        performance={
            "accuracy_percent": 72.5,
            "itr_bits_per_min": 3.4,
            "characters_per_minute": 0.8,
            "spelling_speed_chars_per_min": 0.8,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            stimulus_frequencies_hz=[708.0, 524.0, 380.0],
            frequency_resolution_hz=None,
            code_type=None,
            code_length=None,
            n_targets=9,
            n_repetitions=15,
            isi_ms=125.0,
            soa_ms=225.0,
            imagery_tasks=None,
            cue_duration_s=None,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=27,
            n_trials_per_class=None,
            n_blocks=None,
            block_duration_s=None,
            trials_context="total across all calibration runs (3 runs × 9 trials per run)",
        ),
        abstract="Brain–computer interfaces (BCIs) based on event related potentials (ERPs) strive for offering communication pathways which are independent of muscle activity. While most visual ERP-based BCI paradigms require good control of the user's gaze direction, auditory BCI paradigms overcome this restriction. The present work proposes a novel approach using auditory evoked potentials for the example of a multiclass text spelling application. To control the ERP speller, BCI users focus their attention to two-dimensional auditory stimuli that vary in both, pitch (high/medium/low) and direction (left/middle/right) and that are presented via headphones. The resulting nine different control signals are exploited to drive a predictive text entry system. It enables the user to spell a letter by a single nine-class decision plus two additional decisions to confirm a spelled word. This paradigm – called PASS2D – was investigated in an online study with 12 healthy participants. Users spelled with more than 0.8 characters per minute on average (3.4 bits/min) which makes PASS2D a competitive method. It could enrich the toolbox of existing ERP paradigms for BCI end users like people with amyotrophic lateral sclerosis disease in a late stage.",
        methodology="Participants performed a single session lasting 3-4 hours consisting of calibration phase and online spelling task. Calibration: 3 runs (plus 1 practice run), each with 9 trials covering all 9 stimuli as targets. Each trial had 13-14 pseudo-random sequences of all 9 auditory stimuli (108 subtrials total, 12 target + 96 non-target). Online spelling: 2 runs spelling German sentences using T9-style predictive text system with 9-class decisions. Each trial consisted of 135 subtrials (15 iterations of 9 stimuli). Binary classification using linear FDA with shrinkage regularization on 2-4 amplitude values per channel from discriminative intervals (N200 at 230-300ms and P300 at 350+ ms). Multiclass decision based on one-sided t-test with unequal variances across 15 classifier outputs per key.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-012",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.3389/fnins.2011.00099",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            montage="extended 10-20",
            hardware="Biosemi ActiveTwo",
            sensor_type="active electrode",
            reference="CAR",
            software="Matlab",
            filters="1-10 Hz band-pass filter",
            sensors=[
                "Fp1",
                "AF7",
                "AF3",
                "F1",
                "F3",
                "F5",
                "F7",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "C1",
                "C3",
                "C5",
                "T7",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "P1",
                "P3",
                "P5",
                "P7",
                "P9",
                "PO7",
                "PO3",
                "O1",
                "Iz",
                "Oz",
                "POz",
                "Pz",
                "CPz",
                "Fpz",
                "Fp2",
                "AF8",
                "AF4",
                "AFz",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT8",
                "FC6",
                "FC4",
                "FC2",
                "FCz",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP8",
                "CP6",
                "CP4",
                "CP2",
                "P2",
                "P4",
                "P6",
                "P8",
                "P10",
                "PO8",
                "PO4",
                "O2",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["horizontal"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=6,
            health_status="Healthy",
            gender={"male": 5, "female": 1},
            age_mean=27.83,
            age_std=2.23,
            handedness="not reported",
            bci_experience="not reported",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="monitoring",
            events={
                "correct_left": 5,
                "correct_right": 10,
                "error_left": 6,
                "error_right": 9,
            },
            n_classes=2,
            class_labels=["error", "correct"],
            trial_duration=2.0,
            study_design="Error-related potential (ErrP) monitoring task where subjects observe a cursor moving towards a target. The cursor moves autonomously with 20% or 40% error probability. Subjects monitor performance without control.",
            feedback_type="visual",
            stimulus_type="cursor_movement",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=True,
            instructions="Subjects seat in front of a computer screen and monitor a moving cursor (green square) and target location (blue for left, red for right). No control over cursor movement, only assess whether it performs properly. Fixate center of screen.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TNSRE.2010.2053387",
            description="Dataset on EEG error-related potentials (ErrPs) elicited when users monitor the behavior of an external autonomous agent. One of the first studies showing that error correlates can be observed and decoded during monitoring of external agents without user control.",
            investigators=["Ricardo Chavarriaga", "José del R. Millán"],
            institution="Ecole Polytechnique Fédérale de Lausanne",
            institution_department="Defitech Chair in Brain-Machine Interface, CNBI, Center for Neuroprosthetics",
            country="Switzerland",
            publication_year=2010,
            senior_author="José del R. Millán",
            contact_info=["ricardo.chavarriaga@epfl.ch", "jose.millan@epfl.ch"],
            funding=["EC under Contract BACS FP6-IST-027140"],
            keywords=[
                "error-related potentials",
                "ErrP",
                "brain-computer interface",
                "reinforcement learning",
                "monitoring",
                "error detection",
            ],
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Cognitive"],
            type=["ErrP"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=["spatial filter (CAR)", "bandpass filter", "epoching"],
            highpass_hz=1.0,
            lowpass_hz=10.0,
            bandpass={"low_cutoff_hz": 1.0, "high_cutoff_hz": 10.0},
            filter_type="band-pass",
            artifact_methods=["horizontal EOG computation (F7-F8)"],
            re_reference="CAR",
            notes="HEOG computed as difference between F7 and F8. Data spatially filtered using common average reference (CAR) then band-pass filtered 1-10 Hz. Epochs extracted for erroneous and correct cursor movements.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Gaussian classifier"],
            feature_extraction=["event-related potentials"],
            frequency_bands={
                "analyzed_range": [1.0, 10.0],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="train-test split",
            evaluation_type=["cross_session"],
        ),
        performance={
            "accuracy_percent": 75.8,
            "correct_recognition_rate": 63.2,
            "error_recognition_rate": 75.8,
        },
        bci_application=BCIApplicationMetadata(
            applications=["error correction", "adaptive systems", "shared autonomy"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=2,
        ),
        data_structure=DataStructureMetadata(
            n_trials="~50 trials per block, ~64 trials per block for error_prob=0.20",
            n_blocks=10,
            block_duration_s=180.0,
            trials_context="per_block",
        ),
        sessions_per_subject=20,
        runs_per_session=1,
        data_processed=False,
        file_format="matlab",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 7)),
            sessions_per_subject=20,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-013",
            interval=[0, 0.6],
            paradigm="p300",
            doi="10.1109/TNSRE.2010.2053387",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, _ = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {}
        for ii, raw in enumerate(raws):
            sessions[str(ii)] = {"0": raw}
        return sessions
