"""BNCI 2014 datasets."""

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
    BNCI_URL,
    MNEBNCI,
    _convert_mi,
    _convert_run_p300_sl,
    _enrich_run_with_metadata,
    _finalize_raw,
    data_path,
)
from .utils import validate_subject


_map = {"T": "train", "E": "test"}


@verbose
def _load_data_001_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2014 dataset."""
    validate_subject(subject, 9, "BNCI2014-001")

    # fmt: off
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
        "EOG1", "EOG2", "EOG3",
    ]
    # fmt: on
    ch_types = ["eeg"] * 22 + ["eog"] * 3

    sessions = {}
    filenames = []
    for session_idx, r in enumerate(["T", "E"]):
        url = "{u}001-2014/A{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        filenames += filename
        if only_filenames:
            continue
        runs, ev = _convert_mi(
            filename[0],
            ch_names,
            ch_types,
            dataset_code="BNCI2014-001",
            subject_id=subject,
        )
        # FIXME: deal with run with no event (1:3) and name them
        sessions[f"{session_idx}{_map[r]}"] = {
            str(ii): run for ii, run in enumerate(runs)
        }
    if only_filenames:
        return filenames
    return sessions


@verbose
def _load_data_002_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 002-2014 dataset."""
    validate_subject(subject, 14, "BNCI2014-002")

    runs = []
    filenames = []
    for r in ["T", "E"]:
        url = "{u}002-2014/S{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)[0]
        filenames.append(filename)
        if only_filenames:
            continue
        # FIXME: electrode position and name are not provided directly.
        raws, _ = _convert_mi(
            filename, None, ["eeg"] * 15, dataset_code="BNCI2014-002", subject_id=subject
        )
        runs.extend(zip([r] * len(raws), raws))
    if only_filenames:
        return filenames
    runs = {f"{ii}{_map[r]}": run for ii, (r, run) in enumerate(runs)}
    return {"0": runs}


@verbose
def _load_data_004_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 004-2014 dataset."""
    validate_subject(subject, 9, "BNCI2014-004")

    ch_names = ["C3", "Cz", "C4", "EOG1", "EOG2", "EOG3"]
    ch_types = ["eeg"] * 3 + ["eog"] * 3

    sessions = []
    filenames = []
    for r in ["T", "E"]:
        url = "{u}004-2014/B{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)[0]
        filenames.append(filename)
        if only_filenames:
            continue
        raws, _ = _convert_mi(
            filename, ch_names, ch_types, dataset_code="BNCI2014-004", subject_id=subject
        )
        sessions.extend(zip([r] * len(raws), raws))

    if only_filenames:
        return filenames
    sessions = {f"{ii}{_map[r]}": {"0": run} for ii, (r, run) in enumerate(sessions)}
    return sessions


@verbose
def _load_data_008_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 008-2014 dataset."""
    validate_subject(subject, 8, "BNCI2014-008")

    url = "{u}008-2014/A{s:02d}.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]
    run = loadmat(filename, struct_as_record=False, squeeze_me=True)["data"]
    raw, event_id = _convert_run_p300_sl(run, verbose=verbose)

    # Enrich with BNCI2014-008 specific metadata (age, gender, ALSfrs, onsetALS)
    _enrich_run_with_metadata(raw, run, "BNCI2014-008", subject)

    sessions = {"0": {"0": raw}}

    return sessions


@verbose
def _load_data_009_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 009-2014 dataset."""
    validate_subject(subject, 10, "BNCI2014-009")

    # FIXME there is two type of speller, grid speller and geo-speller.
    # we load only grid speller data
    url = "{u}009-2014/A{s:02d}S.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)["data"]
    sess = []
    event_id = {}
    for run in data:
        raw, ev = _convert_run_p300_sl(run, verbose=verbose)
        _finalize_raw(raw, "BNCI2014-009", subject)
        # Raw EEG data are scaled by a factor 10.
        # See https://github.com/NeuroTechX/moabb/issues/275
        raw._data[:16, :] /= 10.0
        sess.append(raw)
        event_id.update(ev)

    sessions = {}
    for i, sessi in enumerate(sess):
        sessions[str(i)] = {"0": sessi}

    return sessions


@depreciated_alias("BNCI2014001", "1.1")
class BNCI2014_001(MNEBNCI):
    """BNCI 2014-001 Motor Imagery dataset.

    Dataset IIa from BCI Competition 4 [1]_.

    **Dataset Description**

    This data set consists of EEG data from 9 subjects.  The cue-based BCI
    paradigm consisted of four different motor imagery tasks, namely the imag-
    ination of movement of the left hand (class 1), right hand (class 2), both
    feet (class 3), and tongue (class 4).  Two sessions on different days were
    recorded for each subject.  Each session is comprised of 6 runs separated
    by short breaks.  One run consists of 48 trials (12 for each of the four
    possible classes), yielding a total of 288 trials per session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
    were used to record the EEG; the montage is shown in Figure 3 left.  All
    signals were recorded monopolarly with the left mastoid serving as
    reference and the right mastoid as ground. The signals were sampled with.
    250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
    the amplifier was set to 100 uV . An additional 50 Hz notch filter was
    enabled to suppress line noise

    References
    ----------
    .. [1] Tangermann, M., Muller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
           Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
           and Nolte, G., 2012. Review of the BCI competition IV.
           Frontiers in neuroscience, 6, p.55.

    Notes
    -----
    .. versionadded:: 0.4.0

    This is one of the most widely used motor imagery datasets in BCI research,
    commonly referred to as "BCI Competition IV Dataset 2a". It serves as a
    standard benchmark for 4-class motor imagery classification algorithms.

    The dataset is particularly useful for:

    - Multi-class motor imagery classification (4 classes)
    - Transfer learning studies (9 subjects, 2 sessions each)
    - Cross-session variability analysis

    See Also
    --------
    BNCI2014_004 : BCI Competition 2008 2-class motor imagery (Dataset B)
    BNCI2003_004 : BCI Competition III 2-class motor imagery

    Examples
    --------
    >>> from moabb.datasets import BNCI2014_001
    >>> dataset = BNCI2014_001()
    >>> dataset.subject_list
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=22,
            channel_types={"eeg": 22},
            montage="10-20",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="Car",
            ground="mastoid",
            software="BCI2000",
            filters="50 Hz notch",
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
                "P2",
                "POz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=3,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="healthy",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["right_hand", "tongue", "left_hand", "feet"],
            trial_duration=5.0,
            study_design="Four-class motor imagery: left hand, right hand, both feet, tongue",
            feedback_type="none",
            stimulus_type="cursor_feedback",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="multisensory",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnins.2012.00055",
            funding=["Grant R31- R31-"],
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="minimally preprocessed (online filtered)",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filtering", "notch filtering"],
            filter_details=FilterDetails(
                highpass_hz=0.5,
                lowpass_hz=100,
                bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
                notch_hz=[50],
                filter_type="Chebyshev",
                filter_order=10,
            ),
            artifact_methods=["trial rejection", "ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "LDA",
                "SVM",
                "Neural Network",
                "Shrinkage LDA",
                "Naive Bayes",
                "CCA",
            ],
            feature_extraction=[
                "CSP",
                "FBCSP",
                "Bandpower",
                "ERD",
                "ERS",
                "PSD",
                "Wavelet",
                "Time-Frequency",
                "AR",
                "ICA",
            ],
            frequency_bands=FrequencyBands(
                alpha=[8, 13],
                mu=[8, 12],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold",
            cv_folds=5,
            evaluation_type=["cross_subject"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=62.0,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["prosthetic", "vr_ar", "communication"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=288,
            n_blocks=3,
            trials_context="total",
        ),
        file_format="MAT",
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            code="BNCI2014-001",
            interval=[2, 6],
            paradigm="imagery",
            doi="10.3389/fnins.2012.00055",
        )


@depreciated_alias("BNCI2014002", "1.1")
class BNCI2014_002(MNEBNCI):
    """BNCI 2014-002 Motor Imagery dataset.

    Motor Imagery Dataset from [1]_.

    **Dataset description**

    The session consisted of eight runs, five of them for training and three
    with feedback for validation.  One run was composed of 20 trials.  Taken
    together, we recorded 50 trials per class for training and 30 trials per
    class for validation.  Participants had the task of performing sustained (5
    seconds) kinaesthetic motor imagery (MI) of the right hand and of the feet
    each as instructed by the cue. At 0 s, a white colored cross appeared on
    screen, 2 s later a beep sounded to catch the participant's attention. The
    cue was displayed from 3 s to 4 s. Participants were instructed to start
    with MI as soon as they recognized the cue and to perform the indicated MI
    until the cross disappeared at 8 s. A rest period with a random length
    between 2 s and 3 s was presented between trials. Participants did not
    receive feedback during training.  Feedback was presented in form of a
    white
    coloured bar-graph.  The length of the bar-graph reflected the amount of
    correct classifications over the last second.  EEG was measured with a
    biosignal amplifier and active Ag/AgCl electrodes (g.USBamp, g.LADYbird,
    Guger Technologies OG, Schiedlberg, Austria) at a sampling rate of 512 Hz.
    The electrodes placement was designed for obtaining three Laplacian
    derivations.  Center electrodes at positions C3, Cz, and C4 and four
    additional electrodes around each center electrode with a distance of 2.5
    cm, 15 electrodes total.  The reference electrode was mounted on the left
    mastoid and the ground electrode on the right mastoid.  The 13 participants
    were aged between 20 and 30 years, 8 naive to the task, and had no known
    disabilities.

    References
    ----------
    .. [1] Scherer, R., Faller, J., Balderas, D., Friedrich, E. V., &
           Müller-Putz, G. (2015). Brain-computer interfacing: more than the
           sum of its parts. Soft Computing, 19(11), 3173-3186.
           https://doi.org/10.1007/s00500-012-0895-4

    Notes
    -----
    .. versionadded:: 0.4.0

    See Also
    --------
    BNCI2014_001 : 4-class motor imagery (BCI Competition IV Dataset 2a)
    BNCI2014_004 : 2-class motor imagery (Dataset B)
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="laplacian",
            hardware="g.tec",
            sensor_type="Ag/AgCl",
            reference="left mastoid",
            ground="right mastoid",
            filters="50 Hz notch",
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
                "AF7",
                "AF8",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            health_status="healthy",
            gender={"female": 5, "male": 9},
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "left_hand", "feet"],
            trial_duration=8.0,
            study_design="MI",
            feedback_type="none",
            stimulus_type="cursor_feedback",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1515/bmt-2014-0117",
            associated_paper_doi="10.1007/s00500-012-0895-4",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw with online filtering applied",
            preprocessing_applied=True,
            preprocessing_steps=[
                "Bandpass filtering",
                "Notch filtering",
                "Automated outlier rejection",
                "Laplacian spatial filtering",
                "DFT for PSD estimation",
            ],
            filter_details=FilterDetails(
                highpass_hz=0.1,
                lowpass_hz=200,
                bandpass="0.1-200 Hz",
                notch_hz=[50],
                filter_type="Butterworth",
                filter_order=8,
            ),
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "SVM", "Random Forest", "Shrinkage LDA"],
            feature_extraction=["CSP", "FBCSP", "ERD", "PSD"],
            frequency_bands=FrequencyBands(
                alpha=[8, 13],
                analyzed_range=[1.0, 40.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
        ),
        performance=PerformanceMetadata(
            accuracy_percent=68.9,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "prosthetic", "vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=80,
            trials_context="per_class",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events={"right_hand": 1, "feet": 2},
            code="BNCI2014-002",
            interval=[3, 8],
            paradigm="imagery",
            doi="10.1007/s00500-012-0895-4",
        )


@depreciated_alias("BNCI2014004", "1.1")
class BNCI2014_004(MNEBNCI):
    """BNCI 2014-004 Motor Imagery dataset.

    BCI Competition IV Dataset 2b [1]_.

    **Dataset Description**

    This dataset consists of EEG data from 9 subjects. The cue-based BCI
    paradigm consisted of two different motor imagery tasks, namely the
    imagination of movement of the left hand (class 1) and the right hand
    (class 2). Two sessions on different days were recorded for each subject.
    Each session is comprised of 6 runs separated by short breaks. One run
    consists of 20 trials (10 for each of the two possible classes), yielding
    a total of 120 trials per session.

    The subjects were sitting in a comfortable chair in front of a computer
    screen. At the beginning of a trial (t = 0 s), a fixation cross appeared
    on the black screen. In addition, a short acoustic warning tone was
    presented. After two seconds (t = 2 s), a cue in the form of an arrow
    pointing either to the left or to the right appeared and stayed on the
    screen for 1.25 s. This prompted the subjects to perform the desired
    motor imagery task. No feedback was provided. The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared
    from the screen at t = 6 s.

    Three bipolar channels (C3, Cz, C4) and three EOG channels were recorded.
    The signals were sampled at 250 Hz and bandpass-filtered between 0.5 Hz
    and 100 Hz. The reference was the left mastoid and the ground was the right
    mastoid. The electrode montage is a reduced version of the 10-20 system.

    References
    ----------
    .. [1] Tangermann, M., Muller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
           Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
           and Nolte, G., 2012. Review of the BCI competition IV.
           Frontiers in neuroscience, 6, p.55.

    Notes
    -----
    .. versionadded:: 0.4.0

    This dataset is commonly referred to as "BCI Competition IV Dataset 2b".
    It is widely used for binary motor imagery classification tasks.

    See Also
    --------
    BNCI2014_001 : 4-class motor imagery (Dataset 2a)
    BNCI2014_002 : 2-class motor imagery with Laplacian derivations
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=22,
            channel_types={"eeg": 22},
            montage="10-20",
            hardware="g.tec Guger Technologies OEG (two 16-channel biosignal amplifiers)",
            sensor_type="Ag/AgCl",
            reference="Car",
            ground="right mastoid",
            filters="50 Hz notch",
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
                "P2",
                "POz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=3,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                emg_channels=4,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="healthy",
            gender={"male": 6, "female": 4},
            age_mean=24.6,
            handedness="right",
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["right_hand", "right_arm", "left_hand", "feet"],
            study_design="motor\nimagery",
            feedback_type="none",
            stimulus_type="avatar",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="multisensory",
            synchronicity="asynchronous",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TNSRE.2007.906956",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "EOG regression artifact reduction",
                "bandpass filtering",
                "notch filtering",
            ],
            filter_details=FilterDetails(
                highpass_hz=0.5,
                lowpass_hz=100,
                bandpass=[0.5, 100],
                notch_hz=[50],
                filter_type="Butterworth",
                filter_order=5,
            ),
            artifact_methods=["EOG correction", "ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["Bandpower", "ERD"],
            frequency_bands=FrequencyBands(
                mu=[8, 12],
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "wheelchair/navigation",
                "prosthetic",
                "vr_ar",
                "communication",
                "neurofeedback",
            ],
            environment="laboratory",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=20,
            n_blocks=1,
            trials_context="per_run",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=5,
            events={"left_hand": 1, "right_hand": 2},
            code="BNCI2014-004",
            interval=[3, 7.5],
            paradigm="imagery",
            doi="10.1109/TNSRE.2007.906956",
        )


@depreciated_alias("BNCI2014008", "1.1")
class BNCI2014_008(MNEBNCI):
    """BNCI 2014-008 P300 dataset (ALS patients).

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 8 ALS patients using a P300 speller
    system. The dataset includes target and non-target responses during a
    visual P300 paradigm.

    **Participants**

    - 8 participants with amyotrophic lateral sclerosis (ALS)
    - Gender: 6 male, 2 female
    - Age range: 25-60 years

    **Recording Details**

    - Channels: 8 EEG channels
    - Sampling rate: 256 Hz
    - Reference: Linked mastoids

    References
    ----------
    .. [1] Riccio, A., Simione, L., Schettini, F., Pizzimenti, A., Inghilleri,
           M., Belardinelli, M. O., & Mattia, D. (2013). Attention and P300-based
           BCI performance in people with amyotrophic lateral sclerosis. Frontiers
           in human neuroscience, 7, 732.
           https://doi.org/10.3389/fnhum.2013.00732

    Notes
    -----
    .. versionadded:: 0.4.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=8,
            channel_types={"eeg": 8},
            montage="10-10",
            hardware="BCI2000",
            sensor_type="active electrodes",
            reference="right earlobe",
            ground="left mastoid",
            software="BCI2000",
            filters={"bandpass": [0.1, 30], "highpass_hz": 0.1, "lowpass_hz": 30},
            sensors=["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["ecg", "respiration", "gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="healthy",
            gender={"male": 6, "female": 2},
            age_mean=58,
            bci_experience="naive",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=0.8,
            study_design="report if the orientation of the rectangles in the test array was\nidentical to the ones in the memory array.",
            feedback_type="none",
            stimulus_type="rsvp",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="multisensory",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2013.00732",
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
                "bandpass filtering",
                "notch filtering",
                "epoching",
                "artifact rejection",
                "baseline correction",
                "decimation",
            ],
            filter_details=FilterDetails(
                highpass_hz=0.1,
                lowpass_hz=10,
                bandpass=[0.1, 10],
                notch_hz=50,
                filter_type="Butterworth",
                filter_order=4,
            ),
            artifact_methods=["ICA"],
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            frequency_bands=FrequencyBands(
                theta=[4, 8],
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "prosthetic", "vr_ar", "communication"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_repetitions=12,
        ),
        data_structure=DataStructureMetadata(
            n_trials=80,
            trials_context="total",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2014-008",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.3389/fnhum.2013.00732",
        )


@depreciated_alias("BNCI2014009", "1.1")
class BNCI2014_009(MNEBNCI):
    """BNCI 2014-009 P300 dataset.

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG data from 10 subjects using a P300 speller
    system with both grid speller and geo-speller paradigms. This loader
    includes only the grid speller data.

    **Participants**

    - 10 healthy subjects

    **Recording Details**

    - Channels: 16 EEG channels
    - Sampling rate: 256 Hz
    - Reference: Linked mastoids

    References
    ----------
    .. [1] Riccio, A., Simione, L., Schettini, F., Pizzimenti, A., Inghilleri,
           M., Belardinelli, M. O., & Mattia, D. (2013). Attention and P300-based
           BCI performance in people with amyotrophic lateral sclerosis. Frontiers
           in human neuroscience, 7, 732.
           https://doi.org/10.3389/fnhum.2013.00732

    Notes
    -----
    .. versionadded:: 0.4.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="10-10",
            hardware="g.tec",
            sensor_type="Ag/AgCl",
            reference="earlobe",
            ground="right mastoid",
            software="BCI2000",
            filters={"highpass_hz": 0.1, "lowpass_hz": 20},
            impedance_threshold_kohm=10,
            sensors=[
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "CP3",
                "CPz",
                "CP4",
                "Pz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            gender={"male": 6, "female": 4},
            age_mean=26.82,
            bci_experience="previous experience with P300-based BCIs",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=0.8,
            study_design="Subjects focused on one of 36 characters using two paradigms: P300 Speller (overt attention with 6x6 matrix) and GeoSpell (covert attention with hexagonal presentation)",
            feedback_type="none",
            stimulus_type="oddball",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="multisensory",
            synchronicity="asynchronous",
            mode="both",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1080/00140139.2012.661084",
            associated_paper_doi="10.3389/fnhum.2013.00732",
            funding=["EU grant FP7-", "grant FP7- FP7-"],
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="filtered",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filtering"],
            filter_details=FilterDetails(
                highpass_hz=0.1,
                lowpass_hz=20,
                bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 20.0},
                filter_type="Butterworth",
                filter_order=8,
            ),
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["Wavelet", "Time-Frequency"],
            frequency_bands=FrequencyBands(
                analyzed_range=[1.0, 20.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="3-fold",
            cv_folds=3,
            evaluation_type=["cross_subject"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=91.6,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "prosthetic", "vr_ar", "communication"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_repetitions=16,
        ),
        data_structure=DataStructureMetadata(
            n_trials="6 trials per run",
            trials_context="per run",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=3,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2014-009",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1088/1741-2560/11/3/035008",
        )
