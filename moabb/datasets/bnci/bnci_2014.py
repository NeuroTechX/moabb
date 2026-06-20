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
    ParadigmSpecificMetadata,
    ParticipantMetadata,
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

# BNCI2014-002 (Steyrl et al.) ships 15 unlabeled channels: three small
# Laplacian groups centered on C3, Cz, C4. Assumed 3x5 grid approximation
# (anterior FC row / central C row / posterior CP row) so every channel gets
# a standard_1005 position. See _load_data_002_2014 for caveats.
_CH_NAMES_002_2014 = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]  # fmt: skip


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
        # The data file carries no channel labels or positions, and neither
        # the paper nor the official BNCI description provides them (all three
        # verified). Steyrl et al. (Fig 3) only describe the geometry: three
        # small-Laplacian groups centered on C3, Cz, C4, each with four
        # surrounding electrodes (anterior, posterior, left, right) at 2.5 cm.
        # We *assume* the conventional 3x5 grid approximation of that layout
        # (anterior FC row, central C row, posterior CP row), which maps every
        # channel to a standard_1005 position so topomaps/interpolation work.
        # The exact custom 2.5 cm positions and the true column order are not
        # documented, so treat these labels as approximate. Passing real names
        # (not None) makes _convert_run attach standard_1005 automatically.
        raws, _ = _convert_mi(
            filename,
            _CH_NAMES_002_2014,
            ["eeg"] * 15,
            dataset_code="BNCI2014-002",
            subject_id=subject,
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
            n_channels=25,
            channel_types={"eeg": 22, "eog": 3},
            montage="custom",
            hardware="BrainAmp MR plus",
            sensor_type="Ag/AgCl",
            reference="left mastoid",
            ground="right mastoid",
            software="BCI2000",
            filters="bandpass 0.5-100 Hz, 50 Hz notch",
            sensors=[
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
                "CPz",
                "Cz",
                "EOG1",
                "EOG2",
                "EOG3",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FCz",
                "Fz",
                "P1",
                "P2",
                "POz",
                "Pz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=False, has_emg=False),
            cap_manufacturer="EASYCAP GmbH",
            impedance_threshold_kohm=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=9, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "feet", "tongue"],
            trial_duration=4.0,
            study_design="Cue-based four-class motor imagery (left hand, right hand, both feet, tongue); two sessions per subject on different days, each with 6 runs of 48 trials (288 trials per session)",
            feedback_type="none",
            stimulus_type="arrow_cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="multisensory",
            synchronicity="synchronous",
            mode="offline",
            events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            instructions="Subjects instructed to perform motor imagery during cued periods",
            stimulus_presentation={
                "cross_onset": "0 s",
                "arrow_cue": "2 s",
                "trial_duration": "6 s",
            },
            hed_tags={
                "left_hand": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation, "
                    "(Leftward, Arrow)), "
                    "(Agent-action, (Imagine, Move, (Left, Hand)))"
                ),
                "right_hand": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation, "
                    "(Rightward, Arrow)), "
                    "(Agent-action, (Imagine, Move, (Right, Hand)))"
                ),
                "feet": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation, "
                    "(Downward, Arrow)), "
                    "(Agent-action, (Imagine, Move, Foot))"
                ),
                "tongue": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation, "
                    "(Upward, Arrow)), "
                    "(Agent-action, (Imagine, Move, Tongue))"
                ),
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnins.2012.00055",
            description="BCI Competition IV - Data set 2a: cue-based four-class motor imagery (left hand, right hand, both feet, tongue)",
            investigators=[
                "Michael Tangermann",
                "Klaus-Robert Müller",
                "Ad Aertsen",
                "Niels Birbaumer",
                "Christoph Braun",
                "Clemens Brunner",
                "Robert Leeb",
                "Carsten Mehring",
                "Kai J. Miller",
                "Gernot R. Müller-Putz",
                "Guido Nolte",
                "Gert Pfurtscheller",
                "Hubert Preissl",
                "Gerwin Schalk",
                "Alois Schlögl",
                "Carmen Vidaurre",
                "Stephan Waldert",
                "Benjamin Blankertz",
            ],
            institution="Berlin Institute of Technology",
            country="Germany",
            license="CC-BY-ND-4.0",
            repository="BNCI Horizon",
            data_url="http://www.bbci.de/competition/iv/",
            publication_year=2012,
            senior_author="Michael Tangermann",
            contact_info=["michael.tangermann@tu-berlin.de"],
            institution_address="FR 6-9, Franklinstr. 28/29, 10587 Berlin, Germany",
            institution_department="Machine Learning Laboratory",
            keywords=["brain-computer interface", "BCI", "competition"],
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor"]),
        preprocessing=PreprocessingMetadata(
            data_state="minimally preprocessed (bandpass and notch filtered)",
            preprocessing_applied=True,
            preprocessing_steps=[
                "bandpass filtering (0.5-100 Hz)",
                "50 Hz notch filtering",
            ],
            highpass_hz=0.5,
            lowpass_hz=100,
            bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
            filter_type="analog",
            filter_order=None,
            re_reference="none",
            downsampled_to_hz=None,
            notes="Sampled at 250 Hz; bandpass filtered between 0.5 and 100 Hz with an additional 50 Hz notch filter to suppress line noise; amplifier sensitivity 100 uV",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "LDA",
                "SVM",
                "Neural Network",
                "Naive Bayes",
                "RBF Neural Network",
            ],
            feature_extraction=["CSP", "FBCSP", "Bandpower", "ERD", "ERS"],
            frequency_bands={"mu": [8, 12], "beta": [16, 24]},
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="train-test split", evaluation_type=["within_session"]
        ),
        performance={"MSE": 0.382},
        bci_application=BCIApplicationMetadata(
            applications=["cursor_control", "communication"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "feet", "tongue"],
            cue_duration_s=1.25,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials={"training": 288, "test": 288},
            n_blocks=6,
            trials_context="per session: 6 runs of 48 trials (12 per class) = 288 trials; 2 sessions per subject (T = training, E = evaluation)",
        ),
        file_format="GDF",
        data_processed=True,
        sessions_per_subject=2,
        runs_per_session=6,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            code="BNCI2014-001",
            interval=[2, 6],
            paradigm="imagery",
            doi="10.3389/fnins.2012.00055",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
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
            sampling_rate=512.0,
            n_channels=15,
            channel_types={"eeg": 15},
            montage="Laplacian",
            hardware="g.USBamp",
            sensor_type="Ag/AgCl",
            reference="left mastoid",
            ground="right mastoid",
            software="BCI2000",
            filters="8th order Butterworth band-pass filters",
            # Approximate 3x5 grid labels (see _CH_NAMES_002_2014); the data
            # ships unlabeled and the exact custom Laplacian positions are
            # undocumented.
            sensors=list(_CH_NAMES_002_2014),
            line_freq=50.0,
            cap_manufacturer="Guger Technologies OG",
            cap_model="g.LADYbird",
            electrode_type="active",
            electrode_material="Ag/AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=13,
            health_status="healthy",
            age_min=20.0,
            age_max=30.0,
            bci_experience="mixed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "feet"],
            trial_duration=5.0,
            study_design="Two-class motor imagery: right hand and feet. Cue-guided Graz-BCI training paradigm with recording, training, and feedback within a single session.",
            feedback_type="continuous",
            stimulus_type="bar_graph",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            events={"right_hand": 769, "feet": 770},
        ),
        documentation=DocumentationMetadata(
            doi="10.1515/bmt-2014-0117",
            investigators=[
                "David Steyrl",
                "Reinhold Scherer",
                "Oswin Förstner",
                "Gernot R. Müller-Putz",
            ],
            institution="Graz University of Technology",
            institution_department="Institute for Knowledge Discovery, Laboratory of Brain-Computer Interfaces",
            country="Austria",
            license="CC-BY-ND-4.0",
            repository="BNCI Horizon",
            publication_year=2014,
            funding=["FP7 BackHome (No. 288566)", "FP7 ABC (No. 287774)"],
            contact_info=[
                "david.steyrl@tugraz.at",
                "reinhold.scherer@tugraz.at",
                "oswin.foerstner@student.tugraz.at",
                "gernot.mueller@tugraz.at",
            ],
            associated_paper_doi="10.3217/978-3-85125-378-8-61",
            keywords=[
                "brain-computer interfaces",
                "machine learning",
                "random forests",
                "regularized linear discriminant analysis",
                "sensorimotor rhythms",
            ],
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="minimally preprocessed (online filtered)",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filtering"],
            filter_type="Butterworth",
            filter_order=8,
            artifact_methods=None,
            re_reference=None,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Random Forest", "Shrinkage LDA"],
            feature_extraction=["CSP", "DFT", "Bandpower"],
            frequency_bands={"alpha": [6, 14], "beta": [14, 40]},
            spatial_filters=["CSP", "Laplacian"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="train-test split", evaluation_type=["within_subject"]
        ),
        performance={
            "accuracy_percent": 79.30,
            "peak_accuracy": 89.67,
            "median_accuracy": 80.42,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication", "control"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand", "feet"],
            cue_duration_s=None,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=160,
            n_trials_per_class={"right_hand": 80, "feet": 80},
            n_blocks=8,
            trials_context="total per subject",
        ),
        sessions_per_subject=1,
        runs_per_session=8,
        file_format="MAT",
        data_processed=True,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events={"right_hand": 1, "feet": 2},
            code="BNCI2014-002",
            interval=[3, 8],
            paradigm="imagery",
            doi="10.1007/s00500-012-0895-4",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            n_channels=3,
            channel_types={"eeg": 3, "eog": 3},
            montage="standard_1020",
            hardware="g.tec",
            sensor_type="EEG",
            reference="left mastoid",
            ground="Fz",
            software="rtsBCI (MATLAB/Simulink)",
            filters="0.5-100 Hz bandpass, 50 Hz notch",
            sensors=["C3", "C4", "Cz", "EOG1", "EOG2", "EOG3"],
            line_freq=50.0,
            impedance_threshold_kohm=None,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=3,
                eog_type=["horizontal", "vertical", "radial"],
                has_emg=False,
                emg_channels=None,
                other_physiological=None,
            ),
            cap_manufacturer="Easycap",
            cap_model=None,
            electrode_type=None,
            electrode_material="Ag/AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=9,
            health_status="healthy",
            gender=None,
            age_mean=24.7,
            age_std=3.3,
            age_min=None,
            age_max=None,
            ages=None,
            handedness="right",
            clinical_population=None,
            bci_experience="naive",
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="motor_imagery",
            events={
                "276": "Idling EEG (eyes open)",
                "277": "Idling EEG (eyes closed)",
                "768": "Start of a trial",
                "769": "Cue onset left (class 1)",
                "770": "Cue onset right (class 2)",
                "781": "BCI feedback (continuous)",
                "783": "Cue unknown",
                "1023": "Rejected trial",
                "1077": "Horizontal eye movement",
                "1078": "Vertical eye movement",
                "1079": "Eye rotation",
                "1081": "Eye blinks",
                "32766": "Start of a new run",
            },
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class=None,
            trial_duration=7.5,
            tasks=["left_hand_imagery", "right_hand_imagery"],
            study_design="Two-class motor imagery: left hand and right hand. Screening sessions (01T, 02T) without feedback, feedback sessions (03T, 04E, 05E) with smiley feedback.",
            study_domain="brain-computer interface",
            feedback_type="visual",
            stimulus_type="arrow_cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="cue_based",
            mode="both",
            has_training_test_split=True,
            instructions="Subjects selected their best motor imagery strategy (e.g., squeezing a ball or pulling a brake) and performed kinesthetic motor imagery of left or right hand movements.",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation=None,
            hed_tags={
                "left_hand": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation, "
                    "(Leftward, Arrow)), "
                    "(Agent-action, (Imagine, Move, (Left, Hand)))"
                ),
                "right_hand": (
                    "(Sensory-event, Experimental-stimulus, Visual-presentation, "
                    "(Rightward, Arrow)), "
                    "(Agent-action, (Imagine, Move, (Right, Hand)))"
                ),
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TNSRE.2007.906956",
            description="BCI Competition 2008 - Graz data set B: Two-class motor imagery dataset (left/right hand) with screening sessions (no feedback) and smiley feedback sessions. 9 subjects, 3 bipolar EEG channels (C3, Cz, C4) + 3 EOG channels, 250 Hz.",
            investigators=[
                "R. Leeb",
                "C. Brunner",
                "G. R. Müller-Putz",
                "A. Schlögl",
                "G. Pfurtscheller",
                "F. Lee",
                "C. Keinrath",
                "R. Scherer",
                "H. Bischof",
            ],
            institution="Graz University of Technology",
            country="AT",
            repository="BNCI Horizon",
            data_url="http://biosig.sourceforge.net/",
            license="CC-BY-ND-4.0",
            publication_year=2007,
            senior_author="G. Pfurtscheller",
            institution_department="Institute for Knowledge Discovery",
            keywords=[
                "brain-computer interface",
                "BCI",
                "electroencephalogram",
                "EEG",
                "motor imagery",
                "BCI competition",
                "smiley feedback",
            ],
        ),
        sessions_per_subject=5,
        runs_per_session=1,
        sessions=["01T", "02T", "03T", "04E", "05E"],
        data_processed=False,
        file_format="GDF",
        external_links={"source": "http://biosig.sourceforge.net/"},
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw with online filtering",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filtering", "notch filtering"],
            highpass_hz=0.5,
            lowpass_hz=100.0,
            bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
            notch_hz=[50.0],
            filter_type="analog",
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            downsampled_to_hz=None,
            epoch_window=None,
            notes="Online bandpass (0.5-100 Hz) and notch (50 Hz) filters applied during recording. Artifact trials marked with event type 1023. EOG channels provided for user-applied artifact correction.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["Bandpower", "BP"],
            frequency_bands={},
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10x10 cross-validation",
            cv_folds=10,
            evaluation_type=["within_subject"],
        ),
        performance={},
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"], environment="laboratory", online_feedback=True
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type=None,
            code_length=None,
            n_targets=None,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=None,
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=1.25,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials={"screening": 120, "feedback": 160},
            n_trials_per_class=None,
            n_blocks=None,
            block_duration_s=None,
            trials_context="per session",
        ),
        abstract="BCI Competition 2008 Graz data set B. EEG data from 9 subjects performing two-class motor imagery (left hand vs right hand). Two screening sessions without feedback (120 trials each) and three feedback sessions with smiley feedback (160 trials each). Three bipolar EEG channels (C3, Cz, C4) and three EOG channels recorded at 250 Hz.",
        methodology="Subjects performed kinesthetic motor imagery of left or right hand movements. Two screening sessions (01T, 02T) without feedback: 6 runs x 20 trials = 120 trials per session. Three feedback sessions (03T, 04E, 05E) with smiley feedback: 4 runs x 40 trials (20 per class) = 160 trials per session. Screening trials: fixation cross + beep at t=0, arrow cue at ~t=2 for 1.25s, imagery for 4s, break. Feedback trials: smiley at t=0, beep at t=2, cue from t=3 to t=7.5 with continuous smiley feedback. Three bipolar EEG channels (C3, Cz, C4) plus three monopolar EOG channels recorded at 250 Hz with 0.5-100 Hz bandpass and 50 Hz notch filter. EEG ground at Fz, EOG reference at left mastoid. Amplifier: g.tec. Software: rtsBCI (MATLAB/Simulink).",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=5,
            events={"left_hand": 1, "right_hand": 2},
            code="BNCI2014-004",
            interval=[3, 7.5],
            paradigm="imagery",
            doi="10.1109/TNSRE.2007.906956",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
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
            hardware="g.MOBILAB",
            sensor_type="active electrodes",
            reference="right earlobe",
            ground="left mastoid",
            software="BCI2000",
            filters="0.1-10 Hz bandpass, 50 Hz notch",
            sensors=["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"],
            line_freq=50.0,
            electrode_type="g.Ladybird",
            electrode_material="Ag/AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="ALS patients",
            gender={"M": 5, "F": 3},
            age_mean=58.0,
            age_std=12.0,
            age_min=40,
            age_max=72,
            clinical_population="amyotrophic lateral sclerosis",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=2,
            class_labels=["target", "non-target"],
            trial_duration=None,
            study_design="P300 speller with 6x6 matrix for copy-spelling task in ALS patients",
            feedback_type="visual",
            stimulus_type="row-column intensification",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=True,
            instructions="Copy spell seven predefined words of five characters each by focusing attention on desired letters",
            events={"target": 1, "non-target": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2013.00732",
            investigators=[
                "Angela Riccio",
                "Luca Simione",
                "Francesca Schettini",
                "Alessia Pizzimenti",
                "Maurizio Inghilleri",
                "Marta Olivetti Belardinelli",
                "Donatella Mattia",
                "Febo Cincotti",
            ],
            institution="Fondazione Santa Lucia",
            country="Italy",
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
            publication_year=2013,
            senior_author="Febo Cincotti",
            contact_info=["a.riccio@hsantalucia.it"],
            funding=[
                "Italian Agency for Research on ALS-ARiSLA project 'Brindisys'",
                "FARI project C26I12AJZZ at the Sapienza University of Rome",
            ],
            keywords=[
                "brain computer interface",
                "amyotrophic lateral sclerosis",
                "P300",
                "attention",
                "working memory",
            ],
            institution_address="Via Ardeatina, 306, 00179 Rome, Italy",
            institution_department="Neuroelectrical Imaging and BCI Laboratory",
            ethics_approval=["Fondazione Santa Lucia ethic committee"],
        ),
        tags=Tags(pathology=["ALS"], modality=["P300"], type=["ERP"]),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "bandpass filtering",
                "notch filtering",
                "artifact rejection",
                "baseline correction",
            ],
            highpass_hz=0.1,
            lowpass_hz=10.0,
            bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 10.0},
            notch_hz=[50],
            filter_type="Butterworth",
            filter_order=4,
            artifact_methods=["amplitude threshold rejection"],
            re_reference="right earlobe",
            epoch_window=[0.0, 1.0],
            notes="Epochs with peak amplitude >70 μV or <-70 μV were rejected. Baseline correction based on 200 ms preceding each epoch.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["SWLDA"], feature_extraction=["temporal features", "decimation"]
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="7-fold", cv_folds=7, evaluation_type=["within_subject"]
        ),
        performance={
            "accuracy_percent": 97.5,
            "binary_accuracy_offline": 87.4,
            "p300_amplitude_mean_uv": 3.3,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication"], environment="laboratory", online_feedback=True
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=36,
            n_repetitions=10,
            isi_ms=125.0,
            soa_ms=250.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=35,
            n_blocks=7,
            trials_context="per subject (7 words, 5 characters each)",
        ),
        file_format="Unknown",
        data_processed=True,
        sessions_per_subject=1,
        runs_per_session=1,
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2014-008",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.3389/fnhum.2013.00732",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
            hardware="g.USBamp",
            sensor_type="Ag/AgCl",
            reference="linked earlobes",
            ground="right mastoid",
            software="BCI2000",
            filters="bandpass 0.1-20 Hz",
            sensors=[
                "Fz",
                "Cz",
                "Pz",
                "Oz",
                "P3",
                "P4",
                "PO7",
                "PO8",
                "F3",
                "F4",
                "FCz",
                "C3",
                "C4",
                "CP3",
                "CPz",
                "CP4",
            ],
            line_freq=50.0,
            impedance_threshold_kohm=10.0,
            cap_manufacturer="Electro-Cap International, Inc.",
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"female": 10, "male": 0},
            age_mean=26.8,
            age_std=5.6,
            ages=[22, 23, 27, 23, 23, 26, 40, 23, 26, 35],
            bci_experience="experienced",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="spelling",
            events={"target": 1, "non_target": 5},
            n_classes=2,
            class_labels=["target", "non_target"],
            trial_duration=16.0,
            study_design="P300-based BCI with two interfaces: P300 Speller (overt attention) and GeoSpell (covert attention). 36 alphanumeric characters presented. Eight stimulation sequences per trial with 16 target intensifications.",
            feedback_type="none",
            stimulus_type="visual_intensification",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Subject focused on one out of 36 different characters. At the beginning of each trial, the system prompted the subject with the character to attend. Target prompt appeared during a 2 s pre-trial interval.",
            stimulus_presentation={
                "stimulus_duration_ms": "125",
                "isi_ms": "125",
                "soa_ms": "250",
                "n_sequences": "8",
                "n_intensifications_per_target": "16",
                "pre_trial_interval_s": "2.0",
                "tti_min_ms": "500",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2560/11/3/035008",
            description="Complete record of P300 evoked potentials recorded with BCI2000 using two different paradigms: P300 Speller (overt attention) and GeoSpell (covert attention). 10 healthy subjects focused on one out of 36 different characters.",
            investigators=[
                "P Aricò",
                "F Aloise",
                "F Schettini",
                "S Salinari",
                "D Mattia",
                "F Cincotti",
            ],
            institution="Fondazione Santa Lucia IRCCS",
            country="Italy",
            license="CC-BY-NC-ND-4.0",
            repository="BNCI Horizon",
            publication_year=2014,
            senior_author="F Cincotti",
            contact_info=["p.arico@hsantalucia.it"],
            institution_address="Rome, Italy",
            institution_department="Neuroelectrical Imaging and BCI Lab",
            ethics_approval=["Approved by local Ethics Committee"],
            keywords=[
                "P300 latency jitter",
                "brain-computer interface",
                "covert attention",
                "wavelet analysis",
                "single epoch",
            ],
            associated_paper_doi="10.3389/fnhum.2013.00732",
        ),
        sessions_per_subject=4,
        runs_per_session=1,
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filtering"],
            highpass_hz=0.1,
            lowpass_hz=20.0,
            bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 20.0},
            filter_type="Butterworth",
            filter_order=8,
            re_reference="linked earlobes",
            epoch_window=[0.0, 0.8],
            notes="EEG acquired using g.USBamp amplifier (g.Tec, Austria), digitized at 256 Hz",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "SWLDA"],
            feature_extraction=["Wavelet", "Time-Frequency", "CWT"],
            frequency_bands={"analyzed_range": [1.0, 20.0]},
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross-validation", cv_folds=3, evaluation_type=["within_session"]
        ),
        performance={
            "p300_latency_jitter_correlation": "negative correlation with accuracy"
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication", "spelling"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=36,
            n_repetitions=8,
            isi_ms=125.0,
            soa_ms=250.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=18, trials_context="6 trials × 3 runs per session", n_blocks=3
        ),
        tags=Tags(pathology=["Healthy"], modality=["Visual"], type=["P300", "ERP"]),
        file_format="MAT",
        data_processed=True,
        abstract="This dataset represents a complete record of P300 evoked potentials recorded with BCI2000 using two different paradigms: a paradigm based on the P300 Speller originally described by Farwell and Donchin in overt attention condition and a paradigm based on the GeoSpell interface used in covert attention condition. In these sessions, 10 healthy subjects focused on one out of 36 different characters. The objective was to predict the correct character in each of the provided character selection epochs.",
        methodology="Ten healthy subjects (10 female, mean age = 26.8 ± 5.6) with previous experience with P300-based BCIs attended 4 recording sessions. Scalp EEG potentials were measured using 16 Ag/AgCl electrodes arranged on an elastic cap per the 10-10 standard. Each electrode was referenced to the linked earlobes and grounded to the right mastoid. The EEG was acquired using a g.USBamp amplifier (g.Tec, Austria), digitized at 256 Hz, high pass- and low pass-filtered with cutoff frequencies of 0.1 Hz and 20 Hz, respectively. The electrode impedance did not exceed 10 kΩ. Visual stimulation, acquisition and online classification were performed with BCI2000. Each subject attended 4 recording sessions. During each session, the subject performed three runs with each of the stimulation interfaces. Each trial consisted of eight stimulation sequences, and thus, 16 intensifications of the target character. Each stimulus was intensified for 125 ms, with an inter stimulus interval (ISI) of 125 ms, yielding a 250 ms lag between the appearance of two stimuli (SOA). Pseudorandom stimulation sequences were assembled so that each target intensification would not occur within 500 ms after the previous one to avoid the attentional blink phenomenon.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=3,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2014-009",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1088/1741-2560/11/3/035008",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
