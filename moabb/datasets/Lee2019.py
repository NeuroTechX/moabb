"""BMI/OpenBMI dataset."""

from functools import partialmethod

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
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


Lee2019_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"


class Lee2019(BaseDataset):
    """Base dataset class for Lee2019."""

    def __init__(
        self,
        paradigm,
        train_run=True,
        test_run=None,
        resting_state=False,
        sessions=(1, 2),
    ):
        if paradigm.lower() in ["imagery", "mi"]:
            paradigm = "imagery"
            code_suffix = "MI"
            interval = [
                0.0,
                4.0,
            ]  # [1.0, 3.5] is the interval used in paper for online prediction
            events = dict(left_hand=2, right_hand=1)
        elif paradigm.lower() in ["p300", "erp"]:
            paradigm = "p300"
            code_suffix = "ERP"
            interval = [
                0.0,
                1.0,
            ]  # [-0.2, 0.8] is the interval used in paper for online prediction
            events = dict(Target=1, NonTarget=2)
        elif paradigm.lower() in [
            "ssvep",
        ]:
            paradigm = "ssvep"
            code_suffix = "SSVEP"
            interval = [0.0, 4.0]
            events = {
                "12.0": 1,
                "8.57": 2,
                "6.67": 3,
                "5.45": 4,
            }  # dict(up=1, left=2, right=3, down=4)
        else:
            raise ValueError('unknown paradigm "{}"'.format(paradigm))
        for s in sessions:
            if s not in [1, 2]:
                raise ValueError("inexistent session {}".format(s))
        self.sessions = sessions

        super().__init__(
            subjects=list(range(1, 55)),
            sessions_per_subject=2,
            events=events,
            code="Lee2019-" + code_suffix,
            interval=interval,
            paradigm=paradigm,
            doi="10.5524/100542",
        )
        self.code_suffix = code_suffix
        self.train_run = train_run
        self.test_run = paradigm == "p300" if test_run is None else test_run
        self.resting_state = resting_state

    def _translate_class(self, c):
        if self.paradigm == "imagery":
            dictionary = dict(
                left_hand=["left"],
                right_hand=["right"],
            )
        elif self.paradigm == "p300":
            dictionary = dict(
                Target=["target"],
                NonTarget=["nontarget"],
            )
        elif self.paradigm == "ssvep":
            dictionary = {
                "12.0": ["up"],
                "8.57": ["left"],
                "6.67": ["right"],
                "5.45": ["down"],
            }
        for k, v in dictionary.items():
            if c.lower() in v:
                return k
        raise ValueError('unknown class "{}" for "{}" paradigm'.format(c, self.paradigm))

    def _check_mapping(self, file_mapping):
        def raise_error():
            raise ValueError(
                "file_mapping ({}) different than events ({})".format(
                    file_mapping, self.event_id
                )
            )

        if len(file_mapping) != len(self.event_id):
            raise_error()
        for c, v in file_mapping.items():
            v2 = self.event_id.get(self._translate_class(c), None)
            if v != v2 or v2 is None:
                raise_error()

    _scalings = dict(eeg=1e-6, emg=1e-6, stim=1)  # to load the signal in Volts

    def _make_raw_array(self, signal, ch_names, ch_type, sfreq, verbose=False):
        ch_names = [np.squeeze(c).item() for c in np.ravel(ch_names)]
        if len(ch_names) != signal.shape[1]:
            raise ValueError
        info = create_info(
            ch_names=ch_names, ch_types=[ch_type] * len(ch_names), sfreq=sfreq
        )
        factor = self._scalings.get(ch_type)
        raw = RawArray(data=signal.transpose(1, 0) * factor, info=info, verbose=verbose)
        return raw

    def _get_single_run(self, data):
        sfreq = data["fs"].item()
        file_mapping = {c.item(): int(v.item()) for v, c in data["class"]}
        self._check_mapping(file_mapping)

        # Create RawArray
        raw = self._make_raw_array(data["x"], data["chan"], "eeg", sfreq)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Create EMG channels
        emg_raw = self._make_raw_array(data["EMG"], data["EMG_index"], "emg", sfreq)

        # Create stim chan
        event_times_in_samples = data["t"].squeeze()
        event_id = data["y_dec"].squeeze()
        stim_chan = np.zeros(len(raw))
        for i_sample, id_class in zip(event_times_in_samples, event_id):
            stim_chan[i_sample] += id_class
        stim_raw = self._make_raw_array(
            stim_chan[:, None], ["STI 014"], "stim", sfreq, verbose="WARNING"
        )

        # Add EMG and stim channels
        raw = raw.add_channels([emg_raw, stim_raw])
        return raw

    def _get_single_rest_run(self, data, prefix):
        sfreq = data["fs"].item()
        raw = self._make_raw_array(
            data["{}_rest".format(prefix)], data["chan"], "eeg", sfreq
        )
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        return raw

    def _get_single_subject_data(self, subject):
        """Return data for a single subejct."""

        sessions = {}
        file_path_list = self.data_path(subject)

        for session in self.sessions:
            if self.train_run or self.test_run:
                mat = loadmat(file_path_list[self.sessions.index(session)])

            session_name = str(session - 1)
            sessions[session_name] = {}
            if self.train_run:
                sessions[session_name]["1train"] = self._get_single_run(
                    mat["EEG_{}_train".format(self.code_suffix)][0, 0]
                )
            if self.test_run:
                sessions[session_name]["4test"] = self._get_single_run(
                    mat["EEG_{}_test".format(self.code_suffix)][0, 0]
                )
            if self.resting_state:
                prefix = "pre"
                sessions[session_name][f"3{prefix}TestRest"] = self._get_single_rest_run(
                    mat["EEG_{}_test".format(self.code_suffix)][0, 0], prefix
                )
                sessions[session_name][f"0{prefix}TrainRest"] = self._get_single_rest_run(
                    mat["EEG_{}_train".format(self.code_suffix)][0, 0], prefix
                )
                prefix = "post"
                sessions[session_name][f"5{prefix}TestRest"] = self._get_single_rest_run(
                    mat["EEG_{}_test".format(self.code_suffix)][0, 0], prefix
                )
                sessions[session_name][f"2{prefix}TrainRest"] = self._get_single_rest_run(
                    mat["EEG_{}_train".format(self.code_suffix)][0, 0], prefix
                )

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []
        for session in self.sessions:
            url = "{0}session{1}/s{2}/sess{1:02d}_subj{2:02d}_EEG_{3}.mat".format(
                Lee2019_URL, session, subject, self.code_suffix
            )
            data_path = dl.data_dl(url, self.code, path, force_update, verbose)
            subject_paths.append(data_path)

        return subject_paths


class Lee2019_MI(Lee2019):
    """BMI/OpenBMI dataset for MI.

    Dataset from Lee et al 2019 [1]_.

    **Dataset Description**

    EEG signals were recorded with a sampling rate of 1,000 Hz and
    collected with 62 Ag/AgCl electrodes. The EEG amplifier used
    in the experiment was a BrainAmp (Brain Products; Munich,
    Germany). The channels were nasion-referenced and grounded
    to electrode AFz. Additionally, an EMG electrode recorded from
    each flexor digitorum profundus muscle with the olecranon
    used as reference. The EEG/EMG channel configuration and
    indexing numbers are described in Fig. 1. The impedances of the
    EEG electrodes were maintained below 10 k during the entire
    experiment.

    MI paradigm
    The MI paradigm was designed following a well-established system protocol.
    For all blocks, the first 3 s of each trial began
    with a black fixation cross that appeared at the center of the
    monitor to prepare subjects for the MI task. Afterwards, the subject
    performed the imagery task of grasping with the appropriate
    hand for 4 s when the right or left arrow appeared as a visual cue.
    After each task, the screen remained blank for 6 s (± 1.5 s). The
    experiment consisted of training and test phases; each phase
    had 100 trials with balanced right and left hand imagery tasks.
    During the online test phase, the fixation cross appeared at the
    center of the monitor and moved right or left, according to the
    real-time classifier output of the EEG signal.

    Parameters
    ----------
    train_run: bool (default True)
        if True, return runs corresponding to the training/offline phase (see paper).

    test_run: bool (default: False for MI and SSVEP paradigms, True for ERP)
        if True, return runs corresponding to the test/online phase (see paper). Beware that test_run
        for  MI and SSVEP do not have labels associated with trials: these runs could not be used in
        classification tasks.

    resting_state: bool (default False)
        if True, return runs corresponding to the resting phases before and after recordings (see paper).

    sessions: list of int (default [1,2])
        the list of the sessions to load (2 available).

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
           Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
           toolbox for three BCI paradigms: An investigation into BCI
           illiteracy. GigaScience, 8(5), 1–16.
           https://doi.org/10.1093/gigascience/giz002
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="10-20",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nose",
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
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                emg_channels=4,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=54,
            health_status="healthy",
            gender={"female": 25, "male": 29},
            age_mean=29.5,
            age_min=24,
            age_max=35,
            handedness={"right": 4},
            bci_experience="experienced",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "rest"],
            trial_duration=21.0,
            study_design="Three BCI paradigms: binary-class MI (left/right hand imagery), 36-symbol ERP speller (row-column with face stimuli), four-target SSVEP (5.45, 6.67, 8.57, 12 Hz)",
            feedback_type="visual",
            stimulus_type="rc_speller",
            stimulus_modalities=["visual", "multisensory"],
            primary_modality="multisensory",
            mode="both",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/gigascience/giz002",
            license="CC0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG available",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling",
                "bandpass filtering",
                "segmentation",
                "baseline correction (ERP only)",
            ],
            filter_details=FilterDetails(
                bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 40.0},
                filter_type="Butterworth",
                filter_order=5,
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=100,
            epoch_window=[-200.0, 800.0],
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "CNN", "Neural Network", "CCA"],
            feature_extraction=[
                "CSP",
                "FBCSP",
                "Bandpower",
                "ERD",
                "ERS",
                "PSD",
                "Time-Frequency",
            ],
            frequency_bands=FrequencyBands(
                alpha=[8.0, 12.0],
                mu=[8, 12],
                theta=[4, 8],
                analyzed_range=[1.0, 25.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["cross_subject"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=67.4,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            n_targets=4,
            stimulus_frequencies_hz=[5.45, 6.67, 8.57, 12.0],
        ),
        data_structure=DataStructureMetadata(
            n_trials={"MI": 200, "ERP": {"training": 1980, "test": 2160}, "SSVEP": 200},
        ),
        file_format="MAT",
        data_processed=True,
    )

    __init__ = partialmethod(Lee2019.__init__, "MI")


class Lee2019_ERP(Lee2019):
    """BMI/OpenBMI dataset for P300.

    Dataset from Lee et al 2019 [1]_.

    **Dataset Description**

    EEG signals were recorded with a sampling rate of 1,000 Hz and
    collected with 62 Ag/AgCl electrodes. The EEG amplifier used
    in the experiment was a BrainAmp (Brain Products; Munich,
    Germany). The channels were nasion-referenced and grounded
    to electrode AFz. Additionally, an EMG electrode recorded from
    each flexor digitorum profundus muscle with the olecranon
    used as reference. The EEG/EMG channel configuration and
    indexing numbers are described in Fig. 1. The impedances of the
    EEG electrodes were maintained below 10 k during the entire
    experiment.

    ERP paradigm
    The interface layout of the speller followed the typical design
    of a row-column speller. The six rows and six columns were
    configured with 36 symbols (A to Z, 1 to 9, and _). Each symbol
    was presented equally spaced. To enhance the
    signal quality, two additional settings were incorporated into
    the original row-column speller design, namely, random-set
    presentation and face stimuli. These additional settings
    help to elicit stronger ERP responses by minimizing adjacency
    distraction errors and by presenting a familiar face image. The
    stimulus-time interval was set to 80 ms, and the inter-stimulus
    interval (ISI) to 135 ms. A single iteration of stimulus presentation
    in all rows and columns was considered a sequence. Therefore,
    one sequence consisted of 12 stimulus flashes. A maximum
    of five sequences (i.e., 60 flashes) was allotted without prolonged
    inter-sequence intervals for each target character. After the end
    of five sequences, 4.5 s were given to the user for identifying, locating,
    and gazing at the next target character. The participant
    was instructed to attend to the target symbol by counting the
    number of times each target character had been flashed.
    In the training session, subjects were asked to copy-spell
    a given sentence, "NEURAL NETWORKS AND DEEP LEARNING"
    (33 characters including spaces) by gazing at the target character
    on the screen. The training session was performed in the offline
    condition, and no feedback was provided to the subject during
    the EEG recording. In the test session, subjects were instructed to
    copy-spell "PATTERN RECOGNITION MACHINE LEARNING"
    (36 characters including spaces), and the real-time EEG data were
    analyzed based on the classifier that was calculated from the
    training session data. The selected character from the subject’s
    current EEG data was displayed in the top left area of the screen
    at the end of the presentation (i.e., after five sequences).
    Per participant, the collected EEG data for the ERP experiment consisted
    of 1,980 and 2,160 trials (samples) for training and test phase, respectively.

    Parameters
    ----------
    train_run: bool (default True)
        if True, return runs corresponding to the training/offline phase (see paper).

    test_run: bool (default: False for MI and SSVEP paradigms, True for ERP)
        if True, return runs corresponding to the test/online phase (see paper). Beware that test_run
        for  MI and SSVEP do not have labels associated with trials: these runs could not be used in
        classification tasks.

    resting_state: bool (default False)
        if True, return runs corresponding to the resting phases before and after recordings (see paper).

    sessions: list of int (default [1,2])
        the list of the sessions to load (2 available).

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
           Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
           toolbox for three BCI paradigms: An investigation into BCI
           illiteracy. GigaScience, 8(5), 1–16.
           https://doi.org/10.1093/gigascience/giz002
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="10-20",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nose",
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
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                emg_channels=4,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=54,
            health_status="healthy",
            gender={"female": 25, "male": 29},
            age_mean=29.5,
            age_min=24,
            age_max=35,
            handedness={"right": 4},
            bci_experience="experienced",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=2,
            class_labels=["left_hand", "rest"],
            trial_duration=21.0,
            study_design="MI: binary-class left/right hand motor imagery; ERP: 36-symbol row-column speller with random-set presentation and face stimuli; SSVEP: four target frequencies (5.45, 6.67, 8.57, 12 Hz) in four direct...",
            feedback_type="visual",
            stimulus_type="rc_speller",
            stimulus_modalities=["visual", "multisensory"],
            primary_modality="multisensory",
            mode="both",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/gigascience/giz002",
            license="CC0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG data provided, preprocessing applied for analysis",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling",
                "bandpass filtering",
                "segmentation",
                "baseline correction (ERP only)",
            ],
            filter_details=FilterDetails(
                bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 40.0},
                filter_type="Butterworth",
                filter_order=5,
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=100,
            epoch_window=[-200.0, 800.0],
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "CNN", "Neural Network", "CCA"],
            feature_extraction=[
                "CSP",
                "FBCSP",
                "Bandpower",
                "ERD",
                "ERS",
                "PSD",
                "Time-Frequency",
            ],
            frequency_bands=FrequencyBands(
                alpha=[8.0, 12.0],
                mu=[8, 12],
                theta=[4, 8],
                analyzed_range=[1.0, 25.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["cross_subject"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=67.4,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=4,
            stimulus_frequencies_hz=[5.45, 6.67, 8.57, 12.0],
        ),
        data_structure=DataStructureMetadata(
            n_trials={
                "MI": "200 per session (100 training + 100 test)",
                "ERP": "4140 per session (1980 training + 2160 test)",
                "SSVEP": "200 per session (100 training + 100 test)",
            },
        ),
        file_format="MAT",
        data_processed=True,
    )

    __init__ = partialmethod(Lee2019.__init__, "ERP")


class Lee2019_SSVEP(Lee2019):
    """BMI/OpenBMI dataset for SSVEP.

    Dataset from Lee et al 2019 [1]_.

    **Dataset Description**

    EEG signals were recorded with a sampling rate of 1,000 Hz and
    collected with 62 Ag/AgCl electrodes. The EEG amplifier used
    in the experiment was a BrainAmp (Brain Products; Munich,
    Germany). The channels were nasion-referenced and grounded
    to electrode AFz. Additionally, an EMG electrode recorded from
    each flexor digitorum profundus muscle with the olecranon
    used as reference. The EEG/EMG channel configuration and
    indexing numbers are described in Fig. 1. The impedances of the
    EEG electrodes were maintained below 10 k during the entire
    experiment.

    SSVEP paradigm
    Four target SSVEP stimuli were designed to flicker at 5.45, 6.67,
    8.57, and 12 Hz and were presented in four positions (down,
    right, left, and up, respectively) on a monitor. The designed
    paradigm followed the conventional types of SSVEP-based BCI
    systems that require four-direction movements. Partici-
    pants were asked to fixate the center of a black screen and then
    to gaze in the direction where the target stimulus was high-
    lighted in a different color. Each SSVEP stimulus
    was presented for 4 s with an ISI of 6 s. Each target frequency
    was presented 25 times. Therefore, the corrected EEG data had
    100 trials (4 classes x 25 trials) in the offline training phase and
    another 100 trials in the online test phase. Visual feedback was
    presented in the test phase; the estimated target frequency was
    highlighted for 1 s with a red border at the end of each trial.

    Parameters
    ----------
    train_run: bool (default True)
        if True, return runs corresponding to the training/offline phase (see paper).

    test_run: bool (default: False for MI and SSVEP paradigms, True for ERP)
        if True, return runs corresponding to the test/online phase (see paper). Beware that test_run
        for  MI and SSVEP do not have labels associated with trials: these runs could not be used in
        classification tasks.

    resting_state: bool (default False)
        if True, return runs corresponding to the resting phases before and after recordings (see paper).

    sessions: list of int (default [1,2])
        the list of the sessions to load (2 available).

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
           Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
           toolbox for three BCI paradigms: An investigation into BCI
           illiteracy. GigaScience, 8(5), 1–16.
           https://doi.org/10.1093/gigascience/giz002
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="10-20",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nose",
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
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                emg_channels=4,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=54,
            health_status="healthy",
            gender={"female": 25, "male": 29},
            age_mean=29.5,
            age_min=24,
            age_max=35,
            handedness={"right": 4},
            bci_experience="experienced",
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            n_classes=1,
            class_labels=["12hz"],
            trial_duration=21.0,
            study_design="Three BCI paradigms: binary-class MI (left/right hand imagery), 36-symbol ERP speller (row-column paradigm with face stimuli), and four-target SSVEP (5.45, 6.67, 8.57, 12 Hz)",
            feedback_type="visual",
            stimulus_type="rc_speller",
            stimulus_modalities=["visual", "multisensory"],
            primary_modality="multisensory",
            mode="both",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/gigascience/giz002",
            license="CC0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG provided, preprocessing applied for analysis",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling",
                "bandpass filtering",
                "segmentation",
                "baseline correction",
            ],
            filter_details=FilterDetails(
                bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 40.0},
                filter_type="Butterworth",
                filter_order=5,
            ),
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=100,
            epoch_window=[-200.0, 800.0],
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "CNN", "Neural Network", "CCA"],
            feature_extraction=[
                "CSP",
                "FBCSP",
                "Bandpower",
                "ERD",
                "ERS",
                "PSD",
                "Time-Frequency",
            ],
            frequency_bands=FrequencyBands(
                alpha=[8.0, 12.0],
                mu=[8, 12],
                theta=[4, 8],
                analyzed_range=[1.0, 25.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
            evaluation_type=["cross_subject"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=67.4,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            n_targets=4,
            stimulus_frequencies_hz=[5.45, 6.67, 8.57, 12.0],
        ),
        data_structure=DataStructureMetadata(
            n_trials={"MI": 200, "ERP_training": 1980, "ERP_test": 2160, "SSVEP": 200},
        ),
        file_format="MAT",
        data_processed=True,
    )

    __init__ = partialmethod(Lee2019.__init__, "SSVEP")
