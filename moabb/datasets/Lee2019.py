"""BMI/OpenBMI dataset."""

from functools import partialmethod

import numpy as np
from mne import Annotations, create_info
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
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from moabb.utils import _handle_deprecated_kwargs


Lee2019_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"


class Lee2019(BaseDataset):
    """Base dataset class for Lee2019."""

    def __init__(
        self,
        paradigm,
        train_run=True,
        test_run=None,
        resting_state=False,
        sessions=None,
        subjects=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {
            "TrainRun": "train_run",
            "TestRun": "test_run",
            "RestingState": "resting_state",
            "Sessions": "sessions",
            "Subjects": "subjects",
        }
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "Lee2019")
        train_run = resolved.get("train_run", train_run)
        test_run = resolved.get("test_run", test_run)
        resting_state = resolved.get("resting_state", resting_state)
        sessions = resolved.get("sessions", sessions)
        subjects = resolved.get("subjects", subjects)

        if sessions is None:
            sessions = (1, 2)

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
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
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
        rest_key = f"{prefix}_rest"
        raw = self._make_raw_array(data[rest_key], data["chan"], "eeg", sfreq)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Add EMG channels if available and duration matches
        if "EMG" in data.dtype.names and "EMG_index" in data.dtype.names:
            rest_samples = data[rest_key].shape[0]
            if prefix == "pre":
                emg_slice = data["EMG"][:rest_samples]
            else:
                emg_slice = data["EMG"][-rest_samples:]
            if emg_slice.shape[0] == rest_samples:
                emg_raw = self._make_raw_array(emg_slice, data["EMG_index"], "emg", sfreq)
                raw = raw.add_channels([emg_raw])

        # Add annotation for BIDS compatibility
        raw.set_annotations(
            Annotations(onset=[0], duration=[raw.times[-1]], description=["rest"])
        )
        return raw

    def _get_single_subject_data(self, subject):
        """Return data for a single subejct."""

        sessions = {}
        file_path_list = self.data_path(subject)

        for session in self.sessions:
            if self.train_run or self.test_run:
                mat = loadmat(file_path_list[self.sessions.index(session)])

            session_name = str(session)
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
            channel_types={"eeg": 62, "emg": 4},
            montage="standard_1005",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nasion",
            ground="AFz",
            impedance_threshold_kohm=10.0,
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
                "EMG1",
                "EMG2",
                "EMG3",
                "EMG4",
                "F10",
                "F3",
                "F4",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FT10",
                "FT9",
                "FTT10h",
                "FTT9h",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P7",
                "P8",
                "PO10",
                "PO3",
                "PO4",
                "PO9",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP10",
                "TP7",
                "TP8",
                "TP9",
                "TPP10h",
                "TPP8h",
                "TPP9h",
                "TTP7h",
            ],
            line_freq=60.0,
            software=None,
            cap_manufacturer=None,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=False,
                has_emg=True,
                emg_channels=4,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=54,
            health_status="healthy",
            gender={"female": 25, "male": 29},
            age_mean=None,
            age_min=24,
            age_max=35,
            handedness={"right": 50, "left": 2, "ambidexter": 2},
            bci_experience="mixed",
            clinical_population=None,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events={"left_hand": 2, "right_hand": 1},
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=4.0,
            study_design="Binary-class motor imagery (left/right hand grasping). Two sessions on different days, each with offline training and online test phases of 100 trials each.",
            feedback_type="visual",
            stimulus_type="arrow",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
            has_training_test_split=True,
            synchronicity="synchronous",
            instructions="Subjects performed the imagery task of grasping with the appropriate hand for 4 s when the right or left arrow appeared as a visual cue. First 3 s of each trial began with a black fixation cross to prepare subjects for the MI task. After each task, the screen remained blank for 6 s (± 1.5 s).",
            tasks=["MI"],
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
            doi="10.1093/gigascience/giz002",
            description="EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy. Includes MI, ERP, and SSVEP paradigms with a large number of subjects over multiple sessions.",
            investigators=[
                "Min-Ho Lee",
                "O-Yeon Kwon",
                "Yong-Jeong Kim",
                "Hong-Kyung Kim",
                "Young-Eun Lee",
                "John Williamson",
                "Siamac Fazli",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            country="KR",
            publication_year=2019,
            senior_author="Seong-Whan Lee",
            contact_info=["sw.lee@korea.ac.kr"],
            institution_address="145 Anam-ro, Seongbuk-gu, Seoul, 02841, Korea",
            institution_department="Department of Brain and Cognitive Engineering",
            keywords=[
                "EEG datasets",
                "brain-computer interface",
                "event-related potential",
                "steady-state visually evoked potential",
                "motor-imagery",
                "OpenBMI toolbox",
                "BCI illiteracy",
            ],
            how_to_acknowledge="This is an Open Access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.",
            license="GPL-3.0",
            repository="GigaDB",
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+LDA", "CSSP", "FBCSP", "BSSFO"],
            feature_extraction=["CSP", "CSSP", "FBCSP", "BSSFO", "log-variance"],
            frequency_bands={
                "mu": [8.0, 12.0],
                "analyzed_range": [8.0, 30.0],
            },
            spatial_filters=["CSP", "CSSP", "FBCSP", "BSSFO"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="train-test split",
            evaluation_type=["within_session", "cross_session"],
        ),
        performance={
            "accuracy_percent": 71.1,
            "accuracy_std": 0.15,
            "illiteracy_rate_percent": 53.7,
            "session1_accuracy": 70.0,
            "session2_accuracy": 72.2,
        },
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=3.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=200,
            n_trials_per_class={"left_hand": 100, "right_hand": 100},
            trials_context="100 trials per session per phase (50 per class per phase). Training: 50 left + 50 right. Test: 50 left + 50 right. Total per session: 200.",
        ),
        file_format="MAT",
        data_processed=False,
        abstract="Electroencephalography (EEG)-based brain-computer interface (BCI) systems are mainly divided into three major paradigms: motor imagery (MI), event-related potential (ERP), and steady-state visually evoked potential (SSVEP). Here, we present a BCI dataset that includes the three major BCI paradigms with a large number of subjects over multiple sessions. In addition, information about the psychological and physiological conditions of BCI users was obtained using a questionnaire, and task-unrelated parameters such as resting state, artifacts, and electromyography of both arms were also recorded. We evaluated the decoding accuracies for the individual paradigms and determined performance variations across both subjects and sessions. Furthermore, we looked for more general, severe cases of BCI illiteracy than have been previously reported in the literature. Average decoding accuracies across all subjects and sessions were 71.1% (± 0.15), 96.7% (± 0.05), and 95.1% (± 0.09), and rates of BCI illiteracy were 53.7%, 11.1%, and 10.2% for MI, ERP, and SSVEP, respectively. Compared to the ERP and SSVEP paradigms, the MI paradigm exhibited large performance variations between both subjects and sessions. Furthermore, we found that 27.8% (15 out of 54) of users were universally BCI literate, i.e., they were able to proficiently perform all three paradigms. Interestingly, we found no universally illiterate BCI user, i.e., all participants were able to control at least one type of BCI system.",
        methodology="Experimental procedure: 54 healthy subjects participated in two sessions on different days. Each session consisted of three BCI paradigms performed sequentially: ERP speller (36 symbols, row-column presentation with face stimuli), MI task (binary left/right hand imagery), and SSVEP (four target frequencies: 5.45, 6.67, 8.57, 12 Hz). Each paradigm had offline training and online test phases. EEG recorded at 1000 Hz with 62 Ag/AgCl electrodes using BrainAmp amplifier, nose-referenced, grounded to AFz. Impedance maintained below 10 kOhm. Subjects seated 60 cm from 21-inch LCD monitor. Questionnaires collected demographic, physiological, and psychological data. Artifact data (eye blinking, eye movements, teeth clenching, arm flexing) and resting state EEG also recorded. Total experiment duration: ~205 minutes per session.",
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
            channel_types={"eeg": 62, "emg": 4},
            montage="standard_1005",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nasion",
            ground="AFz",
            impedance_threshold_kohm=10,
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
                "EMG1",
                "EMG2",
                "EMG3",
                "EMG4",
                "F10",
                "F3",
                "F4",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FT10",
                "FT9",
                "FTT10h",
                "FTT9h",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P7",
                "P8",
                "PO10",
                "PO3",
                "PO4",
                "PO9",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP10",
                "TP7",
                "TP8",
                "TP9",
                "TPP10h",
                "TPP8h",
                "TPP9h",
                "TTP7h",
            ],
            line_freq=60.0,
            software="OpenBMI",
            cap_manufacturer="Brain Products",
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=False,
                has_emg=True,
                emg_channels=4,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=54,
            health_status="healthy",
            gender={"female": 25, "male": 29},
            age_mean=29.5,
            age_min=24,
            age_max=35,
            handedness="right",
            bci_experience="mixed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 1, "NonTarget": 2},
            paradigm="p300",
            n_classes=2,
            class_labels=["target", "non_target"],
            trial_duration=None,
            study_design="36-symbol ERP row-column speller with random-set presentation and face stimuli, offline training and online test phases",
            feedback_type="visual",
            stimulus_type="rc_speller",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            has_training_test_split=True,
            instructions="Subjects were asked to copy-spell given sentences by gazing at target characters on screen. In training: 'NEURAL NETWORKS AND DEEP LEARNING' (33 characters), in test: 'PATTERN RECOGNITION MACHINE LEARNING' (36 characters). Participants counted number of times each target character flashed.",
            task_type="copy_spelling",
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/gigascience/giz002",
            description="EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy",
            investigators=[
                "Min-Ho Lee",
                "O-Yeon Kwon",
                "Yong-Jeong Kim",
                "Hong-Kyung Kim",
                "Young-Eun Lee",
                "John Williamson",
                "Siamac Fazli",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            institution_department="Department of Brain and Cognitive Engineering",
            institution_address="145 Anam-ro, Seongbuk-gu, Seoul, 02841, Korea",
            country="KR",
            publication_year=2019,
            senior_author="Seong-Whan Lee",
            contact_info=[
                "sw.lee@korea.ac.kr",
                "Tel: +82-2-3290-3197",
                "Fax: +82-2-3290-3583",
            ],
            keywords=[
                "EEG datasets",
                "brain-computer interface",
                "event-related potential",
                "steady-state visually evoked potential",
                "motor-imagery",
                "OpenBMI toolbox",
                "BCI illiteracy",
            ],
            license="GPL-3.0",
            repository="GigaDB",
        ),
        sessions_per_subject=2,
        runs_per_session=2,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["Mean Amplitudes"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="training-test split",
            evaluation_type=["within_session", "cross_session"],
        ),
        performance={
            "accuracy_percent": 96.7,
            "accuracy_std": 0.05,
            "illiteracy_rate": 11.1,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=36,
            n_repetitions=5,
            isi_ms=135.0,
            soa_ms=215.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials={"training": 1980, "test": 2160},
            trials_context="Training: copy-spell 'NEURAL NETWORKS AND DEEP LEARNING' (33 characters). Test: copy-spell 'PATTERN RECOGNITION MACHINE LEARNING' (36 characters). Each character received 5 sequences of 12 flashes (60 flashes total).",
        ),
        file_format="MAT",
        data_processed=False,
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
            channel_types={"eeg": 62, "emg": 4},
            montage="standard_1005",
            hardware="BrainAmp",
            sensor_type="Ag/AgCl",
            reference="nasion",
            ground="AFz",
            impedance_threshold_kohm=10,
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
                "EMG1",
                "EMG2",
                "EMG3",
                "EMG4",
                "F10",
                "F3",
                "F4",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FT10",
                "FT9",
                "FTT10h",
                "FTT9h",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P7",
                "P8",
                "PO10",
                "PO3",
                "PO4",
                "PO9",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP10",
                "TP7",
                "TP8",
                "TP9",
                "TPP10h",
                "TPP8h",
                "TPP9h",
                "TTP7h",
            ],
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=False,
                has_emg=True,
                emg_channels=4,
            ),
            software="OpenBMI",
        ),
        participants=ParticipantMetadata(
            n_subjects=54,
            health_status="healthy",
            gender={"female": 25, "male": 29},
            age_mean=None,
            age_min=24,
            age_max=35,
            handedness=None,
            bci_experience="mixed",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events={"12.0": 1, "8.57": 2, "6.67": 3, "5.45": 4},
            paradigm="ssvep",
            n_classes=4,
            class_labels=["down", "right", "left", "up"],
            trial_duration=4.0,
            study_design="Four-target SSVEP paradigm with frequencies 5.45, 6.67, 8.57, and 12 Hz presented in four screen positions (down, right, left, up). Training phase (offline) and test phase (online with visual feedback). 25 trials per target frequency per phase.",
            feedback_type="visual",
            stimulus_type="flickering_box",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
            has_training_test_split=True,
            instructions="Participants were asked to fixate the center of a black screen and then to gaze in the direction where the target stimulus was highlighted in a different color",
            synchronicity="synchronous",
            task_type="selective_attention",
        ),
        documentation=DocumentationMetadata(
            doi="10.1093/gigascience/giz002",
            description="EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy",
            investigators=[
                "Min-Ho Lee",
                "O-Yeon Kwon",
                "Yong-Jeong Kim",
                "Hong-Kyung Kim",
                "Young-Eun Lee",
                "John Williamson",
                "Siamac Fazli",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            institution_department="Department of Brain and Cognitive Engineering",
            institution_address="145 Anam-ro, Seongbuk-gu, Seoul, 02841, Korea",
            country="KR",
            publication_year=2019,
            senior_author="Seong-Whan Lee",
            contact_info=[
                "sw.lee@korea.ac.kr",
                "Tel: +82-2-3290-3197",
                "Fax: +82-2-3290-3583",
            ],
            keywords=[
                "EEG datasets",
                "brain-computer interface",
                "event-related potential",
                "steady-state visually evoked potential",
                "motor-imagery",
                "OpenBMI toolbox",
                "BCI illiteracy",
            ],
            license="GPL-3.0",
            repository="GigaDB",
        ),
        sessions_per_subject=2,
        runs_per_session=1,
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
            ],
            downsampled_to_hz=100,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CCA"],
            feature_extraction=["CCA", "PSD"],
        ),
        performance={
            "accuracy_percent": 95.1,
            "illiteracy_rate": 10.2,
        },
        bci_application=BCIApplicationMetadata(
            applications=["control"],
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            n_targets=4,
            stimulus_frequencies_hz=[5.45, 6.67, 8.57, 12.0],
        ),
        data_structure=DataStructureMetadata(
            n_trials=200,
            n_trials_per_class={"5.45": 25, "6.67": 25, "8.57": 25, "12.0": 25},
            trials_context="100 trials (4 classes × 25 trials) in offline training phase and 100 trials in online test phase for SSVEP",
        ),
        file_format="MAT",
        data_processed=True,
        abstract="Electroencephalography (EEG)-based brain-computer interface (BCI) systems are mainly divided into three major paradigms: motor imagery (MI), event-related potential (ERP), and steady-state visually evoked potential (SSVEP). Here, we present a BCI dataset that includes the three major BCI paradigms with a large number of subjects over multiple sessions. In addition, information about the psychological and physiological conditions of BCI users was obtained using a questionnaire, and task-unrelated parameters such as resting state, artifacts, and electromyography of both arms were also recorded. We evaluated the decoding accuracies for the individual paradigms and determined performance variations across both subjects and sessions. Furthermore, we looked for more general, severe cases of BCI illiteracy than have been previously reported in the literature. Average decoding accuracies across all subjects and sessions were 71.1% (± 0.15), 96.7% (± 0.05), and 95.1% (± 0.09), and rates of BCI illiteracy were 53.7%, 11.1%, and 10.2% for MI, ERP, and SSVEP, respectively. Compared to the ERP and SSVEP paradigms, the MI paradigm exhibited large performance variations between both subjects and sessions. Furthermore, we found that 27.8% (15 out of 54) of users were universally BCI literate, i.e., they were able to proficiently perform all three paradigms. Interestingly, we found no universally illiterate BCI user, i.e., all participants were able to control at least one type of BCI system.",
        methodology="Three BCI paradigms (MI, ERP, SSVEP) were recorded sequentially from 54 subjects over 2 sessions on different days. EEG recorded at 1000 Hz with 62 Ag/AgCl electrodes using BrainAmp hardware, down-sampled to 100 Hz for analysis. Impedances maintained below 10 kΩ. Subjects seated 60 cm from 21-inch LCD monitor (60 Hz, 1600×1200). SSVEP paradigm: 4 target frequencies (5.45, 6.67, 8.57, 12 Hz) presented in 4 positions (down, right, left, up). Each stimulus presented for 4 s with 6 s ISI. Each target presented 25 times (100 trials total per phase). Training and test phases for online feedback. CCA used for classification. Resting state data recorded before and after each task. Questionnaires collected psychological and physiological conditions.",
    )

    __init__ = partialmethod(Lee2019.__init__, "SSVEP")
