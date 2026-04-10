import glob
import os
import re
import zipfile

import mne

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
from moabb.datasets.utils import stim_channels_with_selected_ids
from moabb.utils import _handle_deprecated_kwargs


# New freidok URL - the old fedora URLs no longer work
SPOT_PILOT_P300_URL = "https://freidok.uni-freiburg.de/dnb/download/154576"


class Sosulski2019(BaseDataset):
    """P300 dataset from initial spot study.

    Dataset [1]_, study on spatial transfer between SOAs [2]_, actual paradigm / online optimization [3]_.

    **Dataset description**
    This dataset contains multiple small trials of an auditory oddball paradigm. The paradigm presented two different
    sinusoidal tones. A low-pitched (500 Hz, 40 ms duration) non-target tone and a high-pitched (1000 Hz,
    40 ms duration) target tone. Subjects were instructed to attend to the high-pitched target tones and ignore the
    low-pitched tones.

    One trial (= one file) consisted of 90 tones, 15 targets and 75 non-targets. The order was pseudo-randomized in a
    way that at least two non-target tones occur between two target tones. Additionally, if you split the 90 tones of
    one trial into consecutive sets of six tones, there will always be exactly one target and five non-target tones
    in each set.

    In the first part of the experiment (run 1), each subject performed 50-70 trials with various different stimulus
    onset asynchronies (SOAs) -- i.e. the time between the onset of successive tones -- for each trial. In the second
    part (run 2), 4-5 SOAs were played, with blocks of 5 trials having the same SOA. All SOAs were in the range of 60
    ms to 600 ms. Regardless of the experiment part, after a set of five trials, subjects were given the opportunity
    to take a short break to e.g. drink etc.

    Finally, before and after each run, resting data was recorded. One minute with eyes open and one minute with eyes
    closed, i.e. in total four minutes of resting data are available for each subject.

    Data was recorded using a BrainAmp DC (BrainVision) amplifier and a 31 passive electrode EasyCap. The cap was
    placed according to the extended 10-20 electrode layout. The reference electrode was placed on the nose. Before
    recording, the cap was prepared such that impedances on all electrodes were around 20 kOhm. The EEG signal was
    recorded at 1000 Hz.

    The data contains 31 scalp channels, one EOG channel and five miscellaneous non-EEG signal channels. However,
    only scalp EEG and the EOG channel is available in all subjects. The markers in the marker file indicate the
    onset of target tones (21) and non-target tones (1).

    .. caution::

       Note that this wrapper currently only loads the second part of the experiment and uses pseudo-sessions
       to achieve the functionality to handle different conditions in MOABB. As a result, the statistical testing
       features of MOABB cannot be used for this dataset.

    References
    ----------

    .. [1] Sosulski, J., Tangermann, M.: Electroencephalogram signals recorded from 13 healthy subjects during an
           auditory oddball paradigm under different stimulus onset asynchrony conditions.
           Dataset. DOI: 10.6094/UNIFR/154576

    .. [2] Sosulski, J., Tangermann, M.: Spatial filters for auditory evoked potentials transfer between different
           experimental conditions. Graz BCI Conference. 2019.

    .. [3] Sosulski, J., Hübner, D., Klein, A., Tangermann, M.:  Online Optimization of Stimulation Speed in
           an Auditory Brain-Computer Interface under Time Constraints. arXiv preprint. 2021.

    Notes
    -----

    .. versionadded:: 0.4.5
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=31,
            channel_types={"eeg": 31, "eog": 1, "misc": 5},
            montage="standard_1020",
            hardware="BrainProducts BrainAmp DC",
            sensor_type="passive Ag/AgCl",
            reference="nose",
            software=None,
            sensors=[
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
                "EOGvu",
                "F10",
                "F3",
                "F4",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC5",
                "FC6",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "P10",
                "P3",
                "P4",
                "P7",
                "P8",
                "P9",
                "Pz",
                "T7",
                "T8",
                "x_EMGl",
                "x_GSR",
                "x_Optic",
                "x_Pulse",
                "x_Respi",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True, eog_channels=1, eog_type=["vertical"]
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=13,
            health_status="healthy",
            gender={"male": 5, "female": 8},
            age_mean=22.7,
            age_std=1.64,
            age_min=20,
            age_max=26,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 21, "NonTarget": 1},
            paradigm="p300",
            n_classes=2,
            class_labels=["target", "non-target"],
            trial_duration=None,
            study_design="Subjects focused attention on target tones (1000 Hz) and ignored non-target tones (500 Hz) presented via speaker at 65 cm distance. One trial consisted of 15 target and 75 non-target stimuli in pseudo-random order with at least two non-target tones between target tones. The experiment was split into optimization and validation parts.",
            stimulus_type="oddball",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="online",
            stimulus_presentation={
                "target_tone_hz": "1000",
                "non_target_tone_hz": "500",
                "tone_duration_ms": "40",
                "distance_cm": "65",
            },
            instructions="Focus on the target tones (1000 Hz) and ignore the non-target tones (500 Hz). Refrain from blinking and movement as much as possible.",
        ),
        documentation=DocumentationMetadata(
            doi="10.48550/arXiv.2109.06011",
            description="Auditory oddball ERP dataset from 13 healthy subjects. Two sinusoidal tones (target 1000 Hz, non-target 500 Hz) presented at various stimulus onset asynchronies (SOAs, 60-600 ms). 31-channel EEG recorded at 1000 Hz with BrainProducts BrainAmp DC. Raw BrainVision format data.",
            investigators=[
                "Jan Sosulski",
                "David Hübner",
                "Aaron Klein",
                "Michael Tangermann",
            ],
            institution="University of Freiburg",
            country="DE",
            data_url="https://freidok.uni-freiburg.de/data/154576",
            publication_year=2021,
            senior_author="Michael Tangermann",
            contact_info=[
                "jan.sosulski@blbt.uni-freiburg.de",
                "davhuebn@gmail.com",
                "kleinaa@cs.uni-freiburg.de",
                "michael.tangermann@donders.ru.nl",
            ],
            license="CC-BY-SA-4.0",
            funding=[
                "Cluster of Excellence BrainLinks-BrainTools funded by the German Research Foundation (DFG) [grant number EXC 1086]",
                "DFG project SuitAble [grant number TA 1258/1-1]",
                "state of Baden-Württemberg, Germany, through bwHPC and the German Research Foundation (DFG) [grant number INST 39/963-1 FUGG]",
            ],
            ethics_approval=[
                "Approved by the ethics committee of the university medical center of Freiburg"
            ],
            acknowledgements="Experiments were performed according to the Declaration of Helsinki.",
            keywords=[
                "Bayesian optimization",
                "individual experimental parameters",
                "brain-computer interfaces",
                "learning from small data",
                "auditory event-related potentials",
                "closed-loop parameter optimization",
            ],
            repository="FreiDok",
        ),
        tags=Tags(pathology=["Healthy"], modality=["Auditory"], type=["Research"]),
        preprocessing=PreprocessingMetadata(),
        signal_processing=SignalProcessingMetadata(
            classifiers=["rLDA", "Shrinkage LDA"],
            feature_extraction=["Mean amplitude in time intervals"],
            frequency_bands={"analyzed_range": [1.5, 40.0]},
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="13-fold", cv_folds=13, evaluation_type=["within_session"]
        ),
        performance={
            "auc": 0.701,
            "mean_auc_ucb": 0.701,
            "mean_auc_rand": 0.704,
            "mean_auc_p300_ucb": 0.670,
            "mean_auc_p300_rand": 0.681,
            "mean_auc_fixed60": 0.517,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication"], online_feedback=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=1,
            soa_ms=None,  # Variable, optimized per subject between 60-600 ms
            isi_ms=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials="Variable: optimization part used time-limited trials (20 minutes per strategy), validation part used 20 trials per SOA",
            n_trials_per_class={
                "target": "13 per trial (after preprocessing, originally 15)",
                "non_target": "65 per trial (after preprocessing, originally 75)",
            },
            trials_context="Each trial consisted of 90 stimuli (15 target, 75 non-target). After preprocessing (removing first and last 6 epochs), 78 data points available per trial: 13 target and 65 non-target epochs.",
        ),
        sessions_per_subject=80,
        runs_per_session=1,
        data_processed=False,
        file_format="brainvision",
        abstract="The decoding of brain signals recorded via, e.g., an electroencephalogram, using machine learning is key to brain-computer interfaces (BCIs). Stimulation parameters or other experimental settings of the BCI protocol typically are chosen according to the literature. The decoding performance directly depends on the choice of parameters, as they influence the elicited brain signals and optimal parameters are subject-dependent. Thus a fast and automated selection procedure for experimental parameters could greatly improve the usability of BCIs. We evaluate a standalone random search and a combined Bayesian optimization with random search into a closed-loop auditory event-related potential protocol. We aimed at finding the individually best stimulation speed—also known as stimulus onset asynchrony (SOA)—that maximizes the classification performance of a regularized linear discriminant analysis.",
        methodology="The experiment was divided into two parts: (1) Optimization part: four strategies (AUC-ucb, AUC-rand, P300-ucb, P300-rand) each allocated 20 minutes to find optimal SOA. Strategies alternated to minimize non-stationarity effects. (2) Validation part: evaluated SOAs from each optimization strategy plus fixed 60ms SOA using 20 trials each (in blocks of 5 trials). Features were mean amplitudes in 5 time intervals ([100, 170], [171, 230], [231, 300], [301, 410], [411, 500] ms) across 31 channels (155 dimensions total). Classification used rLDA with automatic shrinkage regularization and 13-fold cross-validation on single trials.",
    )

    def __init__(
        self,
        use_soas_as_sessions=True,
        load_soa_60=False,
        reject_non_iid=False,
        interval=None,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {
            "UseSoasAsSessions": "use_soas_as_sessions",
            "LoadSoa60": "load_soa_60",
            "RejectNonIid": "reject_non_iid",
            "Interval": "interval",
            "Subjects": "subjects",
            "Sessions": "sessions",
        }
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "Sosulski2019")
        use_soas_as_sessions = resolved.get("use_soas_as_sessions", use_soas_as_sessions)
        load_soa_60 = resolved.get("load_soa_60", load_soa_60)
        reject_non_iid = resolved.get("reject_non_iid", reject_non_iid)
        interval = resolved.get("interval", interval)
        subjects = resolved.get("subjects", subjects)
        sessions = resolved.get("sessions", sessions)

        """
        :param use_soa_as_sessions: 1800 epochs were recorded at different SOAs each. Depending on
        the subject between 3 and 4 (4-5 if 60 is loaded). Training classifiers on mixtures of SOAs
        rarely is useful. Setting this to True loads these as individual sessions for e.g.
        WithinSessionEvaluation.
        :param load_soa_60: whether to load SOA 60. Note that this was always recorded, but the
        recorded ERP was extremely weak (as expected).
        :param reject_non_iid: if true removes the first 6 and last 6 epochs of each trial.
        """
        self.load_soa_60 = load_soa_60
        self.reject_non_iid = reject_non_iid
        self.stimulus_modality = "tone_oddball"
        self.n_channels = 31
        self.use_soas_as_sessions = use_soas_as_sessions
        self.description_map = {"Stimulus/S 21": "Target", "Stimulus/S  1": "NonTarget"}
        self.events = {"Target": 21, "NonTarget": 1}
        code = "Sosulski2019"
        interval = [-0.2, 1] if interval is None else interval
        super().__init__(
            subjects=list(range(1, 13 + 1)),
            sessions_per_subject=80,
            events=self.events,
            code=code,
            interval=interval,
            paradigm="p300",
            doi="10.6094/UNIFR/154576",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    @staticmethod
    def _map_subject_to_filenumber(subject_number):
        # The ordering of the uploaded files on freidok makes no sense, this function maps subject_numbers to corresponding files
        mapping = [5, 2, 4, 6, 3, 1, 10, 7, 12, 9, 8, 11, 13]
        return mapping[subject_number - 1]

    @staticmethod
    def filename_trial_info_extraction(filepath):
        info_pattern = "Oddball_Run_([0-9]+)_Trial_([0-9]+)_SOA_[0-9]\\.([0-9]+)\\.vhdr"
        filename = filepath.split(os.path.sep)[-1]
        trial_info = {}
        re_matches = re.match(info_pattern, filename)
        trial_info["run"] = int(re_matches.group(1))
        trial_info["trial"] = int(re_matches.group(2))
        trial_info["soa"] = int(re_matches.group(3))
        return trial_info

    def _get_single_run_data(self, file_path):
        non_scalp_channels = ["EOGvu", "x_EMGl", "x_GSR", "x_Respi", "x_Pulse", "x_Optic"]
        raw = mne.io.read_raw_brainvision(
            file_path, misc=non_scalp_channels, preload=True
        )
        raw.set_montage("standard_1020")
        if self.reject_non_iid:
            raw.set_annotations(raw.annotations[7:85])  # non-iid rejection
        raw.annotations.rename(self.description_map)
        return stim_channels_with_selected_ids(raw, self.events)

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""

        file_path_list = self.data_path(subject)
        sessions = {}

        for p_i, file_path in enumerate(file_path_list):
            file_exp_info = Sosulski2019.filename_trial_info_extraction(file_path)
            soa = file_exp_info["soa"]
            # trial = file_exp_info["trial"]
            if soa == 60 and not self.load_soa_60:
                continue
            if self.use_soas_as_sessions:
                session_name = f"{p_i}soa{soa}"
            else:
                session_name = "0"

            if session_name not in sessions.keys():
                sessions[session_name] = {}

            if self.use_soas_as_sessions:
                run_name = f"0soa{soa}"
            else:
                run_name = f"{p_i}soa{soa}"
            sessions[session_name][run_name] = self._get_single_run_data(file_path)

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # Download the main ZIP file containing all subjects
        path_zip = dl.data_dl(SPOT_PILOT_P300_URL, "spot")
        path_base = os.path.dirname(path_zip)
        path_extracted = os.path.join(path_base, "extracted")

        # Extract main ZIP if not already done
        if not os.path.isdir(path_extracted):
            with zipfile.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path_extracted)

        # Find and extract subject-specific ZIP
        subject_zip_name = f"subject{subject}.zip"
        subject_zip_path = os.path.join(path_extracted, subject_zip_name)
        path_folder = os.path.join(path_extracted, f"subject{subject}")

        if not os.path.isdir(path_folder):
            if os.path.exists(subject_zip_path):
                with zipfile.ZipFile(subject_zip_path, "r") as zip_ref:
                    zip_ref.extractall(path_extracted)

        # get the path to all files
        # We only load data from the second run. The first run is a potpourri of SOAs
        pattern = "/*Run_2*.vhdr"
        subject_paths = glob.glob(path_folder + pattern)
        return sorted(subject_paths)
