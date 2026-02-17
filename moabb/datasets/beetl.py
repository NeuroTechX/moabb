import logging
import os
import zipfile
from pathlib import Path

import mne
import numpy as np
import pooch

import moabb.datasets.download as dl
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
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

from .base import BaseDataset
from .download import get_dataset_path


LOGGER = logging.getLogger(__name__)
BASE_URL = "https://ndownloader.figshare.com/files/"

LEADERBOARD_ARTICLE_ID = 14839650
FINAL_EVALUATION_ARTICLE_ID = 16586213
FINAL_LABEL_TXT_ARTICLE_ID = 21602622


class Beetl2021_A(BaseDataset):
    """Motor Imagery dataset from BEETL Competition - Dataset A.

    **Dataset description**

    Dataset A contains data from subjects with 500 Hz sampling rate and 63 EEG channels.
    In the leaderboard phase, this includes subjects 1-2, while in the final phase it includes
    subjects 1-3.

    Note: for the BEETL competition, there was a leaderboard phase and a final phase. Both phases
    contained data from two datasets, A and B. However, during leaderboard phase, dataset A contained
    data from subjects 1-2, while dataset B contained data from subjects 3-5. During the final phase,
    dataset A contained data from subjects 1-3, while dataset B contained data from subjects 4-5.

    Note: for the competition the data is cut into 4 second trials, here the data is concatenated
    into one session! In order to get the data as provided in the competition, the data has to be
    cut into 4 second trials.

    For the leaderboard phase, the dataset contains only training data, while for the final phase it
    includes both training and testing data. To learn more about the datasets in detail see [1]_.
    To learn more about the competition see [2]_.

    For benchmarking the BEETL competition use phase "final", train on training data benchmark on testing data.

    Data is sampled at 500 Hz and contains 63 EEG channels. The data underwent frequency-domain preprocessing
    using a bandpass filter (1-100 Hz) and a 50 Hz notch filter to attenuate power line interference.

    Motor imagery tasks include:
    - Rest (label 0)
    - Left hand (label 1)
    - Right hand (label 2)
    - Feet (label 3)

    Attributes
    ----------
    phase : str
        Either "leaderboard" or "final"

    References
    ----------
    .. [1] Wei, X., Faisal, A. A., Grosse-Wentrup, M., Gramfort, A.,
        Chevallier, S., Jayaram, V., ... & Tempczyk, P. (2022, July). 2021
        BEETL competition: Advancing transfer learning for subject independence
        and heterogeneous EEG data sets. In NeurIPS 2021 Competitions and
        Demonstrations Track (pp. 205-219). PMLR.
    .. [2] Competition: https://beetl.ai/introduction
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=63,
            channel_types={"eeg": 63},
            sensors=[
                "Fp1",
                "Fz",
                "F3",
                "F7",
                "FT9",
                "FC5",
                "FC1",
                "C3",
                "T7",
                "TP9",
                "CP5",
                "CP1",
                "Pz",
                "P3",
                "P7",
                "O1",
                "Oz",
                "O2",
                "P4",
                "P8",
                "TP10",
                "CP6",
                "CP2",
                "C4",
                "T8",
                "FT10",
                "FC6",
                "FC2",
                "F4",
                "F8",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "F1",
                "F5",
                "FT7",
                "FC3",
                "FCz",
                "C1",
                "C5",
                "TP7",
                "CP3",
                "P1",
                "P5",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "P6",
                "P2",
                "CPz",
                "CP4",
                "TP8",
                "C6",
                "C2",
                "FC4",
                "FT8",
                "F6",
                "F2",
                "AF4",
                "AF8",
            ],
            montage="standard_1005",
            sensor_type="EEG",
            filters="1-100 Hz bandpass, 50 Hz notch",
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=3,
            health_status="healthy",
            bci_experience="mixed",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="motor_imagery_4_class",
            events={"rest": 0, "left_hand": 1, "right_hand": 2, "feet": 3},
            n_classes=4,
            class_labels=["rest", "left_hand", "right_hand", "feet"],
            trial_duration=4.0,
            mode="offline",
            synchronicity="synchronous",
            has_training_test_split=True,
            feedback_type="online_racing_game",
            study_design="transfer_learning_evaluation",
            study_domain="BCI",
        ),
        documentation=DocumentationMetadata(
            doi="10.48550/arXiv.2202.12950",
            description="Motor Imagery dataset from BEETL Competition - Dataset A. Part of the NeurIPS 2021 BEETL competition focusing on transfer learning for subject independence and heterogeneous EEG datasets.",
            investigators=[
                "Xiaoxi Wei",
                "A. Aldo Faisal",
                "Moritz Grosse-Wentrup",
                "Alexandre Gramfort",
                "Sylvain Chevallier",
                "Vinay Jayaram",
                "Camille Jeunet",
            ],
            institution="Multiple institutions",
            country="GB",
            repository="Figshare",
            data_url="https://beetl.ai/data",
            publication_year=2022,
            senior_author="A. Aldo Faisal",
            contact_info=["xiaoxi.wei18@imperial.ac.uk", "aldo.faisal@imperial.ac.uk"],
            associated_paper_doi="10.48550/arXiv.2202.12950",
            keywords=[
                "machine learning",
                "transfer learning",
                "domain adaptation",
                "Brain-Computer-Interfaces",
                "EEG",
                "neuroscience",
                "NeurIPS2021",
                "motor imagery",
                "cross-dataset",
                "subject independence",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        sessions=["0"],
        contributing_labs=[
            "Brain & Behaviour Lab, Imperial College London",
            "University of Bayreuth",
            "University of Vienna",
            "Universite Paris-Saclay",
            "UVSQ",
            "Reality Labs",
            "University of Bordeaux",
        ],
        n_contributing_labs=7,
        data_processed=True,
        file_format="NPY",
        external_links={
            "source": "https://beetl.ai/data",
            "competition": "https://beetl.ai/challenge",
            "moabb": "https://github.com/NeuroTechX/moabb",
            "arxiv": "https://arxiv.org/abs/2202.12950",
        },
        tags=Tags(
            pathology=["healthy"],
            modality=["motor"],
            type=["imagery"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filter", "notch filter"],
            highpass_hz=1.0,
            lowpass_hz=100.0,
            bandpass=[1.0, 100.0],
            notch_hz=50.0,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "EEGInception",
                "EEGNet",
                "Shallow ConvNet",
                "Deep Sets",
                "SPDNet",
            ],
            feature_extraction=[
                "deep learning features",
                "covariance matrices",
                "Riemannian geometry",
            ],
            spatial_filters=[
                "Euclidean Alignment",
                "Label Alignment",
                "Riemannian Alignment",
            ],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="block-wise k-fold",
            cv_folds=5,
            evaluation_type=["cross-subject", "cross-dataset", "transfer-learning"],
        ),
        performance={
            "accuracy_percent": 76.33,
            "balanced_accuracy_task2": 76.33,
            "baseline_accuracy": 49.9,
            "cogitat_team_accuracy": 76.33,
            "wduong_team_accuracy": 71.33,
            "ms01_team_accuracy": 59.87,
        },
        bci_application=BCIApplicationMetadata(
            applications=["brain-computer interface", "transfer learning benchmark"],
            environment="lab",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["rest", "left_hand", "right_hand", "feet"],
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials={"training": "5 races", "testing": "10 races"},
            n_blocks=15,
            trials_context="Data collected during Cybathlon2020IC online racing game. Training data: 5 races per subject. Testing data: 10 races per subject. Each race contains multiple 4-second trials.",
        ),
        abstract="Transfer learning and meta-learning offer some of the most promising avenues to unlock the scalability of healthcare and consumer technologies driven by biosignal data. This is because current methods cannot generalise well across human subjects' data and handle learning from different heterogeneously collected data sets, thus limiting the scale of training data. On the other side, developments in transfer learning would benefit significantly from a real-world benchmark with immediate practical application. Therefore, we pick electroencephalography (EEG) as an exemplar for what makes biosignal machine learning hard. We design two transfer learning challenges around diagnostics and Brain-Computer Interfacing (BCI), that have to be solved in the face of low signal-to-noise ratios, major variability among subjects, differences in the data recording sessions and techniques, and even between the specific BCI tasks recorded in the dataset. Task 1 is centred on the field of medical diagnostics, addressing automatic sleep stage annotation across subjects. Task 2 is centred on Brain-Computer Interfacing (BCI), addressing motor imagery decoding across both subjects and data sets. The BEETL competition with its over 30 competing teams and its 3 winning entries brought attention to the potential of deep transfer learning and combinations of set theory and conventional machine learning techniques to overcome the challenges. The results set a new state-of-the-art for the real-world BEETL benchmark.",
        methodology="Dataset A is part of the BEETL Competition Task 2 (Motor Imagery). The data was collected in an online racing game format called Cybathlon2020IC. Dataset A contains data from subjects 1-3 (final phase) with 500 Hz sampling rate and 63 EEG channels. The data underwent frequency-domain preprocessing using a bandpass filter (1-100 Hz) and a 50 Hz notch filter to attenuate power line interference. The competition had two phases: leaderboard (subjects 1-2) and final (subjects 1-3). For the final phase, training data from 5 races and testing data from 10 races were provided for each subject. The data is segmented into 4-second trials. Motor imagery tasks include: Rest (label 0), Left hand (label 1), Right hand (label 2), and Feet (label 3). The challenge focused on transfer learning from multiple source datasets (Cho2017, BNCI2014, PhysionetMI) to target datasets (Weibo2014 and Cybathlon2020IC) with different EEG setups, electrode configurations, and task definitions. Winning solutions employed deep learning architectures (EEGInception, EEGNet), latent subject alignment methods (Deep Sets, statistical alignment), domain adaptation techniques (Euclidean Alignment, Label Alignment, Maximum Classifier Discrepancy), and Riemannian geometry approaches (SPDNet, MDRM).",
    )

    def __init__(self, phase="final"):
        """Initialize BEETL Dataset A.

        Parameters
        ----------
        phase : str
            Either "leaderboard" (subjects 1-2) or "final" (subjects 1-3)
        """
        if phase not in ["leaderboard", "final"]:
            raise ValueError("Phase must be either 'leaderboard' or 'final'")

        self.phase = phase
        subjects = list(range(1, 3)) if phase == "leaderboard" else list(range(1, 4))

        # Channel setup
        self.ch_names = [
            "Fp1",
            "Fz",
            "F3",
            "F7",
            "FT9",
            "FC5",
            "FC1",
            "C3",
            "T7",
            "TP9",
            "CP5",
            "CP1",
            "Pz",
            "P3",
            "P7",
            "O1",
            "Oz",
            "O2",
            "P4",
            "P8",
            "TP10",
            "CP6",
            "CP2",
            "C4",
            "T8",
            "FT10",
            "FC6",
            "FC2",
            "F4",
            "F8",
            "Fp2",
            "AF7",
            "AF3",
            "AFz",
            "F1",
            "F5",
            "FT7",
            "FC3",
            "FCz",
            "C1",
            "C5",
            "TP7",
            "CP3",
            "P1",
            "P5",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "P6",
            "P2",
            "CPz",
            "CP4",
            "TP8",
            "C6",
            "C2",
            "FC4",
            "FT8",
            "F6",
            "F2",
            "AF4",
            "AF8",
        ]

        self.sfreq = 500
        self.phase = phase

        super().__init__(
            subjects=subjects,
            sessions_per_subject=1,  # Data is concatenated into one session
            events=dict(rest=0, left_hand=1, right_hand=2, feet=3),
            code="Beetl2021-A",
            interval=[0, 4],  # 4s trial window
            paradigm="imagery",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)

        # Create MNE info
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=["eeg"] * len(self.ch_names),
        )

        phase_str = "leaderboardMI" if self.phase == "leaderboard" else "finalMI"
        base = Path(file_paths[0])
        subject_dir = base / phase_str / f"S{subject}"
        if not subject_dir.exists():
            # Backward compat: old extraction created double-nested dirs
            subject_dir = base / phase_str / phase_str / f"S{subject}"

        data_list = []
        labels_list = []

        # Load training data
        for race in range(1, 6):
            data_file = subject_dir / "training" / f"race{race}_padsData.npy"
            label_file = subject_dir / "training" / f"race{race}_padsLabel.npy"
            if data_file.exists() and label_file.exists():
                data_list.append(np.load(data_file, allow_pickle=True))
                labels_list.append(np.load(label_file, allow_pickle=True))

        data = np.concatenate(data_list)
        labels = np.concatenate(labels_list)

        # Create events array
        events = np.column_stack(
            (
                np.arange(0, len(labels) * data.shape[-1], data.shape[-1]),
                np.zeros(len(labels), dtype=int),
                labels,
            )
        )

        # Create Raw object
        event_desc = {int(code): name for name, code in self.event_id.items()}
        raw = mne.io.RawArray(np.hstack(data), info)
        raw.set_annotations(
            mne.annotations_from_events(
                events=events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        # Load test data
        test_data_list = []
        for race in range(6, 16):
            data_file = subject_dir / "testing" / f"race{race}_padsData.npy"
            if data_file.exists():
                test_data_list.append(np.load(data_file, allow_pickle=True))

        test_data = np.concatenate(test_data_list)

        # load labels from .txt
        test_labels = np.loadtxt(Path(file_paths[0]) / "final_MI_label.txt", dtype=int)
        subject_labels = test_labels[
            (subject - 1) * test_data.shape[0] : (subject) * test_data.shape[0]
        ]

        test_events = np.column_stack(
            (
                np.arange(
                    0, len(subject_labels) * test_data.shape[-1], test_data.shape[-1]
                ),
                np.zeros(len(subject_labels), dtype=int),
                subject_labels,
            )
        )

        # Create Raw object
        test_raw = mne.io.RawArray(np.hstack(test_data), info)
        test_raw.set_annotations(
            mne.annotations_from_events(
                events=test_events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        return {"0": {f"0{self.phase}train": raw, f"1{self.phase}test": test_raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return path to the data files."""
        if subject not in self.subject_list:
            raise ValueError(f"Subject {subject} not in {self.subject_list}")

        path = get_dataset_path("BEETL", path)
        base_path = Path(os.path.join(path, f"MNE-{self.code:s}-data"))
        # Create the directory if it doesn't exist
        base_path.mkdir(parents=True, exist_ok=True)

        # Skip Figshare API calls if data already exists locally
        phase_str = "leaderboardMI" if self.phase == "leaderboard" else "finalMI"
        subject_dir = base_path / phase_str / f"S{subject}"
        if not subject_dir.exists():
            subject_dir = base_path / phase_str / phase_str / f"S{subject}"
        label_file = base_path / "final_MI_label.txt"
        if not force_update and subject_dir.exists() and label_file.exists():
            return [str(base_path)]

        # Download data if needed
        for article_id in [LEADERBOARD_ARTICLE_ID, FINAL_EVALUATION_ARTICLE_ID]:
            file_list = dl.fs_get_file_list(article_id)
            hash_file_list = dl.fs_get_file_hash(file_list)
            id_file_list = dl.fs_get_file_id(file_list)

            for file_name in id_file_list.keys():
                file_path = os.path.join(base_path, file_name)
                extract_dir = base_path / os.path.splitext(file_name)[0]

                # Step 1: Download the zip file if not already downloaded
                if not os.path.exists(file_path):
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

                # Step 2: Unzip the file if not already extracted
                if not extract_dir.exists():
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(base_path)

        # Download labels for final phase
        file_list = dl.fs_get_file_list(FINAL_LABEL_TXT_ARTICLE_ID)
        hash_file_list = dl.fs_get_file_hash(file_list)
        id_file_list = dl.fs_get_file_id(file_list)

        for file_name in id_file_list.keys():
            fpath = base_path / file_name
            if (not fpath.exists() or force_update) and file_name == "final_MI_label.txt":
                fpath = base_path / file_name
                if not fpath.exists() or force_update:
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

        return [str(base_path)]


class Beetl2021_B(BaseDataset):
    """Motor Imagery dataset from BEETL Competition - Dataset B.

    **Dataset description**

    Dataset B contains data from subjects with 200 Hz sampling rate and 32 EEG channels.
    In the leaderboard phase, this includes subjects 3-5, while in the final phase it includes
    subjects 4-5.

    Note: for the BEETL competition, there was a leaderboard phase and a final phase. Both phases
    contained data from two datasets, A and B. However, during leaderboard phase, dataset A contained
    data from subjects 1-2, while dataset B contained data from subjects 3-5. During the final phase,
    dataset A contained data from subjects 1-3, while dataset B contained data from subjects 4-5.

    Note: for the competition the data is cut into 4 second trials, here the data is concatenated
    into one session! In order to get the data as provided in the competition, the data has to be
    cut into 4 second trials.

    For the leaderboard phase, the dataset contains only training data, while for the final phase it
    includes both training and testing data. To learn more about the datasets in detail see [1]_.
    To learn more about the competition see [2]_.

    For benchmarking the BEETL competition use phase "final", train on training data benchmark on testing data.

    The data was filtered using a highpass filter with a cutoff frequency of 1 Hz and a
    lowpass filter with a cutoff frequency of 100 Hz.

    Motor imagery tasks include:
    - Left hand (label 0)
    - Right hand (label 1)
    - Feet (label 2)
    - Rest (label 3)

    Attributes
    ----------
    phase : str
        Either "leaderboard" or "final"

    References
    ----------
    .. [1] Wei, X., Faisal, A. A., Grosse-Wentrup, M., Gramfort, A.,
        Chevallier, S., Jayaram, V., ... & Tempczyk, P. (2022, July). 2021
        BEETL competition: Advancing transfer learning for subject independence
        and heterogeneous EEG data sets. In NeurIPS 2021 Competitions and
        Demonstrations Track (pp. 205-219). PMLR.
    .. [2] Competition: https://beetl.ai/introduction
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=200.0,
            n_channels=32,
            channel_types={"eeg": 32},
            sensors=[
                "Fp1",
                "Fp2",
                "F3",
                "Fz",
                "F4",
                "FC5",
                "FC1",
                "FC2",
                "FC6",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
            ],
            sensor_type="EEG",
            filters="Bandpass filter (1-100 Hz), Notch filter (50 Hz)",
            line_freq=50.0,
            montage="standard_1005",
        ),
        participants=ParticipantMetadata(
            n_subjects=3,  # Beetl2021_B has subjects 3-5 in leaderboard, 4-5 in final
            health_status="healthy",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            task_type="motor imagery",
            events={
                "left_hand": 0,
                "right_hand": 1,
                "feet": 2,
                "rest": 3,
            },
            n_classes=4,
            class_labels=["left_hand", "right_hand", "feet", "rest"],
            trial_duration=4.0,
            study_design="cross-dataset transfer learning",
            study_domain="Brain-Computer Interface",
            feedback_type="online feedback",
            synchronicity="cued",
            mode="active",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            description="Motor Imagery dataset from BEETL Competition - Dataset B. This is part of the BEETL (Benchmarks for EEG Transfer Learning) competition Task 2, which focuses on 3-way motor imagery classification (left-hand, right-hand motor imagery and 'reject') across both subjects and datasets. Dataset B contains data with 200 Hz sampling rate and 32 EEG channels selected from the Weibo2014 dataset.",
            investigators=[
                "Xiaoxi Wei",
                "A. Aldo Faisal",
                "Moritz Grosse-Wentrup",
                "Alexandre Gramfort",
                "Sylvain Chevallier",
                "Vinay Jayaram",
                "Camille Jeunet",
                "Stylianos Bakas",
                "Siegfried Ludwig",
                "Konstantinos Barmpas",
                "Mehdi Bahri",
                "Yannis Panagakis",
                "Nikolaos Laskaris",
                "Dimitrios A. Adamos",
                "Stefanos Zafeiriou",
                "William C. Duong",
                "Stephen M. Gordon",
                "Vernon J. Lawhern",
                "Maciej Śliwowski",
                "Vincent Rouanne",
                "Piotr Tempczyk",
            ],
            institution="Imperial College London",
            country="United Kingdom",
            repository="Figshare",
            data_url="https://beetl.ai/data",
            publication_year=2022,
            senior_author="A. Aldo Faisal",
            associated_paper_doi="10.48550/arXiv.2202.12950",
            funding=[
                "Brain & Behaviour Lab, Imperial College London",
                "Institute of Artificial & Human Intelligence, University of Bayreuth",
                "Human Research and Engineering Directorate, DEVCOM Army Research Laboratory",
            ],
            institution_address="Imperial College London, United Kingdom",
            institution_department="Brain & Behaviour Lab",
            keywords=[
                "machine learning",
                "transfer learning",
                "domain adaptation",
                "Brain-Computer-Interfaces (BCI)",
                "EEG",
                "neuroscience",
                "motor imagery",
                "cross-dataset",
                "NeurIPS2021",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=2,  # training and testing
        sessions=["0leaderboardtrain", "1leaderboardtest", "0finaltrain", "1finaltest"],
        data_processed=True,
        file_format="numpy",
        external_links={
            "source": "https://beetl.ai/challenge",
            "competition_page": "https://beetl.ai/introduction",
            "moabb": "http://moabb.neurotechx.com/docs/datasets.html",
            "github": "https://github.com/NeuroTechX/moabb",
        },
        tags=Tags(
            pathology=["Healthy"],
            modality=["EEG"],
            type=["motor imagery", "BCI", "transfer learning"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "Bandpass filtering (1-100 Hz)",
                "Notch filtering (50 Hz)",
                "Channel selection (32 channels around motor cortex)",
                "Segmentation into 4-second trials",
            ],
            highpass_hz=1.0,
            lowpass_hz=100.0,
            notch_hz=50.0,
            filter_type="frequency-domain",
            notes="Data was filtered in frequency domain and cut into 4 second trials for the competition",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "EEGNet",
                "EEGInception",
                "DeepSleep",
                "Shallow ConvNet",
                "SPDNet",
                "Deep Sets",
                "MDRM",
            ],
            feature_extraction=[
                "Deep learning features",
                "Covariance matrices",
                "Riemannian geometry",
            ],
            spatial_filters=[
                "Euclidean Alignment",
                "Label Alignment",
                "Deep Set alignment",
                "Statistical alignment",
                "Maximum Classifier Discrepancy",
            ],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="block-wise cross-validation",
            cv_folds=5,
            evaluation_type=["cross-subject", "cross-dataset"],
        ),
        performance={
            "accuracy_percent": 76.33,
            "balanced_accuracy_cogitat": 76.33,
            "balanced_accuracy_wduong": 71.33,
            "balanced_accuracy_ms01": 59.87,
            "balanced_accuracy_baseline": 49.9,
        },
        bci_application=BCIApplicationMetadata(
            applications=[
                "motor imagery decoding",
                "BCI control",
                "transfer learning benchmark",
            ],
            environment="online racing game (Cybathlon2020IC)",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor imagery",
            imagery_tasks=["left_hand", "right_hand", "feet", "rest"],
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            trials_context="Training and testing data separated. Competition data cut into 4-second trials. Training has 5 races, testing has 10 races per subject.",
        ),
        abstract="Transfer learning and meta-learning offer some of the most promising avenues to unlock the scalability of healthcare and consumer technologies driven by biosignal data. This is because current methods cannot generalise well across human subjects' data and handle learning from different heterogeneously collected data sets, thus limiting the scale of training data. Task 2 is a 3-way motor imagery classification challenge (left-hand, right-hand motor imagery and 'reject') that gets at the heart of the problem of current BCI systems: motor imagery data is exhausting for subjects to record, and historically has been difficult to use in a cross-subject and cross-dataset manner. Dataset B contains data from the Weibo2014 dataset with 32 channels around the motor cortex selected, sampled at 200 Hz.",
        methodology="Task 2 is centred on transfer learning for BCI, addressing motor imagery decoding. The challenge lies in transferring from multiple data sets, which use different EEG setups comprising hundreds of users, to a set of new users that need to be up and running with only minutes worth of calibration data (transfer across subjects and data sets). Three source data sets (Cho2017, BNCI2014, PhysionetMI) are provided as training data. The algorithms are evaluated on new data sets with different setups, including differences in electrode channels, task definitions, and subjects. Dataset B is from Weibo2014 with 32 channels selected around motor cortex. For the leaderboard phase (subjects 3-5), only training data is provided. For the final phase (subjects 4-5), both training and testing data are included.",
    )

    def __init__(self, phase="final"):
        """Initialize BEETL Dataset B.

        Parameters
        ----------
        phase : str
            Either "leaderboard" (subjects 3-5) or "final" (subjects 4-5)
        """
        if phase not in ["leaderboard", "final"]:
            raise ValueError("Phase must be either 'leaderboard' or 'final'")

        self.phase = phase
        subjects = list(range(3, 6)) if phase == "leaderboard" else list(range(4, 6))

        super().__init__(
            subjects=subjects,
            sessions_per_subject=1,  # Data is concatenated into one session
            events=dict(left_hand=0, right_hand=1, feet=2, rest=3),
            code="Beetl2021-B",
            interval=[0, 4],  # 4s trial window
            paradigm="imagery",
        )

        self.ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "Fz",
            "F4",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
        ]
        self.sfreq = 200
        self.phase = phase

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)

        # Create MNE info
        info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=["eeg"] * len(self.ch_names),
        )

        # Load data
        phase_str = "leaderboardMI" if self.phase == "leaderboard" else "finalMI"
        base = Path(file_paths[0])
        subject_dir = base / phase_str / f"S{subject}"
        if not subject_dir.exists():
            # Backward compat: old extraction created double-nested dirs
            subject_dir = base / phase_str / phase_str / f"S{subject}"

        # Load training data
        train_data = np.load(
            subject_dir / "training" / f"training_s{subject}X.npy", allow_pickle=True
        )
        train_labels = np.load(
            subject_dir / "training" / f"training_s{subject}y.npy", allow_pickle=True
        )

        # Create events array
        events = np.column_stack(
            (
                np.arange(
                    0, len(train_labels) * train_data.shape[-1], train_data.shape[-1]
                ),
                np.zeros(len(train_labels), dtype=int),
                train_labels,
            )
        )

        # Create Raw object
        event_desc = {int(code): name for name, code in self.event_id.items()}
        raw = mne.io.RawArray(np.hstack(train_data * 1e-6), info)
        raw.set_annotations(
            mne.annotations_from_events(
                events=events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        # Load test data
        test_data = np.load(
            subject_dir / "testing" / f"testing_s{subject}X.npy", allow_pickle=True
        )
        # load labels from .txt
        test_labels = np.loadtxt(Path(file_paths[0]) / "final_MI_label.txt", dtype=int)
        subject_labels = test_labels[
            (subject - 1) * test_data.shape[0] : (subject) * test_data.shape[0]
        ]

        test_events = np.column_stack(
            (
                np.arange(
                    0, len(subject_labels) * test_data.shape[-1], test_data.shape[-1]
                ),
                np.zeros(len(subject_labels), dtype=int),
                subject_labels,
            )
        )

        # Create Raw object
        test_raw = mne.io.RawArray(np.hstack(test_data * 1e-6), info)
        test_raw.set_annotations(
            mne.annotations_from_events(
                events=test_events, event_desc=event_desc, sfreq=self.sfreq
            )
        )

        return {"0": {f"0{self.phase}train": raw, f"1{self.phase}test": test_raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return path to the data files."""
        if subject not in self.subject_list:
            raise ValueError(f"Subject {subject} not in {self.subject_list}")

        path = get_dataset_path("BEETL", path)
        base_path = Path(os.path.join(path, f"MNE-{self.code:s}-data"))

        # Create the directory if it doesn't exist
        base_path.mkdir(parents=True, exist_ok=True)

        # Skip Figshare API calls if data already exists locally
        phase_str = "leaderboardMI" if self.phase == "leaderboard" else "finalMI"
        subject_dir = base_path / phase_str / f"S{subject}"
        if not subject_dir.exists():
            subject_dir = base_path / phase_str / phase_str / f"S{subject}"
        label_file = base_path / "final_MI_label.txt"
        if not force_update and subject_dir.exists() and label_file.exists():
            return [str(base_path)]

        # Download data if needed
        for article_id in [LEADERBOARD_ARTICLE_ID, FINAL_EVALUATION_ARTICLE_ID]:
            file_list = dl.fs_get_file_list(article_id)
            hash_file_list = dl.fs_get_file_hash(file_list)
            id_file_list = dl.fs_get_file_id(file_list)

            for file_name in id_file_list.keys():
                file_path = os.path.join(base_path, file_name)
                extract_dir = base_path / os.path.splitext(file_name)[0]

                # Step 1: Download the zip file if not already downloaded
                if not os.path.exists(file_path):
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

                # Step 2: Unzip the file if not already extracted
                if not extract_dir.exists():
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(base_path)

        # Download labels for final phase
        file_list = dl.fs_get_file_list(FINAL_LABEL_TXT_ARTICLE_ID)
        hash_file_list = dl.fs_get_file_hash(file_list)
        id_file_list = dl.fs_get_file_id(file_list)

        for file_name in id_file_list.keys():
            fpath = base_path / file_name
            if (not fpath.exists() or force_update) and file_name == "final_MI_label.txt":
                fpath = base_path / file_name
                if not fpath.exists() or force_update:
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=base_path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

        return [str(base_path)]
