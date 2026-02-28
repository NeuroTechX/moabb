import logging
import os
import shutil

import mne
from mne.channels import make_standard_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
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


log = logging.getLogger(__name__)

GIN_URL = (
    "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"  # noqa
)


class Schirrmeister2017(BaseDataset):
    """High-gamma dataset described in Schirrmeister et al. 2017.

    Dataset from [1]_

    Our "High-Gamma Dataset" is a 128-electrode dataset (of which we later only use
    44 sensors covering the motor cortex, (see Section 2.7.1), obtained from 14
    healthy subjects (6 female, 2 left-handed, age 27.2 ± 3.6 (mean ± std)) with
    roughly 1000 (963.1 ± 150.9, mean ± std) four-second trials of executed
    movements divided into 13 runs per subject.  The four classes of movements were
    movements of either the left hand, the right hand, both feet, and rest (no
    movement, but same type of visual cue as for the other classes).  The training
    set consists of the approx.  880 trials of all runs except the last two runs,
    the test set of the approx.  160 trials of the last 2 runs.  This dataset was
    acquired in an EEG lab optimized for non-invasive detection of high- frequency
    movement-related EEG components (Ball et al., 2008; Darvas et al., 2010).

    Depending on the direction of a gray arrow that was shown on black back-
    ground, the subjects had to repetitively clench their toes (downward arrow),
    perform sequential finger-tapping of their left (leftward arrow) or right
    (rightward arrow) hand, or relax (upward arrow).  The movements were selected
    to require little proximal muscular activity while still being complex enough
    to keep subjects in- volved.  Within the 4-s trials, the subjects performed the
    repetitive movements at their own pace, which had to be maintained as long as
    the arrow was showing.  Per run, 80 arrows were displayed for 4 s each, with 3
    to 4 s of continuous random inter-trial interval.  The order of presentation
    was pseudo-randomized, with all four arrows being shown every four trials.
    Ideally 13 runs were performed to collect 260 trials of each movement and rest.
    The stimuli were presented and the data recorded with BCI2000 (Schalk et al.,
    2004).  The experiment was approved by the ethical committee of the University
    of Freiburg.

    References
    ----------

    .. [1] Schirrmeister, Robin Tibor, et al. "Deep learning with convolutional
           neural networks for EEG decoding and visualization." Human brain mapping 38.11
           (2017): 5391-5420.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=128,
            channel_types={"eeg": 128},
            hardware="128-electrode EEG system optimized for high-frequency detection",
            reference="Car",
            sensors=[
                "Fp1",
                "Fp2",
                "Fpz",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "FC5",
                "FC1",
                "FC2",
                "FC6",
                "M1",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "M2",
                "CP5",
                "CP1",
                "CP2",
                "CP6",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "POz",
                "O1",
                "Oz",
                "O2",
                "AF7",
                "AF3",
                "AF4",
                "AF8",
                "F5",
                "F1",
                "F2",
                "F6",
                "FC3",
                "FCz",
                "FC4",
                "C5",
                "C1",
                "C2",
                "C6",
                "CP3",
                "CPz",
                "CP4",
                "P5",
                "P1",
                "P2",
                "P6",
                "PO5",
                "PO3",
                "PO4",
                "PO6",
                "FT7",
                "FT8",
                "TP7",
                "TP8",
                "PO7",
                "PO8",
                "FT9",
                "FT10",
                "TPP9h",
                "TPP10h",
                "PO9",
                "PO10",
                "P9",
                "P10",
                "AFF1",
                "AFz",
                "AFF2",
                "FFC5h",
                "FFC3h",
                "FFC4h",
                "FFC6h",
                "FCC5h",
                "FCC3h",
                "FCC4h",
                "FCC6h",
                "CCP5h",
                "CCP3h",
                "CCP4h",
                "CCP6h",
                "CPP5h",
                "CPP3h",
                "CPP4h",
                "CPP6h",
                "PPO1",
                "PPO2",
                "I1",
                "Iz",
                "I2",
                "AFp3h",
                "AFp4h",
                "AFF5h",
                "AFF6h",
                "FFT7h",
                "FFC1h",
                "FFC2h",
                "FFT8h",
                "FTT9h",
                "FTT7h",
                "FCC1h",
                "FCC2h",
                "FTT8h",
                "FTT10h",
                "TTP7h",
                "CCP1h",
                "CCP2h",
                "TTP8h",
                "TPP7h",
                "CPP1h",
                "CPP2h",
                "TPP8h",
                "PPO9h",
                "PPO5h",
                "PPO6h",
                "PPO10h",
                "POO9h",
                "POO3h",
                "POO4h",
                "POO10h",
                "OI1h",
                "OI2h",
            ],
            line_freq=50.0,
            software="BCI2000",
            montage="standard_1005",
            sensor_type="EEG",
        ),
        participants=ParticipantMetadata(
            n_subjects=14,
            gender={"female": 6, "male": 8},
            age_mean=27.2,
            age_std=3.6,
            handedness={"right": 12, "left": 2},
            health_status="healthy",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "feet", "rest"],
            study_design="Executed movements including left hand (sequential finger-tapping), right hand (sequential finger-tapping), feet (repetitive toe clenching), and rest conditions",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="execution",
            trial_duration=4.0,
            synchronicity="cue-based",
            stimulus_presentation={
                "type": "gray arrow on black background",
                "direction_mapping": "downward=feet, leftward=left_hand, rightward=right_hand, upward=rest",
            },
            instructions="Subjects performed repetitive movements at their own pace when arrow was showing",
            events={"left_hand": 2, "right_hand": 1, "feet": 4, "rest": 3},
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1002/hbm.23730",
            publication_year=2017,
            investigators=[
                "Robin Tibor Schirrmeister",
                "Jost Tobias Springenberg",
                "Lukas Dominique Josef Fiederer",
                "Martin Glasstetter",
                "Katharina Eggensperger",
                "Michael Tangermann",
                "Frank Hutter",
                "Wolfram Burgard",
                "Tonio Ball",
            ],
            senior_author="Tonio Ball",
            institution="University of Freiburg",
            institution_department="Translational Neurotechnology Lab, Epilepsy Center, Medical Center",
            institution_address="Engelberger Str. 21, Freiburg 79106, Germany",
            country="Germany",
            funding=[
                "BrainLinks-BrainTools Cluster of Excellence (DFG) EXC1086",
                "Federal Ministry of Education and Research (BMBF) Motor-BIC 13GW0053D",
            ],
            contact_info=["robin.schirrmeister@uniklinik-freiburg.de"],
            ethics_approval=[
                "Approved by the ethical committee of the University of Freiburg"
            ],
            keywords=[
                "electroencephalography",
                "EEG analysis",
                "machine learning",
                "end-to-end learning",
                "brain-machine interface",
                "brain-computer interface",
                "model interpretability",
                "brain mapping",
            ],
            repository="GitHub",
            data_url="https://github.com/robintibor/braindecode/",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        tags=Tags(
            pathology=["Healthy"],
            modality=["EEG"],
            type=["Motor Imagery", "Motor Execution"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed signals with minimal preprocessing for end-to-end comparison",
            preprocessing_applied=True,
            filter_type="Butterworth",
            artifact_methods=["ICA"],
            re_reference="car",
            notes="Minimal preprocessing applied to conduct fair end-to-end comparison of ConvNets and FBCSP",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Deep ConvNet", "Shallow ConvNet", "ResNet", "FBCSP with LDA"],
            feature_extraction=[
                "FBCSP",
                "CSP",
                "Bandpower",
                "Spectral power modulations",
            ],
            frequency_bands={
                "alpha": [7.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 100.0],
            },
            spatial_filters=["CSP", "Common Spatial Patterns"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="holdout",
            evaluation_type=["within_subject"],
        ),
        performance={
            "accuracy_percent": 84.0,
            "FBCSP_accuracy": 82.1,
            "Deep_ConvNet_accuracy": 84.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation", "communication"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_execution",
            imagery_tasks=[
                "left_hand_finger_tapping",
                "right_hand_finger_tapping",
                "feet_toe_clenching",
                "rest",
            ],
            cue_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials={"total_per_subject": 963, "training_set": 880, "test_set": 160},
            n_trials_per_class={"per_class_per_subject": 260},
            n_blocks=13,
            trials_context="13 runs per subject, 80 trials per run (4 seconds each), 3-4 seconds inter-trial interval, pseudo-randomized presentation with all 4 classes shown every 4 trials",
        ),
        data_processed=True,
        file_format="EDF",
        abstract="Deep learning with convolutional neural networks (deep ConvNets) has revolutionized computer vision through end-to-end learning. This study investigates deep ConvNets for end-to-end EEG decoding of imagined or executed movements from raw EEG. Results show that recent advances including batch normalization and exponential linear units, together with a cropped training strategy, boosted decoding performance to match or exceed FBCSP (82.1% FBCSP vs 84.0% deep ConvNets). Novel visualization methods demonstrated that ConvNets learned to use spectral power modulations in alpha, beta, and high gamma frequencies with meaningful spatial distributions.",
        methodology="End-to-end deep learning approach comparing shallow ConvNets, deep ConvNets, and ResNets against FBCSP baseline. Evaluated design choices including batch normalization, exponential linear units, dropout, and cropped training strategies. Novel visualization techniques developed to understand learned features and verify that ConvNets use spectral power modulations in task-relevant frequency bands.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events=dict(right_hand=1, left_hand=2, rest=3, feet=4),
            code="Schirrmeister2017",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1002/hbm.23730",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        def _url(prefix):
            return "/".join([GIN_URL, prefix, "{:d}.edf".format(subject)])

        # Get the base path for the dataset
        base_path = dl.get_dataset_path("SCHIRRMEISTER2017", path)
        dataset_folder = os.path.join(base_path, "MNE-schirrmeister2017-data")

        # Create subfolder paths
        paths = []
        for t in ["train", "test"]:
            url = _url(t)
            # Extract subfolder name from URL
            subfolder = t

            # Download the file to a temporary location
            temp_path = dl.data_dl(url, "SCHIRRMEISTER2017", path, force_update, verbose)

            # Create the proper subfolder structure
            subfolder_path = os.path.join(dataset_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)

            # Move file to the correct subfolder
            filename = os.path.basename(temp_path)
            new_path = os.path.join(subfolder_path, filename)

            # If file already exists in target location, no need to move it
            if not os.path.exists(new_path):
                shutil.move(temp_path, new_path)

            paths.append(new_path)

        return paths

    def _get_single_subject_data(self, subject):
        train_raw, test_raw = [
            mne.io.read_raw_edf(path, infer_types=True, preload=True)
            for path in self.data_path(subject)
        ]

        # Select only EEG sensors (remove EOG, EMG),
        # and also set montage for visualizations
        montage = make_standard_montage("standard_1005")
        train_raw, test_raw = [
            raw.pick_types(eeg=True).set_montage(montage) for raw in (train_raw, test_raw)
        ]
        sessions = {
            "0": {"0train": train_raw, "1test": test_raw},
        }
        return sessions
