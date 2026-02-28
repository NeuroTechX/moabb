"""Simple and compound motor imagery.

https://doi.org/10.1371/journal.pone.0114853
"""

import json
import logging
from pathlib import Path
from zipfile import ZipFile

import requests
from mne.utils import _open_lock

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

from .base import BaseBIDSDataset
from .bids_interface import get_bids_root
from .download import download_if_missing, get_dataset_path, get_user_agent


log = logging.getLogger(__name__)

ZENODO_RECORD_ID = 16534752
# Zenodo API endpoint for published records
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


class Zhou2016(BaseBIDSDataset):
    """Motor Imagery dataset from Zhou et al 2016.

    Dataset from the article *A Fully Automated Trial Selection Method for
    Optimization of Motor Imagery Based Brain-Computer Interface* [1]_.
    This dataset contains data recorded on 4 subjects performing 3 type of
    motor imagery: left hand, right hand and feet.

    Every subject went through three sessions, each of which contained two
    consecutive runs with several minutes inter-run breaks, and each run
    comprised 75 trials (25 trials per class). The intervals between two
    sessions varied from several days to several months.

    A trial started by a short beep indicating 1 s preparation time,
    and followed by a red arrow pointing randomly to three directions (left,
    right, or bottom) lasting for 5 s and then presented a black screen for
    4 s. The subject was instructed to immediately perform the imagination
    tasks of the left hand, right hand or foot movement respectively according
    to the cue direction, and try to relax during the black screen.

    References
    ----------

    .. [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated
           Trial Selection Method for Optimization of Motor Imagery Based
           Brain-Computer Interface. PLoS ONE 11(9).
           https://doi.org/10.1371/journal.pone.0162657
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=14,
            channel_types={"eeg": 14},
            montage="10-20",
            sensor_type="EEG",
            hardware="BCI2000",
            reference="left mastoid",
            ground="right mastoid",
            software="BCI2000",
            filters="0.1-100 Hz bandpass, 50 Hz notch",
            sensors=[
                "Fp1",
                "Fp2",
                "FC3",
                "FCz",
                "FC4",
                "C3",
                "Cz",
                "C4",
                "CP3",
                "CPz",
                "CP4",
                "O1",
                "Oz",
                "O2",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["horizontal"],
                has_emg=True,
                other_physiological=["ecg", "gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=4,
            health_status="healthy",
            gender={"male": 1, "female": 3},
            age_min=22,
            age_max=28,
            bci_experience="prior experience in the experimental paradigm",
        ),
        experiment=ExperimentMetadata(
            events={"left_hand": 1, "right_hand": 2, "feet": 3},
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "left_hand", "feet"],
            trial_duration=10.0,
            study_design="Three-class motor imagery (left hand, right hand, foot movement imagination) according to cue direction",
            feedback_type="visual cue (red arrow)",
            stimulus_type="visual arrow and beep",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            mode="online",
            instructions="Subject sat in comfortable armchair facing computer screen. Trial started with short beep (1s preparation), followed by red arrow pointing randomly to three directions (left, right, or bottom) lasting 5s, then black screen for 4s. Subject instructed to immediately perform imagination tasks of left hand, right hand or foot movement according to cue direction, and relax during black screen.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0162657",
            investigators=[
                "Bangyan Zhou",
                "Xiaopei Wu",
                "Zhao Lv",
                "Lei Zhang",
                "Xiaojin Guo",
            ],
            institution="Anhui University",
            institution_department="School of Computer Science and Technology",
            country="China",
            institution_address="Hefei, China",
            data_url="https://doi.org/10.6084/m9.figshare.2061654",
            publication_year=2016,
            senior_author="Xiaopei Wu",
            contact_info=["wxp2001@ahu.edu.cn"],
            funding=[
                "National Natural Science Foundation of China (61271352; 61401002)",
                "Anhui Province Natural Science Foundation (1408085QF125)",
                "Anhui University Center of Information Support & Assurance Technology Open Foundation (ADXXBZ2014-3)",
            ],
            ethics_approval=["Institutional Review Board at Anhui University"],
            keywords=[
                "motor imagery",
                "brain-computer interface",
                "independent component analysis",
                "trial selection",
                "artifact rejection",
                "ICA optimization",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=3,
        runs_per_session=2,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG available",
            preprocessing_applied=True,
            preprocessing_steps=["bandpass filtering", "trial rejection", "ICA"],
            highpass_hz=8,
            lowpass_hz=30,
            bandpass={"low_cutoff_hz": 0.1, "high_cutoff_hz": 100.0},
            notch_hz=[50],
            filter_type="zero-phase FIR",
            artifact_methods=["trial rejection", "ICA"],
            re_reference="left mastoid",
            epoch_window=[0.5, 5.0],
            notes="Two different electrode-distributions were defined: eight-channel scheme (FP1, FP2, C3, Cz, C4, O1, Oz, O2) and nine-channel scheme (FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4). The one with higher classification accuracy was chosen for each subject.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "zero-training classifier"],
            feature_extraction=[
                "CSP",
                "Bandpower",
                "ERD",
                "ERS",
                "Covariance/Riemannian",
                "Time-Frequency",
                "ICA",
            ],
            frequency_bands={
                "analyzed_range": [8.0, 30.0],
                "mu": [10.0, 14.0],
                "beta": [12.0, 16.0],
            },
            spatial_filters=["ICA", "CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session", "within_session", "cross_run"],
        ),
        performance={
            "accuracy_percent": 80.6,
            "ICA-T_self_test_mean": 80.6,
            "ICA-T_session_transfer_success_rate": 67.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["vr_ar", "communication"],
            environment="indoor laboratory",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand", "left_hand", "feet"],
            cue_duration_s=5.0,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=75,
            trials_context="per_run",
            n_trials_per_class={"right_hand": 25, "left_hand": 25, "feet": 25},
        ),
        data_processed=True,
    )

    def __init__(self, subjects=None, sessions=None):
        """Initialize the BIDS dataset."""
        super().__init__(
            subjects=list(range(1, 5)),
            sessions_per_subject=3,
            events=dict(left_hand=1, right_hand=2, feet=3),
            code="Zhou2016",
            # MI 1-6s, prepare 0-1, break 6-10
            # boundary effects
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1371/journal.pone.0162657",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self.zenodo_record_id = ZENODO_RECORD_ID

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download the subject data."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path = Path(get_dataset_path(self.code, path))
        dataset_path = get_bids_root(code=self.code, path=path)

        if not dataset_path.exists():
            log.info(f"Creating dataset path: {dataset_path}")
            dataset_path.mkdir(parents=True, exist_ok=True)

        metainfo = self.get_metainfo(path=dataset_path)

        for file in metainfo["files"]:
            file_name = file["key"]
            file_url = file["links"]["self"]

            file_path = dataset_path / file_name
            if "sub" in file_name:
                # Check if the file corresponds to the current subject
                if file_name == f"sub-{subject}.zip":
                    folder_path = file_path.with_suffix("")

                    if not folder_path.exists():
                        log.info(
                            f"Downloading {file_name} for subject {subject} to {file_path}"
                        )
                        download_if_missing(
                            file_path=file_path,
                            url=file_url,
                            warn_missing=False,
                            verbose=verbose,
                        )

                        log.info(f"Extracting {file_name} to {folder_path}")
                        with ZipFile(str(file_path), "r") as zip_ref:
                            zip_ref.extractall(folder_path.parent)

            else:
                download_if_missing(
                    file_path=file_path, url=file_url, warn_missing=False, verbose=verbose
                )

        return dataset_path

    def get_metainfo(self, path=None):
        """Fetch a Zenodo record by its ID."""
        # first thing try to get the record from the path if already downloaded

        file_path = f"{path}/{self.zenodo_record_id}.json"

        if not Path(file_path).exists():
            # If not found, fetch from Zenodo
            response = requests.get(ZENODO_URL, headers={"User-Agent": get_user_agent()})
            response.raise_for_status()
            # Save the response to a file
            with _open_lock(file_path, "w") as f:
                json.dump(response.json(), f, indent=4)
            return response.json()
        else:
            with _open_lock(file_path, "r") as f:
                return json.load(f)
