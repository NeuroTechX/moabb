"""GigaDb Motor imagery dataset."""

import logging

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
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

from . import download as dl
from .base import BaseDataset


log = logging.getLogger(__name__)
GIGA_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/"


class Cho2017(BaseDataset):
    """Motor Imagery dataset from Cho et al 2017.

    Dataset from the paper [1]_.

    **Dataset Description**

    We conducted a BCI experiment for motor imagery movement (MI movement)
    of the left and right hands with 52 subjects (19 females, mean age ± SD
    age = 24.8 ± 3.86 years); Each subject took part in the same experiment,
    and subject ID was denoted and indexed as s1, s2, …, s52.
    Subjects s20 and s33 were both-handed, and the other 50 subjects
    were right-handed.

    EEG data were collected using 64 Ag/AgCl active electrodes.
    A 64-channel montage based on the international 10-10 system was used to
    record the EEG signals with 512 Hz sampling rates.
    The EEG device used in this experiment was the Biosemi ActiveTwo system.
    The BCI2000 system 3.0.2 was used to collect EEG data and present
    instructions (left hand or right hand MI). Furthermore, we recorded
    EMG as well as EEG simultaneously with the same system and sampling rate
    to check actual hand movements. Two EMG electrodes were attached to the
    flexor digitorum profundus and extensor digitorum on each arm.

    Subjects were asked to imagine the hand movement depending on the
    instruction given. Five or six runs were performed during the MI
    experiment. After each run, we calculated the classification
    accuracy over one run and gave the subject feedback to increase motivation.
    Between each run, a maximum 4-minute break was given depending on
    the subject's demands.

    References
    ----------

    .. [1] Cho, H., Ahn, M., Ahn, S., Kwon, M. and Jun, S.C., 2017.
           EEG datasets for motor imagery brain computer interface.
           GigaScience. https://doi.org/10.1093/gigascience/gix034
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=64,
            channel_types={"eeg": 64, "emg": 4},
            montage="10-10",
            hardware="Biosemi ActiveTwo",
            sensor_type="active electrodes",
            electrode_type="Ag/AgCl active electrodes",
            reference="Car",
            software="BCI2000 3.0.2",
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "AFz",
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
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
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
                "Fpz",
                "Fz",
                "Iz",
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
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True,
                emg_channels=4,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=52,
            health_status="healthy",
            gender={"female": 19, "male": 33},
            age_mean=24.8,
            age_std=3.86,
            handedness={"right": 50, "both": 2},
            bci_experience="collected via questionnaire (0 = no, number = how many times)",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "left_hand"],
            trial_duration=3.0,
            study_design="motor imagery",
            stimulus_type="visual instruction",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            feedback_type="classification accuracy feedback after each run",
            instructions="Subjects were asked to imagine kinesthetic finger movements (touching index, middle, ring, and little finger to thumb within 3 seconds)",
            events={"left_hand": 1, "right_hand": 1},
        ),
        documentation=DocumentationMetadata(
            doi="10.5524/100295",
            description="EEG datasets for motor imagery brain-computer interface from 52 subjects with psychological and physiological questionnaire, EMG datasets, 3D EEG electrode locations, and non-task-related states",
            investigators=[
                "Hohyun Cho",
                "Minkyu Ahn",
                "Sangtae Ahn",
                "Moonyoung Kwon",
                "Sung Chan Jun",
            ],
            senior_author="Sung Chan Jun",
            institution="Gwangju Institute of Science and Technology",
            institution_address="123 Cheomdangwagi-ro, Buk-gu, Gwangju 61005, Korea",
            institution_department="School of Electrical Engineering and Computer Science",
            country="Korea",
            repository="GigaDB",
            data_url="http://dx.doi.org/10.5524/100295",
            license="CC-BY-4.0",
            publication_year=2017,
            contact_info=[
                "scjun@gist.ac.kr",
                "TEL: +82-62-715-2216",
                "FAX: +82-62-715-2204",
            ],
            funding=[
                "GIST Research Institute (GRI) grant funded by the GIST in 2017",
                "Institute for Information & Communication Technology Promotion (IITP) grant funded by the Korea government (No. 2017-0-00451)",
            ],
            ethics_approval=[
                "Institutional Review Board of Gwangju Institute of Science and Technology"
            ],
            keywords=[
                "motor imagery",
                "EEG",
                "brain-computer interface",
                "performance variation",
                "subject-to-subject transfer",
            ],
            associated_paper_doi="10.1093/gigascience/gix034",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw with bad trial indices provided",
            preprocessing_applied=True,
            preprocessing_steps=[
                "high-pass filtering above 0.5 Hz",
                "common average reference",
                "band-pass filtering (8-30 Hz for analysis, 8-14 Hz for ERD/ERS)",
                "Laplacian filtering (for ERD/ERS)",
                "Hilbert transform",
                "bad trial rejection (amplitude > ±100 μV)",
                "EMG correlation detection",
            ],
            highpass_hz=0.5,
            bandpass="8-30 Hz (SMR analysis), 8-14 Hz (mu rhythm ERD/ERS), 50-250 Hz (EMG)",
            filter_type="Butterworth",
            filter_order=4,
            artifact_methods=["EMG removal", "voltage threshold rejection"],
            re_reference="Car",
            epoch_window=[0.5, 2.5],
            notes="Bad trials detected by amplitude threshold and EMG correlation; baseline correction using -500 to 0 msec for ERD/ERS",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["FLDA"],
            feature_extraction=["CSP", "ERD", "ERS"],
            frequency_bands={
                "alpha": [8.0, 14.0],
                "mu": [8, 12],
                "analyzed_range": [8.0, 30.0],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="random subset selection",
            cv_folds=10,
            evaluation_type=["within_session"],
        ),
        performance={
            "accuracy_percent": 67.46,
            "accuracy_std": 13.17,
            "discriminative_subjects": 38,
            "total_subjects": 52,
        },
        bci_application=BCIApplicationMetadata(
            applications=[
                "subject-to-subject transfer",
                "performance variation investigation",
            ],
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            imagery_duration_s=3.0,
            cue_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials="100 or 120 per class (200-240 total)",
            trials_context="per_class",
            n_blocks=5,
        ),
        data_processed=True,
        file_format=".mat (MATLAB)",
        abstract="Motor imagery (MI)-based brain-computer interface (BCI) dataset from 52 subjects with EEG, EMG, psychological and physiological questionnaire, 3D EEG electrode locations, and non-task-related states. The dataset includes 100 or 120 trials per class (left/right hand) with validation showing 73.08% (38 subjects) had discriminative information. Mean accuracy of 67.46% (±13.17%) over 50 subjects (excluding 2 bad subjects). Dataset stored in GigaDB and validated using bad trial percentage, ERD/ERS analysis, and classification analysis.",
        methodology="Subjects performed motor imagery of left and right hand finger movements (kinesthetic imagery). Each trial consisted of: 2 seconds fixation cross, 3 seconds instruction (left/right hand), followed by random 4.1-4.8 second break. Five or six runs performed with feedback after each run. Additional data collected: 6 types of non-task-related data (eye blinking, eyeball movements, head movement, jaw clenching, resting state) and 20 trials of real hand movement per class. 3D electrode coordinates measured with Polhemus Fastrak digitizer. Experiments conducted August-September 2011 in four time slots (9:30-12:00, 12:30-15:00, 15:30-18:00, 19:00-21:30) with background noise 37-39 dB.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 53)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            code="Cho2017",
            interval=[0, 3],  # full trial is 0-3s, but edge effects
            paradigm="imagery",
            doi="10.5524/100295",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        fname = self.data_path(subject)

        data = loadmat(
            fname,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )["eeg"]

        # fmt: off
        eeg_ch_names = [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ]
        # fmt: on
        emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
        ch_names = eeg_ch_names + emg_ch_names + ["Stim"]
        ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]
        montage = make_standard_montage("standard_1005")
        imagery_left = data.imagery_left - data.imagery_left.mean(axis=1, keepdims=True)
        imagery_right = data.imagery_right - data.imagery_right.mean(
            axis=1, keepdims=True
        )

        eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
        eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])

        # trials are already non continuous. edge artifact can appears but
        # are likely to be present during rest / inter-trial activity
        eeg_data = np.hstack(
            [eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r]
        )
        log.warning(
            "Trials demeaned and stacked with zero buffer to create "
            "continuous data -- edge effects present"
        )

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=data.srate)
        raw = RawArray(data=eeg_data, info=info, verbose=False)
        raw.set_montage(montage)

        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = "{:s}s{:02d}.mat".format(GIGA_URL, subject)
        return dl.data_dl(url, "GIGADB", path, force_update, verbose)
