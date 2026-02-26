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
    FilterDetails,
    FrequencyBands,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PerformanceMetadata,
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
            channel_types={"eeg": 64},
            montage="10-10",
            hardware="Biosemi ActiveTwo",
            sensor_type="active electrodes",
            reference="Car",
            software="BCI2000",
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
                has_emg=True,
                emg_channels=4,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=52,
            health_status="healthy",
            gender={"female": 19, "male": 33},
            bci_experience="collected via questionnaire (0 = no, number = how many times)",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "left_hand"],
            trial_duration=25.0,
            study_design="motor imagery",
            stimulus_type="cursor_feedback",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.5524/100295",
            license="Unknown",
            associated_paper_doi="10.1093/gigascience/gix034",
            funding=["grant funded funded", "grant\nfunded funded"],
            repository="GigaDB",
        ),
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
            filter_details=FilterDetails(
                highpass_hz=0.5,
                bandpass="8-30 Hz (SMR analysis), 8-14 Hz (mu rhythm ERD/ERS), 50-250 Hz (EMG)",
                filter_type="Butterworth",
                filter_order=4,
            ),
            artifact_methods=["EMG removal", "ICA"],
            re_reference="Car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["CSP", "ERD", "ERS"],
            frequency_bands=FrequencyBands(
                alpha=[8.0, 14.0],
                mu=[8, 12],
                analyzed_range=[8.0, 30.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_subject"],
        ),
        performance=PerformanceMetadata(
            accuracy_percent=60.42,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["smart_home", "vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=10,
            trials_context="per_class",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 53)),
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            code="Cho2017",
            interval=[0, 3],  # full trial is 0-3s, but edge effects
            paradigm="imagery",
            doi="10.5524/100295",
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
