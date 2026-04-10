"""59-Subject 40-Class SSVEP Dataset.

Dong and Tian (2023), Brain Science Advances.
DOI: 10.26599/BSA.2023.9050020
"""

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
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
from .utils import build_raw_from_epochs


ZENODO_URL = "https://zenodo.org/records/18847318/files/"

# fmt: off
# 40 frequencies (8.0-15.8 Hz, 0.2 Hz step) in 4x10 row-major order
_EVENTS = {
    "8": 1, "8.2": 2, "8.4": 3, "8.6": 4, "8.8": 5,
    "9": 6, "9.2": 7, "9.4": 8, "9.6": 9, "9.8": 10,
    "10": 11, "10.2": 12, "10.4": 13, "10.6": 14, "10.8": 15,
    "11": 16, "11.2": 17, "11.4": 18, "11.6": 19, "11.8": 20,
    "12": 21, "12.2": 22, "12.4": 23, "12.6": 24, "12.8": 25,
    "13": 26, "13.2": 27, "13.4": 28, "13.6": 29, "13.8": 30,
    "14": 31, "14.2": 32, "14.4": 33, "14.6": 34, "14.8": 35,
    "15": 36, "15.2": 37, "15.4": 38, "15.6": 39, "15.8": 40,
}
# fmt: on


class Dong2023(BaseDataset):
    """59-subject 40-class SSVEP dataset.

    Dataset from [1]_.

    This dataset contains 8-channel EEG recordings from 59 healthy adolescent
    volunteers (37 males, 22 females, aged 10-16, mean age 12.4) from the
    Suzhou Junior Competition of BCI Olympics 2022. Subjects performed a
    40-target SSVEP-BCI task using joint frequency and phase modulation (JFPM).

    Stimulation frequencies ranged from 8.0 to 15.8 Hz (0.2 Hz step) with a
    4x10 matrix layout. Each subject completed 4 blocks of 40 trials with
    1 s cue, 4 s stimulation, and 1 s feedback per trial.

    EEG was recorded at 1000 Hz with a NeuSenW system (Neuracle) using 8
    semi-dry (pre-gelled) electrodes around the occipital lobe, then
    downsampled to 250 Hz. Data is stored as 4D matrices of shape
    [8, 1250, 40, 4] (channels, time points, targets, blocks).

    Each trial epoch spans 5 s (0.5 s pre-stimulus + 4 s stimulation + 0.5 s
    post-stimulus) at 250 Hz = 1250 samples.

    References
    ----------
    .. [1] Y. Dong and S. Tian, "A large database towards user-friendly
       SSVEP-based BCI," Brain Science Advances, vol. 9, no. 4,
       pp. 297-309, 2023. DOI: 10.26599/BSA.2023.9050020
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=8,
            channel_types={"eeg": 8},
            montage="standard_1005",
            hardware="NeuSenW (Neuracle)",
            sensor_type="semi-dry (pre-gelled)",
            sensors=["POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2"],
            line_freq=50.0,
            reference="Fp1",
            ground="Fp2",
        ),
        participants=ParticipantMetadata(
            n_subjects=59,
            health_status="healthy",
            gender={"male": 37, "female": 22},
            age_mean=12.4,
            age_min=10,
            age_max=16,
            # fmt: off
            ages=[
                15,
                11,
                13,
                13,
                13,
                12,
                12,
                11,
                13,
                11,
                11,
                16,
                11,
                12,
                13,
                12,
                14,
                11,
                11,
                11,
                12,
                13,
                10,
                12,
                11,
                11,
                13,
                14,
                12,
                13,
                13,
                11,
                12,
                13,
                13,
                15,
                14,
                13,
                14,
                11,
                12,
                11,
                13,
                13,
                14,
                11,
                11,
                13,
                14,
                14,
                11,
                14,
                14,
                10,
                12,
                11,
                11,
                13,
                12,
            ],
            sexes=[
                "male",
                "female",
                "male",
                "female",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "male",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "male",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "female",
                "female",
                "male",
            ],
            handedness_list=["right"] * 59,
            # fmt: on
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            events=dict(_EVENTS),
            n_classes=40,
            trial_duration=4.0,
            stimulus_type="JFPM visual flicker",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            feedback_type="visual",
            task_type="SSVEP speller",
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi="10.26599/BSA.2023.9050020",
            investigators=["Yue Dong", "Sen Tian"],
            senior_author="Yue Dong",
            institution="Jiangsu JITRI Brain Machine Fusion Intelligence Institute",
            country="CN",
            repository="Zenodo",
            data_url="https://zenodo.org/records/18847318",
            license="CC BY-NC 4.0",
            publication_year=2023,
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched", downsampled_to_hz=250.0
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[8.0 + i * 0.2 for i in range(40)],
            frequency_resolution_hz=0.2,
        ),
        data_structure=DataStructureMetadata(n_blocks=4, n_trials=160),
        signal_processing=SignalProcessingMetadata(
            classifiers=["FBCCA", "eTRCA", "msTRCA"],
            feature_extraction=None,
            frequency_bands=None,
            spatial_filters=["CCA", "TRCA"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-block-out",
            cv_folds=4,
            evaluation_type=["within_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            environment="non-shielded", online_feedback=True
        ),
        tags=Tags(pathology=["healthy"], modality=["visual"], type=["perception"]),
        file_format="MAT",
    )

    _events = _EVENTS

    _ch_names = ["POz", "PO3", "PO4", "PO7", "PO8", "Oz", "O1", "O2"]

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 60)),
            sessions_per_subject=1,
            events=self._events,
            code="Dong2023",
            interval=[0.5, 4.5],
            paradigm="ssvep",
            doi="10.26599/BSA.2023.9050020",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        n_channels, n_blocks = 8, 4
        n_classes = len(self.event_id)

        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True)

        # .mat key "eegdata" with shape [8, 1250, 40, 4]
        # (channels, timepoints, targets, blocks)
        eeg = mat["eegdata"]
        data = np.transpose(eeg, axes=(2, 3, 0, 1))
        data = np.reshape(data, (-1, n_channels, eeg.shape[1]))

        event_ids = np.repeat(np.arange(1, n_classes + 1), n_blocks)
        raw = build_raw_from_epochs(data, self._ch_names, 250, event_ids, "standard_1005")
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        url = f"{ZENODO_URL}S{subject}.mat?download=1"
        return dl.data_dl(url, sign, path, force_update, verbose)
