"""Munich MI dataset."""

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    ExperimentMetadata,
    FilterDetails,
    FrequencyBands,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids
from moabb.utils import depreciated_alias


DOWNLOAD_URL = "https://zenodo.org/record/1217449/files/"


@depreciated_alias("MunichMI", "1.1")
class GrosseWentrup2009(BaseDataset):
    """Munich Motor Imagery dataset.

    Motor imagery dataset from Grosse-Wentrup et al. 2009 [1]_.

    A trial started with the central display of a white fixation cross. After 3
    s, a white arrow was superimposed on the fixation cross, either pointing to
    the left or the right.
    Subjects were instructed to perform haptic motor imagery of the
    left or the right hand during display of the arrow, as indicated by the
    direction of the arrow. After another 7 s, the arrow was removed,
    indicating the end of the trial and start of the next trial. While subjects
    were explicitly instructed to perform haptic motor imagery with the
    specified hand, i.e., to imagine feeling instead of visualizing how their
    hands moved, the exact choice of which type of imaginary movement, i.e.,
    moving the fingers up and down, gripping an object, etc., was left
    unspecified.
    A total of 150 trials per condition were carried out by each subject,
    with trials presented in pseudorandomized order.

    Ten healthy subjects (S1–S10) participated in the experimental
    evaluation. Of these, two were females, eight were right handed, and their
    average age was 25.6 years with a standard deviation of 2.5 years. Subject
    S3 had already participated twice in a BCI experiment, while all other
    subjects were naive to BCIs. EEG was recorded at M=128 electrodes placed
    according to the extended 10–20 system. Data were recorded at 500 Hz with
    electrode Cz as reference. Four BrainAmp amplifiers were used for this
    purpose, using a temporal analog high-pass filter with a time constant of
    10 s. The data were re-referenced to common average reference
    offline. Electrode impedances were below 10 kΩ for all electrodes and
    subjects at the beginning of each recording session. No trials were
    rejected and no artifact correction was performed. For each subject, the
    locations of the 128 electrodes were measured in three dimensions using a
    Zebris ultrasound tracking system and stored for further offline analysis.

    References
    ----------
    .. [1] Grosse-Wentrup, Moritz, et al. "Beamforming in noninvasive
           brain–computer interfaces." IEEE Transactions on Biomedical
           Engineering 56.4 (2009): 1209-1219.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=128,
            channel_types={"eeg": 128},
            montage="10-20",
            hardware="BrainAmp",
            reference="Car",
            software="EEGLAB",
            filters={"highpass_time_constant_s": 10},
            impedance_threshold_kohm=10,
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF9",
                "AF7",
                "AF5",
                "AF3",
                "AF1",
                "AFz",
                "AF2",
                "AF4",
                "AF6",
                "AF8",
                "AF10",
                "F9",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "F10",
                "FT9",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "FT10",
                "T9",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "T10",
                "TP9",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "TP10",
                "P9",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "P10",
                "PO9",
                "PO7",
                "PO5",
                "PO3",
                "PO1",
                "POz",
                "PO2",
                "PO4",
                "PO6",
                "PO8",
                "PO10",
                "O1",
                "Oz",
                "O2",
                "O9",
                "O10",
                "I1",
                "Iz",
                "I2",
                "AFp9",
                "AFp7",
                "AFp5",
                "AFp3",
                "AFp1",
                "AFpz",
                "AFp2",
                "AFp4",
                "AFp6",
                "AFp8",
                "AFp10",
                "AFF9",
                "AFF7",
                "AFF5",
                "AFF3",
                "AFF1",
                "AFFz",
                "AFF2",
                "AFF4",
                "AFF6",
                "AFF8",
                "AFF10",
                "FFT9",
                "FFT7",
                "FFC5",
                "FFC3",
                "FFC1",
                "FFCz",
                "FFC2",
                "FFC4",
                "FFC6",
                "FFT8",
                "FFT10",
                "FTT9",
                "FTT7",
                "FCC5",
                "FCC3",
                "FCC1",
                "FCCz",
                "FCC2",
                "FCC4",
                "FCC6",
                "FTT8",
                "FTT10",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"male": 8, "female": 2},
            age_mean=25.6,
            handedness={"right": 8, "left": 2},
            bci_experience="experienced",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "left_hand", "feet"],
            trial_duration=10,
            feedback_type="none (offline); visual cursor control (online experiment)",
            stimulus_type="cursor_feedback",
            stimulus_modalities=["visual", "multisensory"],
            primary_modality="multisensory",
            synchronicity="synchronous",
            mode="both",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            filter_details=FilterDetails(
                filter_type="Butterworth",
            ),
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["SVM", "Logistic Regression"],
            feature_extraction=[
                "CSP",
                "Bandpower",
                "ERD",
                "ERS",
                "Covariance/Riemannian",
                "ICA",
            ],
            frequency_bands=FrequencyBands(
                analyzed_range=[1.0, 41.0],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_subject"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["smart_home", "vr_ar", "communication"],
            environment="shielded_room",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=150,
            trials_context="total",
        ),
        data_processed=False,
    )

    def __init__(self):
        self.events_id = dict(right_hand=2, left_hand=1)
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=self.events_id,
            code="GrosseWentrup2009",
            interval=[0, 7],
            paradigm="imagery",
            doi="10.1109/TBME.2008.2009768",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raw = mne.io.read_raw_eeglab(
            self.data_path(subject), preload=True, verbose="ERROR"
        )
        stim = raw.annotations.description.astype(np.dtype("<10U"))

        stim[stim == "20"] = "right_hand"
        stim[stim == "10"] = "left_hand"
        raw.annotations.description = stim
        return {"0": {"0": stim_channels_with_selected_ids(raw, self.event_id)}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # download .set
        _set = "{:s}subject{:d}.set".format(DOWNLOAD_URL, subject)
        set_local = dl.data_dl(_set, "MUNICHMI", path, force_update, verbose)
        # download .fdt
        _fdt = "{:s}subject{:d}.fdt".format(DOWNLOAD_URL, subject)
        dl.data_dl(_fdt, "MUNICHMI", path, force_update, verbose)
        return set_local
