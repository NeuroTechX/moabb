"""Munich MI dataset."""

import mne
import numpy as np

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
            montage="standard_1020",
            hardware="BrainAmp",
            reference="Cz",
            software=None,
            filters={"highpass_time_constant_s": 10},
            impedance_threshold_kohm=10,
            sensors=[
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
                "32",
                "33",
                "34",
                "35",
                "36",
                "37",
                "38",
                "39",
                "40",
                "41",
                "42",
                "43",
                "44",
                "45",
                "46",
                "47",
                "48",
                "49",
                "50",
                "51",
                "52",
                "53",
                "54",
                "55",
                "56",
                "57",
                "58",
                "59",
                "60",
                "61",
                "62",
                "63",
                "64",
                "65",
                "66",
                "67",
                "68",
                "69",
                "70",
                "71",
                "72",
                "73",
                "74",
                "75",
                "76",
                "77",
                "78",
                "79",
                "80",
                "81",
                "82",
                "83",
                "84",
                "85",
                "86",
                "87",
                "88",
                "89",
                "90",
                "91",
                "92",
                "93",
                "94",
                "95",
                "96",
                "97",
                "98",
                "99",
                "100",
                "101",
                "102",
                "103",
                "104",
                "105",
                "106",
                "107",
                "108",
                "109",
                "110",
                "111",
                "112",
                "113",
                "114",
                "115",
                "116",
                "117",
                "118",
                "119",
                "120",
                "121",
                "122",
                "123",
                "124",
                "125",
                "126",
                "127",
                "128",
            ],
            line_freq=50.0,
            sensor_type=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"male": 8, "female": 2},
            age_mean=25.6,
            age_std=2.5,
            handedness={"right": 8},
            bci_experience="mixed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "left_hand"],
            trial_duration=10,
            feedback_type="none",
            stimulus_type="arrow_cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            tasks=["motor_imagery"],
            task_type="motor_imagery",
            instructions="Subjects were instructed to perform haptic motor imagery of the left or the right hand during display of the arrow, as indicated by the direction of the arrow",
            events={"left_hand": 1, "right_hand": 2},
            study_design="two-class motor imagery with arrow cues",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TBME.2008.2009768",
            investigators=[
                "Moritz Grosse-Wentrup",
                "Christian Liefhold",
                "Klaus Gramann",
                "Martin Buss",
            ],
            institution="Technische Universität München",
            country="DE",
            publication_year=2009,
            senior_author="Martin Buss",
            contact_info=["moritzgw@ieee.org"],
            institution_department="Institute of Automatic Control Engineering (LSR)",
            keywords=[
                "Beamforming",
                "brain-computer interfaces",
                "common spatial patterns",
                "electroencephalography",
                "motor imagery",
                "spatial filtering",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            artifact_methods=["none"],
            re_reference="car",
            notes="No trials were rejected and no artifact correction was performed. Data were re-referenced to common average reference offline.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Logistic Regression"],
            feature_extraction=[
                "CSP",
                "Beamforming",
                "Laplacian",
                "Bandpower",
            ],
            frequency_bands={
                "analyzed_range": [7.0, 30.0],
            },
            spatial_filters=["CSP", "Beamforming", "Laplacian"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject"],
            cv_method="bootstrapping",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"],
            environment="shielded_room",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=7.0,
            imagery_duration_s=7.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=150,
            trials_context="per_class",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        data_processed=True,
        file_format="set",
    )

    def __init__(self, subjects=None, sessions=None):
        self.events_id = dict(right_hand=2, left_hand=1)
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=self.events_id,
            code="GrosseWentrup2009",
            interval=[0, 7],
            paradigm="imagery",
            doi="10.1109/TBME.2008.2009768",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
