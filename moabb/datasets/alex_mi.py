"""Alex Motor imagery dataset."""

from mne.io import Raw

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


ALEX_URL = "https://zenodo.org/record/806023/files/"


class AlexMI(BaseDataset):
    """Alex Motor Imagery dataset.

    Motor imagery dataset from the PhD dissertation of A. Barachant [1]_.

    This Dataset contains EEG recordings from 8 subjects, performing 2 task of
    motor imagination (right hand, feet or rest). Data have been recorded at
    512Hz with 16 wet electrodes (Fpz, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
    P7, P3, Pz, P4, P8) with a g.tec g.USBamp EEG amplifier.

    File are provided in MNE raw file format. A stimulation channel encoding
    the timing of the motor imagination. The start of a trial is encoded as 1,
    then the actual start of the motor imagination is encoded with 2 for
    imagination of a right hand movement, 3 for imagination of both feet
    movement and 4 with a rest trial.

    The duration of each trial is 3 second. There is 20 trial of each class.

    references
    ----------

    .. [1] Barachant, A., 2012. Commande robuste d'un effecteur par une
           interface cerveau machine EEG asynchrone (Doctoral dissertation,
           Université de Grenoble).
           https://tel.archives-ouvertes.fr/tel-01196752
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=16,
            channel_types={"eeg": 16},
            reference="Car",
            software="OpenViBE",
            sensors=[
                "Fpz",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
            ],
            line_freq=50.0,
            sensor_type="EEG",
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr", "ppg"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
            health_status="healthy",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"right_hand": 2, "feet": 3, "rest": 4},
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "feet", "rest"],
            trial_duration=3.0,
            study_design="Brain-switch based on motor imagery for asynchronous BCI control of an effector",
            feedback_type="visual (primarily), auditory, haptic (rare cases)",
            stimulus_type="avatar",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="asynchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.806022",
            investigators=["Alexandre Barachant"],
            institution="Université de Grenoble",
            country="France",
            license="CC-BY-SA-4.0",
            repository="Zenodo",
            data_url="https://zenodo.org/record/806023/files/",
            publication_year=2012,
            senior_author="Christian Jutten",
            contact_info=["alexandre.barachant@gmail.com"],
            institution_department="Laboratoire Électronique et système pour la santé CEA-LETI",
            institution_address="CEA-LETI Grenoble, France",
            associated_paper_doi="tel-01196752v1",
            keywords=[
                "brain-computer interface",
                "motor imagery",
                "EEG",
                "Riemannian geometry",
                "asynchronous BCI",
                "brain-switch",
                "covariance matrices",
                "Common Spatial Pattern",
            ],
            description="Motor imagery dataset from the PhD dissertation of A. Barachant. Contains EEG recordings from 8 subjects performing motor imagination tasks (right hand, feet, or rest). Used to validate robust control of an effector via asynchronous EEG-based brain-machine interface.",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        data_processed=True,
        file_format="fif",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Research"],
        ),
        preprocessing=PreprocessingMetadata(
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "LDA",
                "SVM",
                "MDM",
                "Riemannian",
                "kNN",
                "Naive Bayes",
                "Logistic Regression",
            ],
            feature_extraction=[
                "CSP",
                "FBCSP",
                "ERD",
                "ERS",
                "PSD",
                "Covariance/Riemannian",
                "AR",
                "ICA",
            ],
            frequency_bands={
                "alpha": [8.0, 12.0],
                "mu": [8.0, 12.0],
            },
            spatial_filters=["CSP", "Geodesic filtering"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross-validation",
            evaluation_type=["within_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["vr_ar", "communication", "motor_control"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand", "feet", "rest"],
            cue_duration_s=1.0,
            imagery_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=60,
            n_trials_per_class={"right_hand": 20, "feet": 20, "rest": 20},
            trials_context="20 trials per class, 3 second duration each",
        ),
        abstract="This thesis addresses the robust control of an effector through an asynchronous EEG-based brain-machine interface. The work focuses on motor imagery paradigms and introduces novel signal processing approaches based on Riemannian geometry for covariance matrices. The dataset contains recordings from 8 subjects performing motor imagery tasks (right hand, feet, rest) with the goal of controlling an avatar in a virtual environment.",
        methodology="Subjects performed motor imagery tasks in an asynchronous brain-switch paradigm. EEG was recorded at 250 Hz with 22 electrodes. Signal processing employed Riemannian geometry approaches including Minimum Distance to Mean (MDM) classification, geodesic filtering, and tangent space mapping. The research validated adaptive learning and effector coupling through two experimental campaigns.",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events=dict(right_hand=2, feet=3, rest=4),
            code="AlexandreMotorImagery",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.5281/zenodo.806022",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raw = Raw(self.data_path(subject), preload=True, verbose="ERROR")
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        url = "{:s}subject{:d}.raw.fif".format(ALEX_URL, subject)
        return dl.data_dl(url, "ALEXEEG", path, force_update, verbose)
