"""Alex Motor imagery dataset."""

from mne.io import Raw

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
            reference="earlobe",
            software="Matlab/Simulink",
            hardware="g.tec g.USBamp",
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
            study_design="Cue-based motor imagery paradigm (Step B of Brain Switch campaign) for familiarization and algorithm development",
            feedback_type="none",
            stimulus_type="visual cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            instructions="Cue-based paradigm without feedback. Subjects perform 20 imagined movements per class (right hand, feet, rest) following a visual cue, lasting 3 seconds each. Total duration approximately 10 minutes.",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.806022",
            investigators=["Alexandre Barachant"],
            institution="Université de Grenoble",
            country="France",
            license="CC-BY-SA-4.0",
            repository="Zenodo",
            data_url="https://zenodo.org/record/806023",
            publication_year=2012,
            senior_author="Alexandre Barachant",
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
            modality=["Motor"],
            type=["Research"],
        ),
        preprocessing=PreprocessingMetadata(
            re_reference="earlobe",
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
            applications=["motor_control"],
            environment="laboratory",
            online_feedback=False,
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
        abstract="Motor imagery dataset from the PhD thesis on robust control of an effector via asynchronous EEG brain-machine interface (Barachant, 2012). This shared dataset corresponds to Step B (cue-based imagery without feedback) of the Brain Switch campaign. Contains recordings from 8 subjects performing 3 motor imagery tasks (right hand, feet, rest) with 20 trials per class.",
        methodology="Cue-based paradigm without feedback (Step B of Brain Switch campaign). EEG recorded at 512 Hz with 16 active electrodes using a g.tec g.USBamp amplifier. Reference electrode placed on the ear. Subjects performed imagined movements following visual cues: right hand, feet, and rest, 20 trials per class, 3 seconds each. Recorded in standard office conditions (not shielded laboratory). Software: Matlab/Simulink with g.tec drivers.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events=dict(right_hand=2, feet=3, rest=4),
            code="AlexandreMotorImagery",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.5281/zenodo.806022",
            selected_subjects=subjects,
            selected_sessions=sessions,
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
