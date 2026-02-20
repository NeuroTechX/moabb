"""Alex Motor imagery dataset."""

from mne.io import Raw

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FrequencyBands,
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
            sampling_rate=250.0,
            n_channels=22,
            channel_types={"eeg": 22},
            reference="Car",
            software="OpenViBE.",
            sensors=[
                "Fz",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "P1",
                "Pz",
                "P2",
                "POz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["gsr", "ppg"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=8,
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=1,
            class_labels=["rest"],
            study_design="Brain-switch based on motor imagery for asynchronous BCI control of an effector",
            feedback_type="visual (primarily), auditory, haptic (rare cases)",
            stimulus_type="avatar",
            stimulus_modalities=["visual"],
            primary_modality="visual",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.806022",
            license="CC BY 4.0",
        ),
        tags=Tags(
            pathology=["Other"],
            modality=["Visual"],
            type=["Clinical/Intervention"],
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
                "xDAWN",
                "Riemannian",
                "kNN",
                "Naive Bayes",
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
                "xDAWN",
            ],
            frequency_bands=FrequencyBands(
                alpha=[8.0, 12.0],
                mu=[8, 12],
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="30-fold",
            cv_folds=30,
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["vr_ar", "communication"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
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
