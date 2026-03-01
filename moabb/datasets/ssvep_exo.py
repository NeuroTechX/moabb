"""SSVEP Exoskeleton dataset."""

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
from moabb.utils import depreciated_alias

from . import download as dl
from .base import BaseDataset


SSVEPEXO_URL = "https://zenodo.org/record/2392979/files/"


@depreciated_alias("SSVEPExo", "1.1")
class Kalunga2016(BaseDataset):
    """SSVEP Exo dataset.

    SSVEP dataset from E. Kalunga PhD in University of Versailles [1]_.

    The datasets contains recording from 12 male and female subjects aged
    between 20 and 28 years. Informed consent was obtained from all subjects,
    each one has signed a form attesting her or his consent. The subject sits
    in an electric wheelchair, his right upper limb is resting on the
    exoskeleton. The exoskeleton is functional but is not used during the
    recording of this experiment.

    A panel of size 20x30 cm is attached on the left side of the chair, with
    3 groups of 4 LEDs blinking at different frequencies. Even if the panel
    is on the left side, the user could see it without moving its head. The
    subjects were asked to sit comfortably in the wheelchair and to follow the
    auditory instructions, they could move and blink freely.

    A sequence of trials is proposed to the user. A trial begin by an audio cue
    indicating which LED to focus on, or to focus on a fixation point set at an
    equal distance from all LEDs for the reject class. A trial lasts 5 seconds
    and there is a 3 second pause between each trial. The evaluation is
    conducted during a session consisting of 32 trials, with 8 trials for each
    frequency (13Hz, 17Hz and 21Hz) and 8 trials for the reject class, i.e.
    when the subject is not focusing on any specific blinking LED.

    There is between 2 and 5 sessions for each user, recorded on different
    days, by the same operators, on the same hardware and in the same
    conditions.

    Notes
    -----
    The events notation 17Hz and 21Hz were swapped after an investigation conducted
    by ponpopon at Github.

    The dataset includes recordings from 12 healthy subjects.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] Emmanuel K. Kalunga, Sylvain Chevallier, Quentin Barthelemy. "Online
           SSVEP-based BCI using Riemannian Geometry". Neurocomputing, 2016.
           arXiv report: https://arxiv.org/abs/1501.03227

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=8,
            channel_types={"eeg": 8},
            montage="standard_1005",
            sensor_type="EEG",
            hardware="g.tec MobiLab",
            reference="right mastoid",
            sensors=["Oz", "O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"],
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"13": 2, "17": 4, "21": 3, "rest": 1},
            paradigm="ssvep",
            n_classes=4,
            class_labels=["13hz", "17hz", "21hz", "rest"],
            trial_duration=6.0,
            study_design="SSVEP",
            stimulus_type="flickering",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            feedback_type="none",
            stimulus_presentation={
                "device": "LED stimuli",
                "frequencies": "13 Hz, 17 Hz, 21 Hz",
                "note": "No phase synchronization required",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neucom.2016.01.007",
            repository="Zenodo",
            data_url="https://zenodo.org/record/2392979",
            publication_year=2016,
            investigators=[
                "Emmanuel K. Kalunga",
                "Sylvain Chevallier",
                "Quentin Barthélemy",
                "Karim Djouani",
                "Eric Monacelli",
                "Yskandar Hamam",
            ],
            institution="Universite de Versailles Saint-Quentin",
            institution_address="78140 Velizy, France",
            institution_department="Laboratoire d'Ingénierie des Systèmes de Versailles",
            country="FR",
            senior_author="Sylvain Chevallier",
            keywords=[
                "Riemannian geometry",
                "Online",
                "Asynchronous",
                "Brain-Computer Interfaces",
                "Steady State Visually Evoked Potentials",
            ],
            description="Online SSVEP-based BCI using Riemannian geometry for assistive robotics with shared control scheme",
            license="CC-BY-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            preprocessing_applied=False,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["MDRM", "CCA"],
            feature_extraction=["Covariance/Riemannian"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="bootstrap",
            evaluation_type=["cross_subject", "cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["assistive_robotics"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            n_targets=3,
            stimulus_frequencies_hz=[13.0, 17.0, 21.0],
        ),
        data_structure=DataStructureMetadata(
            n_trials="32 trials per session (8 per visual stimulus, 8 for resting class)",
            trials_context="per session",
        ),
        sessions_per_subject=1,
        runs_per_session=None,
        data_processed=False,
        file_format="fif",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events={"13": 2, "17": 4, "21": 3, "rest": 1},
            code="Kalunga2016",
            interval=[2, 4],
            paradigm="ssvep",
            doi="10.1016/j.neucom.2016.01.007",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""

        out = {}
        paths = self.data_path(subject, update_path=True, verbose=False)
        for ii, path in enumerate(paths):
            raw = Raw(path, preload=True, verbose="ERROR")
            out[str(ii)] = raw
        return {"0": out}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        runs = {s + 1: n for s, n in enumerate([2] * 6 + [3] + [2] * 2 + [4, 2, 5])}

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        paths = []
        for run in range(runs[subject]):
            url = "{:s}subject{:02d}_run{:d}_raw.fif".format(
                SSVEPEXO_URL, subject, run + 1
            )
            p = dl.data_dl(url, "SSVEPEXO", path, force_update, verbose)
            paths.append(p)
        return paths
