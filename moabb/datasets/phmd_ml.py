import os

import mne
import numpy as np
from scipy.io import loadmat

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)

from . import download as dl
from .base import BaseDataset


HEADMOUNTED_URL = "https://zenodo.org/record/2617085/files/"


class Cattan2019_PHMD(BaseDataset):
    """Passive Head Mounted Display with Music Listening dataset [1]_.

    We describe the experimental procedures for a dataset that we have made publicly available
    at https://doi.org/10.5281/zenodo.2617084 in mat (Mathworks, Natick, USA) and csv formats.
    This dataset contains electroencephalographic recordings of 12 subjects listening to music
    with and without a passive head-mounted display, that is, a head-mounted display which does
    not include any electronics at the exception of a smartphone. The electroencephalographic
    headset consisted of 16 electrodes. Data were recorded during a pilot experiment taking
    place in the GIPSA-lab, Grenoble, France, in 2017 (Cattan and al, 2018).
    The ID of this dataset is PHMDML.EEG.2017-GIPSA.

    **full description of the experiment**
    https://hal.archives-ouvertes.fr/hal-02085118

    **Link to the data**
    https://doi.org/10.5281/zenodo.2617084

    **Authors**
    Principal Investigator: Eng. Grégoire Cattan
    Technical Supervisors: Eng. Pedro L. C. Rodrigues
    Scientific Supervisor: Dr. Marco Congedo

    **ID of the dataset**
    PHMDML.EEG.2017-GIPSA

    Notes
    -----

    .. versionadded:: 1.0.0

    References
    ----------

    .. [1] G. Cattan, P. L. Coelho Rodrigues, and M. Congedo,
        ‘Passive Head-Mounted Display Music-Listening EEG dataset’,
        Gipsa-Lab ; IHMTEK, Research Report 2, Mar. 2019. doi: 10.5281/zenodo.2617084.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="standard_1020",
            hardware="g.USBamp",
            sensor_type="wet",
            reference="right earlobe",
            ground="AFz",
            software="OpenViBE",
            filters="no digital filter",
            sensors=[
                "Cz",
                "Fc5",
                "Fc6",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P3",
                "P4",
                "P7",
                "P8",
                "Pz",
                "T7",
                "T8",
            ],
            line_freq=50.0,
            cap_manufacturer="EasyCap",
            cap_model="EC20",
            electrode_type="wet",
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            gender={"male": 9, "female": 3},
            age_mean=26.25,
            age_std=2.63,
            health_status="healthy",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="rstate",
            trial_duration=60.0,
            study_design="focus on the marker and to listen to the music that was diffused during the experiment (Bach Invention from one to ten on harpsichord).",
            stimulus_type="visual fixation marker",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="auditory",
            mode=None,
            instructions="Subjects were asked to focus on the marker and to listen to the music that was diffused during the experiment",
            events={"switched-off": 1, "switched-on": 2},
            n_classes=2,
            class_labels=["smartphone switched-off", "smartphone switched-on"],
            synchronicity=None,
            feedback_type="none",
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.2617084",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.2617084",
            investigators=["G. Cattan", "P. L. C. Rodrigues", "M. Congedo"],
            senior_author="M. Congedo",
            institution="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            country="FR",
            publication_year=2019,
            associated_paper_doi="10.2312/vriphys.20181064",
            description="This dataset contains electroencephalographic recordings of 12 subjects listening to music with and without a passive head-mounted display",
            keywords=[
                "Electroencephalography (EEG)",
                "Virtual Reality (VR)",
                "Passive Head-Mounted Display (PHMD)",
                "experiment",
            ],
            how_to_acknowledge="Python code for manipulating the data is available at https://github.com/plcrodrigues/py.PHMDML.EEG.2017-GIPSA",
            license="CC-BY-4.0",
        ),
        tags=Tags(pathology=["Healthy"], modality=["EEG"], type=["Resting State"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw, unfiltered",
            preprocessing_applied=False,
            artifact_methods=None,
            notes="Data were acquired with no digital filter. No Faraday cage used to mimic real-world usage.",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["vr_ar"], environment="laboratory", online_feedback=False
        ),
        data_structure=DataStructureMetadata(
            n_blocks=10,
            block_duration_s=60.0,
            trials_context="5 blocks with smartphone switched-off and 5 blocks with smartphone switched-on, randomized sequence",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        data_processed=False,
        file_format="mat and csv",
        abstract="We describe the experimental procedures for a dataset that we have made publicly available at https://doi.org/10.5281/zenodo.2617084 in mat (Mathworks, Natick, USA) and csv formats. This dataset contains electroencephalographic recordings of 12 subjects listening to music with and without a passive head-mounted display, that is, a head-mounted display which does not include any electronics at the exception of a smartphone. The electroencephalographic headset consisted of 16 electrodes. Data were recorded during a pilot experiment taking place in the GIPSA-lab, Grenoble, France, in 2017. Python code for manipulating the data is available at https://github.com/plcrodrigues/py.PHMDML.EEG.2017-GIPSA. The ID of this dataset is PHMDML.EEG.2017-GIPSA.",
        methodology="Subjects sat in front of screen at ~50 cm distance without instrumental noise-reduction devices. EEG cap and Samsung Gear were placed on subject. Smartphones were continuously swapped between switched-on and switched-off conditions. Each block consisted of 1 minute of EEG recording with eyes opened. The sequence of 10 blocks was randomized prior to experiment using random number generator with no autocorrelation. Triggers marked beginning of each block (1=switched-off, 2=switched-on).",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=1,
            events={"off": 1, "on": 2},
            code="Cattan2019-PHMD",  # Before: "PHMD-ML"
            interval=[0, 1],
            paradigm="rstate",
            doi="10.5281/zenodo.2617084",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self._chnames = [
            "Fp1",
            "Fp2",
            "Fc5",
            "Fz",
            "Fc6",
            "T7",
            "Cz",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
            "stim",
        ]
        self._chtypes = ["eeg"] * 16 + ["stim"]

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""

        filepath = self.data_path(subject)[0]
        data = loadmat(os.path.join(filepath, os.listdir(filepath)[0]))

        first_channel = 1
        last_channel = 17
        S = data["data"][:, first_channel:last_channel]
        stim = data["data"][:, -1]

        X = np.concatenate([S * 1e-6, stim[:, None]], axis=1).T

        info = mne.create_info(
            ch_names=self._chnames, sfreq=512, ch_types=self._chtypes, verbose=False
        )
        raw = mne.io.RawArray(data=X, info=info, verbose=False)
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        url = "{:s}subject_{:02d}.mat".format(HEADMOUNTED_URL, subject)
        file_path = dl.data_path(url, "HEADMOUNTED")

        return [file_path]
