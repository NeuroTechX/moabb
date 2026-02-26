import os

import mne
import numpy as np
from scipy.io import loadmat

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)
from moabb.utils import depreciated_alias

from . import download as dl
from .base import BaseDataset


HEADMOUNTED_URL = "https://zenodo.org/record/2617085/files/"


@depreciated_alias("HeadMountedDisplay", "1.1")
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
            montage="10-10",
            hardware="g.tec",
            sensor_type="wet electrodes",
            reference="right earlobe",
            software="OpenVibe",
            filters="no digital filter",
            sensors=[
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "CP3",
                "CPz",
                "CP4",
                "Pz",
            ],
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            gender={"male": 9, "female": 3},
            age_mean=26.25,
        ),
        experiment=ExperimentMetadata(
            paradigm="rstate",
            trial_duration=60,
            study_design="focus on the marker and to listen to the music that was diffused during \nthe experiment (Bach Invention from one to ten on harpsichord).",
            stimulus_type="avatar",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.2617084",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.2617084",
            license="CC-BY-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Other"],
            type=["Other"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw, unfiltered",
            preprocessing_applied=False,
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        bci_application=BCIApplicationMetadata(
            applications=["vr_ar"],
            environment="outdoor",
        ),
        data_processed=False,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=1,
            events=dict(on=1, off=2),
            code="Cattan2019-PHMD",  # Before: "PHMD-ML"
            interval=[0, 1],
            paradigm="rstate",
            doi="10.5281/zenodo.2617084",
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
