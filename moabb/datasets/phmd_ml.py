import os

import mne
import numpy as np
from scipy.io import loadmat

from moabb.utils import depreciated_alias

from . import download as dl
from .base import BaseDataset


HEADMOUNTED_URL = "https://zenodo.org/record/2617085/files/"


@depreciated_alias("HeadMountedDisplay", "1.1")
class Cattan2019_PHMD(BaseDataset):
    """Passive Head Mounted Display with Music Listening dataset.

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

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=1,
            events=dict(on=1, off=2),
            code="Cattan2019-PHMD",  # Before: "PHMD-ML"
            interval=[0, 1],
            paradigm="rstate",
            doi="https://doi.org/10.5281/zenodo.2617084 ",
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

        X = np.concatenate([S, stim[:, None]], axis=1).T

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
