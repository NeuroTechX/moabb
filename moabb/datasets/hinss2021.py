import os
import os.path as osp
import zipfile as z

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


URL = "https://zenodo.org/record/5055046/files/"
# private dictionary to map events to integers
_HINNS_EVENTS = dict(rs=1, easy=2, medium=3, diff=4)


class Hinss2021(BaseDataset):
    """Neuroergonomic 2021 dataset.

    We describe the experimental procedures for a dataset that is publicly available
    at https://zenodo.org/records/5055046.
    This dataset contains electroencephalographic recordings of 15 subjects (6 female, with an
    average age of 25 years). A total of 62 active Ag–AgCl
    electrodes were available in the dataset.

    The participants engaged in 3 (2 available here) distinct experimental sessions, each of which
    was separated by 1 week.

    At the beginning of each session, the resting state of the participant
    (measured as 1 minute with eyes open) was recorded.

    Subsequently, participants undertook 3 tasks of varying difficulty levels
    (i.e., easy, medium, and difficult). The task assignments
    were randomized. For more details, please check [Hinss2021]_.

    Notes
    -----

    .. versionadded:: 1.0.1

    References
    ----------

    .. [Hinss2021] M. Hinss, B. Somon, F. Dehais & R. N. Roy (2021)
            Open EEG Datasets for Passive Brain-Computer
            Interface Applications: Lacks and Perspectives.
            IEEE Neural Engineering Conference.
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 16)),  # 15 participants
            sessions_per_subject=2,  # 2 sessions per subject
            events=_HINNS_EVENTS,
            code="Hinss2021",
            interval=[0, 2],  # Epochs are 2-second long
            paradigm="rstate",
        )

    def _get_stim_channel(self, rs_epochs, easy_epochs, med_epochs, n_epochs, n_samples):
        n_epochs_rs = rs_epochs.get_data().shape[0]
        n_epochs_easy = easy_epochs.get_data().shape[0]
        n_epochs_med = med_epochs.get_data().shape[0]
        stim = np.zeros((1, n_epochs * n_samples))
        for i in range(n_epochs):
            stim[0, n_samples * i + 1] = (
                _HINNS_EVENTS["rs"]
                if i < n_epochs_rs
                else (
                    _HINNS_EVENTS["easy"]
                    if i < n_epochs_rs + n_epochs_easy
                    else (
                        _HINNS_EVENTS["medium"]
                        if i < n_epochs_rs + n_epochs_easy + n_epochs_med
                        else _HINNS_EVENTS["diff"]
                    )
                )
            )
        return stim

    def _get_epochs(self, session_path, subject, session, event_file):
        raw = os.path.join(
            session_path,
            f"alldata_sbj{str(subject).zfill(2)}_sess{session}_{event_file}.set",
        )
        epochs = mne.io.read_epochs_eeglab(raw)
        return epochs

    def _get_single_subject_data(self, subject):
        """Load data for a single subject."""
        data = {}

        subject_path = self.data_path(subject)[0]

        for session in range(1, self.n_sessions + 1):
            session_path = os.path.join(subject_path, f"S{session}/eeg/")

            # get 'resting state'
            rs_epochs = self._get_epochs(session_path, subject, session, "RS")

            # get task 'easy'
            easy_epochs = self._get_epochs(session_path, subject, session, "MATBeasy")

            # get task 'med'
            med_epochs = self._get_epochs(session_path, subject, session, "MATBmed")

            # get task 'diff'
            diff_epochs = self._get_epochs(session_path, subject, session, "MATBdiff")

            # concatenate raw data
            raw_data = np.concatenate(
                (
                    rs_epochs.get_data(),
                    easy_epochs.get_data(),
                    med_epochs.get_data(),
                    diff_epochs.get_data(),
                )
            )

            # reshape data in the form n_channel x n_sample
            raw_data = raw_data.transpose((1, 0, 2))
            n_channel, n_epochs, n_samples = raw_data.shape
            raw_data = raw_data.reshape((n_channel, n_epochs * n_samples))

            # add stim channel
            stim = self._get_stim_channel(
                rs_epochs, easy_epochs, med_epochs, n_epochs, n_samples
            )
            raw_data = np.concatenate((raw_data, stim))

            # create info
            self._chnames = rs_epochs.ch_names + ["stim"]
            self._chtypes = ["eeg"] * (raw_data.shape[0] - 1) + ["stim"]

            info = mne.create_info(
                ch_names=self._chnames, sfreq=500, ch_types=self._chtypes, verbose=False
            )
            raw = mne.io.RawArray(raw_data, info)

            # Only one run => "0"
            data[str(session)] = {"0": raw}

        return data

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # check if has the .zip
        url = f"{URL}P{subject:02}.zip"

        path_zip = dl.data_dl(url, "Neuroergonomics2021")
        path_folder = path_zip.strip(f"P{subject:02}.zip")

        # check if has to unzip
        if not (osp.isdir(path_folder + f"P{subject:02}")) and not (
            osp.isdir(path_folder + f"P{subject:02}")
        ):
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        final_path = f"{path_folder}P{subject:02}"
        return [final_path]
