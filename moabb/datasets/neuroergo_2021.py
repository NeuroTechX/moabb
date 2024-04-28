import mne
from moabb.datasets.base import BaseDataset
import os
import glob
import zipfile as z
from moabb.datasets import download as dl
import os.path as osp
import numpy as np

URL = "https://zenodo.org/record/5055046/files/"

events = dict(rs=1, easy=2, medium=3, diff=4)
class Neuroergonomics2021Dataset(BaseDataset):
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 16)),  # 15 participants
            sessions_per_subject=2,  # 2 sessions per subject
            events=events,
            code='Neuroergonomics2021',
            interval=[0, 1],  # Epochs are 2-second long
            paradigm='rstate' 
        )

    def _get_single_subject_data(self, subject):
        """Load data for a single subject."""
        data = {}

        subject_path = self.data_path(subject)[0]

        for session in range(1, self.n_sessions + 1):
            session_path = os.path.join(subject_path, f"S{session}/eeg/")
            
            # get 'resting state'
            rs = os.path.join(session_path, f"alldata_sbj{str(subject).zfill(2)}_sess{session}_{'RS'}.set")
            rs_epochs = mne.io.read_epochs_eeglab(rs)

            # get task 'easy'
            easy = os.path.join(session_path, f"alldata_sbj{str(subject).zfill(2)}_sess{session}_{'MATBeasy'}.set")
            easy_epochs = mne.io.read_epochs_eeglab(easy)
           
            # get task 'med'
            med = os.path.join(session_path, f"alldata_sbj{str(subject).zfill(2)}_sess{session}_{'MATBmed'}.set")
            med_epochs = mne.io.read_epochs_eeglab(med)
            
            # get task 'diff'
            diff = os.path.join(session_path, f"alldata_sbj{str(subject).zfill(2)}_sess{session}_{'MATBdiff'}.set")
            diff_epochs = mne.io.read_epochs_eeglab(diff)

            # concatenate raw data
            raw_data = np.concatenate((
                rs_epochs.get_data(),
                easy_epochs.get_data(),
                med_epochs.get_data(),
                diff_epochs.get_data()))
            # reshape data in the form n_channel x n_sample
            raw_data = raw_data.transpose((1, 0, 2))
            n_channel, n_epochs, n_samples = raw_data.shape
            raw_data = raw_data.reshape((n_channel, n_epochs * n_samples))
            
            # add stim channel
            n_epochs_rs = rs_epochs.get_data().shape[0]
            n_epochs_easy = easy_epochs.get_data().shape[0]
            n_epochs_med = med_epochs.get_data().shape[0]
            # n_epochs_diff = diff_epochs.get_data().shape[0]
            stim = np.zeros((1, n_epochs * n_samples))
            for i in range(n_epochs):
                stim[0, n_samples * i + 1] = \
                    events['rs'] if i < n_epochs_rs else \
                    events['easy'] if i < n_epochs_rs + n_epochs_easy else \
                    events['medium'] if i < n_epochs_rs + n_epochs_easy + n_epochs_med \
                    else events['diff']

            raw_data = np.concatenate((raw_data, stim))
            
            # create info
            self._chnames = [str(i) for i in range((raw_data.shape[0] - 1))] + ['stim'] # TODO: real chnames and location
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

