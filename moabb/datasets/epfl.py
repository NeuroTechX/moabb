import datetime as dt
import glob
import os
import zipfile

import mne
import numpy as np
from mne.channels import make_standard_montage
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


EPFLP300_URL = "http://documents.epfl.ch/groups/m/mm/mmspg/www/BCI/p300/"


class EPFLP300(BaseDataset):
    """P300 dataset from Hoffmann et al 2008.

    Dataset from the paper [1]_.

    **Dataset Description**

    In the present work a six-choice P300 paradigm is tested using a population
    of five disabled and four able-bodied subjects. Six different images were
    flashed in random order with a stimulus interval of 400 ms. Users were
    facing a laptop screen on which six im- ages were displayed. The images
    showed a television, a telephone, a lamp, a door, a window, and a radio.

    The images were flashed in random sequences, one image at a time. Each
    flash of an image lasted for 100 ms and during the following 300 ms none of
    the images was flashed, i.e. the interstimulus interval was 400 ms. The EEG
    was recorded at 2048 Hz sampling rate from 32 electrodes placed at the
    standard positions of the 10-20 international system. The system was tested
    with five disabled and four healthy subjects. The disabled subjects were
    all wheelchair-bound but had varying communication and limb muscle control
    abilities (Subjects 1 to 5). In particular, Subject 5 was only able
    to perform extremely slow and relatively uncontrolled movements with hands
    and arms. Due to a severe hypophony and large fluctuations in the level of
    alertness, communication with subject 5 was very difficult, which is why
    its data is not available in this dataset. Subjects 6 to 9 were PhD
    students recruited from our laboratory (all male, age 30 ± 2.3).

    Each subject completed four recording sessions. The first two sessions were
    performed on one day and the last two sessions on another day. For all
    subjects the time between the first and the last session was less than two
    weeks. Each of the sessions consisted of six runs, one run for each of the
    six images. The duration of one run was approximately one minute and the
    duration of one session including setup of electrodes and short breaks
    between runs was approximately 30 minutes. One session comprised on average
    810 trials, and the whole data for one subject consisted on average of 3240
    trials.

    References
    ----------

    .. [1] Hoffmann, U., Vesin, J-M., Ebrahimi, T., Diserens, K., 2008.
           An efficient P300-based brain-computer interfacefor disabled
           subjects. Journal of Neuroscience Methods .
           https://doi.org/10.1016/j.jneumeth.2007.03.005
    """

    def __init__(self):
        super().__init__(
            subjects=[1, 2, 3, 4, 6, 7, 8, 9],
            sessions_per_subject=4,
            events=dict(Target=2, NonTarget=1),
            code="EPFL P300 dataset",
            interval=[0, 1],
            paradigm="p300",
            doi="10.1016/j.jneumeth.2007.03.005",
        )

    def _get_single_run_data(self, file_path):

        # data from the .mat
        data = loadmat(file_path)
        signals = data["data"]
        stimuli = data["stimuli"].squeeze()
        events = data["events"]
        target = data["target"][0][0]

        # meta-info from the readme.pdf
        sfreq = 2048
        # fmt: off
        ch_names = [
            "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3",
            "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
            "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz", "MA1", "MA2",
        ]
        # fmt: on
        ch_types = ["eeg"] * 32 + ["misc"] * 2

        # The last X entries are 0 for all signals. This leads to
        # artifacts when epoching and band-pass filtering the data.
        # Correct the signals for this.
        sig_i = np.where(np.diff(np.all(signals == 0, axis=0).astype(int)) != 0)[0][0]
        signals = signals[:, :sig_i]
        signals *= 1e-6  # data is stored as uV, but MNE expects V
        # we have to re-reference the signals
        # the average signal on the mastoids electrodes is used as reference
        references = [32, 33]
        ref = np.mean(signals[references, :], axis=0)
        signals = signals - ref

        # getting the event time in a Python standardized way
        events_datetime = []
        for eventi in events:
            events_datetime.append(
                dt.datetime(*eventi.astype(int), int(eventi[-1] * 1e3) % 1000 * 1000)
            )

        # get the indices of the stimuli
        pos = []
        n_trials = len(stimuli)
        for j in range(n_trials):
            delta_seconds = (events_datetime[j] - events_datetime[0]).total_seconds()
            delta_indices = int(delta_seconds * sfreq)
            # has to add an offset
            pos.append(delta_indices + int(0.4 * sfreq))

        # create a stimulus channel
        stim_aux = np.copy(stimuli)
        stim_aux[stimuli == target] = 2
        stim_aux[stimuli != target] = 1
        stim_channel = np.zeros(signals.shape[1])
        stim_channel[pos] = stim_aux
        ch_names = ch_names + ["STI"]
        ch_types = ch_types + ["stim"]
        signals = np.concatenate([signals, stim_channel[None, :]])

        # create info dictionary
        info = mne.create_info(ch_names, sfreq, ch_types)
        info["description"] = "EPFL P300 dataset"

        # create the Raw structure
        raw = mne.io.RawArray(signals, info, verbose=False)
        montage = make_standard_montage("biosemi32")
        raw.set_montage(montage)

        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}

        for file_path in sorted(file_path_list):

            session_name = "session_" + file_path.split(os.sep)[-2].replace("session", "")

            if session_name not in sessions.keys():
                sessions[session_name] = {}

            run_name = "run_" + str(len(sessions[session_name]) + 1)
            sessions[session_name][run_name] = self._get_single_run_data(file_path)

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # check if has the .zip
        url = "{:s}subject{:d}.zip".format(EPFLP300_URL, subject)
        path_zip = dl.data_dl(url, "EPFLP300")
        path_folder = path_zip.strip("subject{:d}.zip".format(subject))

        # check if has to unzip
        if not (os.path.isdir(path_folder + "subject{:d}".format(subject))):
            print("unzip", path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        # get the path to all files
        pattern = os.path.join("subject{:d}".format(subject), "*", "*")
        subject_paths = glob.glob(path_folder + pattern)

        return subject_paths
