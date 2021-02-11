"""
SSVEP MAMEM2 dataset.
"""

from .base import BaseDataset
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.datasets.utils import _get_path
import glob
import os
import logging
import numpy as np

try:
    import wfdb
except ImportError:
    raise ImportError("Loading this dataset requires installing wfdb")
# Heads up on wfdb, has a problem in the latest release for windows
# and continues to have the problem, (Issue #254 on wfdb-python)
# better to do pip install git+https://github.com/MIT-LCP/wfdb-python.git

log = logging.getLogger()

# Alternate Download Location
# MAMEM1_URL = 'https://ndownloader.figshare.com/articles/3413851/versions/1'

MAMEM1_URL = "https://archive.physionet.org/physiobank/database/mssvepdb/dataset3/"  # noqa: E501


class MAMEM3(BaseDataset):
    """
    EEG signals with 14 channels captured from 11 subjects executing a SSVEP-based experimental protocol. Five
    different frequencies (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual stimulation, and
    the Emotiv EPOC, using 14 wireless channels has been used for capturing the signals.

    Subjects were exposed to flickering lights from five magenta boxes with frequencies [6.66Hz, 7.5Hz, 8.57Hz
    10Hz and 12Hz] simultaneously. Prior to and during each flicking window, one of the boxes is marked by a
    yellow arrow indicating the box to be focused on by the subject. The Emotiv EPOC 14 channel wireless EEG
    headset was used to capture the subjects' signals.

    Each subject underwent a single adaptation period before the first of their 5 sessions (unlike experiment
    1 in which each session began with its own adaptation period). In the adaptation period, the subject is
    exposed to ten 5-second flickering windows from the five boxes simultaneously, with the target frequencies
    specified in the FREQUENCIES3.txt file under 'adaptation'. The flickering windows are separated by 5 seconds
    of rest, and the 100s adaptation period precedes the first session by 30 seconds.

    Each session consisted of the following:
    For the series of frequencies specified in the FREQUENCIES3.txt file under 'sessions':
        A 5 second window with all boxes flickering and the subject focusing on the specified frequency's marked
        box, followed by 5 seconds of rest. Between the 12th and 13th flickering window, there is a 30s resting
        period.
        This gives a total of 25 flickering windows for each session (not including the first adaptation period).
    Five minutes of rest before the next session (not including the 5th session).

    The order of chosen frequencies is the same for each session, although there are small-moderate variations
    in the actual frequencies of each individual window.

    *Note: Each 'session' in experiment 1 includes an adaptation period, unlike experiment 2 and 3 where each
    subject undergoes only one adaptation period before their first 'session'.

    Waveforms and Annotations
    File names are in the form U0NNn, where NN is the subject number and n is a - e for the session letter or
    x for the adaptation period. In addition, session file names end with either i or ii, corresponding to the
    first 12 or second 13 windows of the session respectively. Each session lasts in the order of several minutes
    and is sampled at 128Hz.
    Each session half and adaptation period has the following files:

    A waveform file of the EEG signals (.dat) along with its header file (.hea).
    An annotation file (.win) containing the locations of the beginning and end of each 5 second flickering window.
    The annotations are labeled as '(' for start and ')' for stop, along with auxiliary strings indicating the
    focal frequency of the flashing windows.

    The FREQUENCIES3.txt file indicates the approximate marked frequencies of the flickering windows, equal for
    each session, adaptation, and subject. These values are equal to those contained in the .win annotations.

    Trial  manipulation:
    The  trial  initiation  is  defined by  an  event  code  (32779)  and  the end by another (32780). There
    are five different labels that indicate the box subjects were  instructed to focus  on  (1,  2,  3,  4
    and  5)  and  correspond to frequencies  12.00, 10.00, 8.57, 7.50 and 6.66 Hz respectively.
    5 3 2 1 4 5 2 1 4 3 is the trial sequence for  the  adaptation and
    4  2  3  5  1  2  5  4  2  3  1  5  4  3  2  4  1  2  5  3  4  1  3  1  3 is  the sequence for each
    session.

    Observed  artifacts:
    During  the  stimulus  presentation to  subject  S007  the  research stuff noted that the subject had a
    tendency to eye blink. As a result the interference, in matters of artifacts, on the recorded signal is
    expected to be high.

    References -
    [1] https://archive.physionet.org/physiobank/database/mssvepdb/
    [2] DataAcquisitionDetails.pdf on
    https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_III_14_channels_11_subjects_5_frequencies_presented_simultaneously_/3413851  # noqa: E501
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=5,
            events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
            code="SSVEP MAMEM3",
            # Some part is cut so that only the "good" signal is obtained
            interval=[1, 4],
            paradigm="ssvep",
            doi="https://arxiv.org/abs/1602.00904",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fnames = self.data_path(subject)
        sessions = {}

        for fpath in fnames:
            fnamed = os.path.basename(fpath)
            session_name = 'session_' + fnamed[3]
            # Since the data for each session is saved in 2 files, it is being saved in 2 runs
            run_number = len(fnamed) - 6
            run_name = 'run_' + str(run_number)
            record = wfdb.rdrecord(fpath)
            if session_name not in sessions.keys():
                sessions[session_name] = {}

            data = record.p_signal.T
            annots = wfdb.rdann(fpath, "win")
            # the number of samples isn't exactly equal in all the trials
            n_samples = record.sig_len
            # n_channels, n_trials = 14, 25
            stim_freq = np.array([float(e) for e in self.event_id.keys()])
            # aux_note are the exact frequencies, matched to nearest class
            events_label = [np.argmin(np.abs(stim_freq - float(f))) + 1
                            for f in annots.aux_note]
            raw_events = np.zeros([1, n_samples])
            #  annots.sample indicates the start of the trial
            # of class "events_label"
            for label, samploc in zip(events_label, annots.sample):
                raw_events[0, samploc] = label
            # append the data as another channel(stim) in the data
            data = np.concatenate((data, raw_events), axis=0)
            ch_names = record.sig_name
            ch_names.append("stim")
            ch_types = ["eeg"] * 14 + ["stim"]
            sfreq = 128

            info = create_info(ch_names, sfreq, ch_types)
            raw = RawArray(data, info, verbose=False)
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage)
            sessions[session_name].update({run_name: raw})
        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        # Check if the .dat, .hea and .win files are present
        # The .dat and .hea files give the main data
        # .win file gives the event windows and the frequencies
        # .flash file can give the exact times of the flashes if necessary
        # Return the file paths depending on the number of sessions for each
        # subject that are denoted a, b, c, ...
        sub = "{:02d}".format(subject)
        sign = "MAMEM3"
        key = "MNE_DATASETS_{:s}_PATH".format(sign)
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = _get_path(path, key, sign)
        path = os.path.join(path, key_dest)
        s_paths = glob.glob(os.path.join(
            path, "dataset3/U0{}*.dat".format(sub)))
        subject_paths = []
        for name in s_paths:
            subject_paths.append(os.path.splitext(name)[0])
        # if files for the subject are not present
        if not subject_paths or force_update:
            # if not downloaded, get the list of files to download
            datarec = wfdb.get_record_list("mssvepdb")
            datalist = []
            for ele in datarec:
                if "dataset3/U0{}".format(sub) in ele:
                    datalist.append(ele)
            wfdb.io.dl_database("mssvepdb", path, datalist,
                                annotators="win", overwrite=force_update)
        # Return the file paths depending on the number of sessions
        s_paths = glob.glob(os.path.join(
            path, "dataset3", "U0{}*.dat".format(sub)))
        subject_paths = []
        for name in s_paths:
            # The adaptation session has the letter x at the end
            # So this is being done to remove the adaptation session from the file name to be returned
            if (os.path.splitext(name)[0][-1]) != 'x':
                subject_paths.append(os.path.splitext(name)[0])
        return subject_paths
