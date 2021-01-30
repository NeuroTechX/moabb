"""
SSVEP MAMEM1 dataset.
"""

# from . import download as dl

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
# MAMEM1_URL = 'https://ndownloader.figshare.com/articles/2068677/versions/5'
# foldername = '2068677'

MAMEM1_URL = 'https://archive.physionet.org/physiobank/database/mssvepdb/dataset1/'


class MAMEM1(BaseDataset):
    """EEG signals with 256 channels captured from 11 subjects executing a
    SSVEP-based experimental protocol. Five different frequencies
    (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual stimulation,
    and the EGI 300 Geodesic EEG System (GES 300), using a 256-channel
    HydroCel Geodesic Sensor Net (HCGSN) and a sampling rate of 250 Hz
    has been used for capturing the signals.

    Check http://arxiv.org/abs/1602.00904 for the technical report.
    From [1]
    The Experiment Details -
    Subjects were exposed to non-overlapping flickering lights from five magenta boxes with
    frequencies [6.66Hz, 7.5Hz, 8.57Hz 10Hz and 12Hz]. 256 channel EEG recordings were captured.

    Each session of the experimental procedure consisted of the following:

        1. 100 seconds of rest.
        2. An adaptation period in which the subject is exposed to eight 5 second windows
        of flickering from a magenta box. Each flickering window is of a single isolated
        frequency, randomly chosen from the above set, specified in the FREQUENCIES1.txt
        file under 'adaptation'. The individual flickering windows are separated by 5 seconds of rest.
        3. 30 seconds of rest.
        4. For each of the frequencies from the above set in ascending order, also specified
        in FREQUENCIES1.txt under 'main trials':
            a. Three 5 second windows of flickering at the chosen frequency, separated by 5 seconds of rest.
            b. 30 seconds of rest.

        This gives a total of 15 flickering windows, or 23 including the adaptation period.

    The order of chosen frequencies is the same for each session, although there are small-moderate
    variations in the actual frequencies of each individual window. The .freq annotations list the
    different frequencies at a higher level of precision.

    *Note: Each 'session' in experiment 1 includes an adaptation period, unlike experiment 2
    and 3 where each subject undergoes only one adaptation period before their first 'session'.

    From [2]
    Important Notes
    Eligible signals: The EEG signal is sensitive to external factors that have to do
    with the environment or the configuration of the acquisition setup The research stuff
    was responsible for the elimination of trials that were considered faulty. As a result
    the following sessions were noted and excluded from further analysis:
      I) S003, during session 4 the stimulation program crashed
      II) S004, during session 2 the stimulation program crashed, and
      III) S008, during session 4 the Stim Tracker was detuned.
      Furthermore, we must also note that subject S001 participated in 3 sessions and
      subjects S003 and S004 participated in 4 sessions, compared to all other subjects
      that participated in 5 sessions.
    As a result, the utilized dataset consists of 1104 trials of 5 seconds each

    Flickering frequencies: Usually the refresh rate for an LCD Screen is 60 Hz creating a
    restriction to the number of frequencies that can be selected. Specifically, only the
    frequencies that when divided with the refresh rate of the screen result in an integer
    quotient could be selected. As a result, the frequendies that could be obtained were the
    following: 30.00. 20.00, 15.00, 1200, 10.00, 857. 7.50 and 6.66 Hz. In addition, it is
    also important to avoid using frequencies that are multiples of another frequency, for
    example making the choice to use 10.00Hz prohibits the use of 20.00 and 30.00 Mhz. With
    the previously described limitations in mind, the selected frequencies for the experiment
    were: 12.00, 10.00, 8.57, 7.50 and 6.66 Hz.

    Stimuli Layout: In an effort to keep the experimental process as simple as possible, we
    used only one flickering box instead of more common choices, such as 4 or 5 boxes flickering
    simultaneously The fact that the subject could focus on one stimulus without having the
    distraction of other flickering sources allowed us to minimize the noise of our signals
    and verify the appropriateness of our acquisition setup Nevertheless, having concluded
    the optimal configuration for analyzing the EEG signals, the experiment will be repeated
    with more concurrent visual stimulus.

    Trial duration: The duration of each trial was set to 5 seconds, as this time was considered
    adequate to allow the occipital part of the bran to mimic the stimulation frequency and still
    be small enough for making a selection in the context

    [1] https://archive.physionet.org/physiobank/database/mssvepdb/
    [2] DataAcquisitionDetails.pdf on
    https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_I_256_channels_11_subjects_5_frequencies_/2068677?file=3793738

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,  # Has to be set properly
            events={'6.66': 1, '7.50': 2, '8.57': 3, '10.00': 4, '12.00': 5},
            code='SSVEP MAMEM1',
            interval=[1, 4],  # Some part is cut so that only the "good" signal is obtained
            paradigm='ssvep',
            doi='https://arxiv.org/abs/1602.00904')

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        fnames = self.data_path(subject)
        sessions = {}

        for fpath in fnames:

            session_name = fpath[-1]
            record = [wfdb.rdrecord(fpath)]
            if session_name not in sessions.keys():
                sessions[session_name] = {}

            # do print(record.__dict__["p_signal"]) for the sig and print(record.__dict__) for all the contents
            data = (record.__dict__["p_signal"]).T
            annots = [wfdb.rdann(fpath, 'win')]

            # the number of samples isn't exactly equal in all of the samples..
            n_samples, n_channels, n_trials = record.__dict__["sig_len"], 256, 23
            # n_classes = len(self.event_id)

            # In annots.__dict__ the "sample" has the start and end of each trial and "aux_note" has the exact frequency
            stim_freq = np.array([float(e) for e in self.event_id.keys()])
            # The annotations file has the exact frequencies of the stimuli, so the class is found here
            events_label = [np.argmin(np.abs(stim_freq - float(f))) + 1 for f in annots.__dict__['aux_note']]
            raw_events = np.zeros([1, n_samples])

            # The events structure for ssvep is such that it has the class mentioned in the stim channel at the sample
            # where the corresponding trial starts
            for label, samploc in zip(events_label, annots.__dict__['samples']):
                raw_events[0, samploc] = label

            # append the data as another channel(stim) in the data
            np.append(data, raw_events, axis=0)

            ch_names = record.__dict__["sig_name"]
            ch_names.append('stim')
            ch_types = ['eeg']*256 + ['stim']
            sfreq = 250
            info = create_info(ch_names, sfreq, ch_types)
            raw = RawArray(data, info, verbose=False)

            # Generate the montage
            montage = make_standard_montage('GSN-HydroCel-256')
            raw.set_montage(montage)

            sessions[session_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # Currently the entire check and download is being performed locally without
        # calling the usual download function due to the wfdb functions being used for
        # physionet. This can be changed by updating the download function in a later update

        # Check if the .dat, .hea and .win files are present
        # The .dat and .hea files give the main data
        # The .win file gives the event windows and the frequencies in the windows
        # .flash file can give the exact times of the flashes if necessary
        # Return the file paths depending on the number of sessions for each subject

        if subject < 10:
            sub = '0{}'.format(subject)
        else:
            sub = '{}'.format(subject)

        sign = 'MAMEM1'
        key = 'MNE_DATASETS_{:s}_PATH'.format(sign)
        key_dest = 'MNE-{:s}-data'.format(sign.lower())
        path = _get_path(path, key, sign)
        path = os.path.join(path, key_dest)

        s_paths = glob.glob(os.path.join(path, 'S0{}*.dat'.format(sub)))
        subject_paths = []
        for name in s_paths:
            subject_paths.append(os.path.splitext(name)[0])

        # if files for the subject are not present
        if not subject_paths or force_update:

            # if not downloaded, get the list of files to download
            datarec = wfdb.get_record_list('mssvepdb')
            datalist = []
            for ele in datarec:
                if "dataset1/S0{}".format(sub) in ele:
                    datalist.append(ele)

            wfdb.io.dl_database('mssvepdb', path, datalist, annotators='win')

            # Return the file paths depending on the number of sessions for each subject
            s_paths = glob.glob(os.path.join(path, 'dataset1', 'S0{}*.dat'.format(sub)))
            subject_paths = []
            for name in s_paths:
                subject_paths.append(os.path.splitext(name)[0])

        return subject_paths
