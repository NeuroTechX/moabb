"""
SSVEP MAMEM1 dataset.
"""

import logging
import os.path as osp

import numpy as np
import pooch
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from .base import BaseDataset
from .download import (
    fs_get_file_hash,
    fs_get_file_id,
    fs_get_file_list,
    fs_get_file_name,
    get_dataset_path,
)


log = logging.getLogger(__name__)

MAMEM_URL = "https://ndownloader.figshare.com/files/"
# Specific release
# MAMEM1_URL = 'https://ndownloader.figshare.com/articles/2068677/versions/6'
# MAMEM2_URL = 'https://ndownloader.figshare.com/articles/3153409/versions/4'
# MAMEM3_URL = 'https://ndownloader.figshare.com/articles/3413851/versions/3'

# Alternate Download Location
# MAMEM1_URL = "https://archive.physionet.org/physiobank/database/mssvepdb/dataset1/"
# MAMEM2_URL = "https://archive.physionet.org/physiobank/database/mssvepdb/dataset2/"
# MAMEM3_URL = "https://archive.physionet.org/physiobank/database/mssvepdb/dataset3/"


def mamem_event(eeg, dins, labels=None):
    """Convert DIN field into events

    Code adapted from https://github.com/MAMEM/eeg-processing-toolbox
    """
    thres_split = 2000
    timestamps = dins[1, :]
    samples = dins[3, :]
    numDins = dins.shape[1]

    sampleA = samples[0]
    previous = timestamps[0]
    t_start, freqs = [], []
    s, c = 0, 0
    for i in range(1, numDins):
        current = timestamps[i]
        if (current - previous) > thres_split:
            sampleB = samples[i - 1]
            freqs.append(s // c)
            if (sampleB - sampleA) > 382:
                t_start.append(sampleA)
            sampleA = samples[i]
            s = 0
            c = 0
        else:
            s = s + (current - previous)
            c = c + 1
        previous = timestamps[i]
    sampleB = samples[i - 1]
    freqs.append(s // c)
    t_start.append(sampleA)
    freqs = np.array(freqs, dtype=np.int) * 2
    freqs = 1000 // freqs
    t_start = np.array(t_start)

    if labels is None:
        freqs_labels = {6: 1, 7: 2, 8: 3, 9: 4, 11: 5}
        for f, t in zip(freqs, t_start):
            eeg[-1, t] = freqs_labels[f]
    else:
        for f, t in zip(labels, t_start):
            eeg[-1, t] = f
    return eeg


class BaseMAMEM(BaseDataset):
    """Base class for MAMEM datasets"""

    def __init__(self, events, sessions_per_subject, code, doi, figshare_id):
        super().__init__(
            subjects=list(range(1, 11)),
            events=events,
            interval=[1, 4],
            paradigm="ssvep",
            sessions_per_subject=sessions_per_subject,
            code=code,
            doi=doi,
        )
        self.figshare_id = figshare_id

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fnames = self.data_path(subject)
        filelist = fs_get_file_list(self.figshare_id)
        fsn = fs_get_file_name(filelist)
        sessions = {}

        for fpath in fnames:
            fnamed = fsn[osp.basename(fpath)]
            if fnamed[4] == "x":
                continue
            session_name = "session_" + fnamed[4]
            if self.code == "SSVEP MAMEM3":
                # Since the data for each session is saved in 2 files,
                # it is being saved in 2 runs
                run_number = len(fnamed) - 10
                run_name = "run_" + str(run_number)
            else:
                run_name = "run_0"

            if self.code == "SSVEP MAMEM3":
                m = loadmat(fpath)
                ch_names = [e[0] for e in m["info"][0, 0][9][0]]
                sfreq = 128
                montage = make_standard_montage("standard_1020")
                eeg = m["eeg"]
            else:
                m = loadmat(fpath, squeeze_me=True)
                ch_names = ["E{}".format(i + 1) for i in range(0, 256)]
                ch_names.append("stim")
                # ch_names = ["{}-{}".format(s, i) if s == "EEG" else s
                #             for i, s in enumerate(record.sig_name)]
                sfreq = 250
                if self.code == "SSVEP MAMEM2":
                    labels = m["labels"]
                else:
                    labels = None
                eeg = mamem_event(m["eeg"], m["DIN_1"], labels=labels)
                montage = make_standard_montage("GSN-HydroCel-256")
            ch_types = ["eeg"] * (len(ch_names) - 1) + ["stim"]
            info = create_info(ch_names, sfreq, ch_types)
            raw = RawArray(eeg, info, verbose=False)
            raw.set_montage(montage)
            if session_name not in sessions.keys():
                sessions[session_name] = {}
            if len(sessions[session_name]) == 0:
                sessions[session_name] = {run_name: raw}
            else:
                sessions[session_name][run_name] = raw
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sub = "{:02d}".format(subject)
        sign = self.code.split()[1]
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = osp.join(get_dataset_path(sign, path), key_dest)

        filelist = fs_get_file_list(self.figshare_id)
        reg = fs_get_file_hash(filelist)
        fsn = fs_get_file_id(filelist)
        gb = pooch.create(path=path, base_url=MAMEM_URL, registry=reg)

        spath = []
        for f in fsn.keys():
            if f[2:4] == sub:
                spath.append(gb.fetch(fsn[f]))
        return spath


class MAMEM1(BaseMAMEM):
    """SSVEP MAMEM 1 dataset

    Dataset from [1]_.

    EEG signals with 256 channels captured from 11 subjects executing a
    SSVEP-based experimental protocol. Five different frequencies
    (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual
    stimulation,and the EGI 300 Geodesic EEG System (GES 300), using a
    stimulation, HydroCel Geodesic Sensor Net (HCGSN) and a sampling rate of
    250 Hz has been used for capturing the signals.

    Check the technical report [2]_ for more detail.
    From [1]_, subjects were exposed to non-overlapping flickering lights from five
    magenta boxes with frequencies [6.66Hz, 7.5Hz, 8.57Hz 10Hz and 12Hz].
    256 channel EEG recordings were captured.

    Each session of the experimental procedure consisted of the following:

    1. 100 seconds of rest.
    2. An adaptation period in which the subject is exposed to eight
       5 second windows of flickering from a magenta box. Each flickering
       window is of a single isolated frequency, randomly chosen from the
       above set, specified in the FREQUENCIES1.txt file under
       'adaptation'. The individual flickering windows are separated by 5
       seconds of rest.
    3. 30 seconds of rest.
    4. For each of the frequencies from the above set in ascending order,
       also specified in FREQUENCIES1.txt under 'main trials':

       1. Three 5 second windows of flickering at the chosen frequency,
           separated by 5 seconds of rest.
       2. 30 seconds of rest.

    This gives a total of 15 flickering windows, or 23 including the
    adaptation period.

    The order of chosen frequencies is the same for each session, although
    there are small-moderate variations in the actual frequencies of each
    individual window. The .freq annotations list the different frequencies at
    a higher level of precision.

    **Note**: Each 'session' in experiment 1 includes an adaptation period, unlike
    experiment 2 and 3 where each subject undergoes only one adaptation period
    before their first 'session'.

    From [3]_:

    **Eligible signals**: The EEG signal is sensitive to external factors that have
    to do with the environment or the configuration of the acquisition setup
    The research stuff was responsible for the elimination of trials that were
    considered faulty. As a result the following sessions were noted and
    excluded from further analysis:
    1. S003, during session 4 the stimulation program crashed
    2. S004, during session 2 the stimulation program crashed, and
    3. S008, during session 4 the Stim Tracker was detuned.
    Furthermore, we must also note that subject S001 participated in 3 sessions
    and subjects S003 and S004 participated in 4 sessions, compared to all
    other subjects that participated in 5 sessions (NB: in fact, there is only
    3 sessions for subjects 1, 3 and 8, and 4 sessions for subject 4 available
    to download). As a result, the utilized dataset consists of 1104 trials of
    5 seconds each.

    **Flickering frequencies**: Usually the refresh rate for an LCD Screen is 60 Hz
    creating a restriction to the number of frequencies that can be selected.
    Specifically, only the frequencies that when divided with the refresh rate
    of the screen result in an integer quotient could be selected. As a result,
    the frequendies that could be obtained were the following: 30.00. 20.00,
    15.00, 1200, 10.00, 857. 7.50 and 6.66 Hz. In addition, it is also
    important to avoid using frequencies that are multiples of another
    frequency, for example making the choice to use 10.00Hz prohibits the use
    of 20.00 and 30.00 Mhz. With the previously described limitations in mind,
    the selected frequencies for the experiment were: 12.00, 10.00, 8.57, 7.50
    and 6.66 Hz.

    **Stimuli Layout**: In an effort to keep the experimental process as simple as
    possible, we used only one flickering box instead of more common choices,
    such as 4 or 5 boxes flickering simultaneously The fact that the subject
    could focus on one stimulus without having the distraction of other
    flickering sources allowed us to minimize the noise of our signals and
    verify the appropriateness of our acquisition setup Nevertheless, having
    concluded the optimal configuration for analyzing the EEG signals, the
    experiment will be repeated with more concurrent visual stimulus.

    **Trial duration**: The duration of each trial was set to 5 seconds, as this
    time was considered adequate to allow the occipital part of the bran to
    mimic the stimulation frequency and still be small enough for making a
    selection in the context

    References
    ----------
    .. [1] MAMEM Steady State Visually Evoked Potential EEG Database
           `<https://archive.physionet.org/physiobank/database/mssvepdb/>`_
    .. [2] V.P. Oikonomou et al, 2016, Comparative evaluation of state-of-the-art
           algorithms for SSVEP-based BCIs. arXiv.
           `<http://arxiv.org/abs/1602.00904>`-
    .. [3] S. Nikolopoulos, 2016, DataAcquisitionDetails.pdf
           `<https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_I_256_channels_11_subjects_5_frequencies_/2068677?file=3793738>`_  # noqa: E501
    """

    def __init__(self):
        super().__init__(
            events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
            sessions_per_subject=3,
            # 3 for S001, S003, S008, 4 for S004
            code="SSVEP MAMEM1",
            doi="https://arxiv.org/abs/1602.00904",
            figshare_id=2068677,
        )


class MAMEM2(BaseMAMEM):
    """SSVEP MAMEM 2 dataset

    Dataset from [1]_.

    EEG signals with 256 channels captured from 11 subjects executing a
    SSVEP-based experimental protocol. Five different frequencies
    (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual
    stimulation,and the EGI 300 Geodesic EEG System (GES 300), using a
    stimulation, HydroCel Geodesic Sensor Net (HCGSN) and a sampling rate of
    250 Hz has been used for capturing the signals.

    Subjects were exposed to flickering lights from five violet boxes with
    frequencies [6.66Hz, 7.5Hz, 8.57Hz, 10Hz, and 12Hz] simultaneously. Prior
    to and during each flicking window, one of the boxes is marked by a yellow
    arrow indicating the box to be focused on by the subject. 256 channel EEG
    recordings were captured.

    From [2]_, each subject underwent a single adaptation period before the first of
    their 5 sessions (unlike experiment 1 in which each session began with its own
    adaptation period). In the adaptation period, the subject is exposed to ten
    5-second flickering windows from the five boxes simultaneously, with the
    target frequencies specified in the FREQUENCIES2.txt file under
    'adaptation'. The flickering windows are separated by 5 seconds of rest,
    and the 100s adaptation period precedes the first session by 30 seconds.

    Each session consisted of the following:
    For the series of frequencies specified in the FREQUENCIES2.txt file under
    'sessions':
    A 5 second window with all boxes flickering and the subject focusing
    on the specified frequency's marked box, followed by 5 seconds of rest.
    This gives a total of 25 flickering windows for each session (not
    including the first adaptation period). Five minutes of rest before
    the next session (not including the 5th session).

    The order of chosen frequencies is the same for each session, although
    there are small-moderate variations in the actual frequencies of each
    individual window.
    **Note**: Each 'session' in experiment 1 includes an adaptation period,
    unlike experiment 2 and 3 where each subject undergoes only one adaptation
    period before their first 'session'.

    **Waveforms and Annotations**
    File names are in the form T0NNn, where NN is the subject number and n is
    a - e for the session letter or x for the adaptation period. Each session
    lasts in the order of several minutes and is sampled at 250Hz. Each session
    and adaptation period has the following files:
    A waveform file of the EEG signals (.dat) along with its header file
    (.hea). If the channel corresponds to an international 10-20 channel then
    it is labeled as such. Otherwise, it is just labeled 'EEG'. An annotation
    file (.flash) containing the locations of each individual flash. An
    annotation file (.win) containing the locations of the beginning and end
    of each 5 second flickering window. The annotations are labeled as '(' for
    start and ')' for stop, along with auxiliary strings indicating the focal
    frequency of the flashing windows.

    The FREQUENCIES2.txt file indicates the approximate marked frequencies of
    the flickering windows, equal for each session, adaptation, and subject.
    These values are equal to those contained in the .win annotations.

    **Observed  artifacts:**
    During the  stimulus  presentation  to  subject  S007  the  research stuff
    noted that the subject had a tendency to eye blink. As a result the
    interference, in matters of artifacts, on the recorded signal is expected
    to be high.

    References
    ----------
    .. [1] MAMEM Steady State Visually Evoked Potential EEG Database
           `<https://archive.physionet.org/physiobank/database/mssvepdb/>`_
    .. [2] S. Nikolopoulos, 2016, DataAcquisitionDetails.pdf
           `<https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_II_256_channels_11_subjects_5_frequencies_presented_simultaneously_/3153409?file=4911931>`_  # noqa: E501
    """

    def __init__(self):
        super().__init__(
            events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
            sessions_per_subject=5,
            code="SSVEP MAMEM2",
            doi="https://arxiv.org/abs/1602.00904",
            figshare_id=3153409,
        )


class MAMEM3(BaseMAMEM):
    """SSVEP MAMEM 3 dataset

    Dataset from [1]_.

    EEG signals with 14 channels captured from 11 subjects executing a
    SSVEP-based experimental protocol. Five different frequencies
    (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual
    stimulation, and the Emotiv EPOC, using 14 wireless channels has been used
    for capturing the signals.

    Subjects were exposed to flickering lights from five magenta boxes with
    frequencies [6.66Hz, 7.5Hz, 8.57Hz, 10Hz and 12Hz] simultaneously. Prior
    to and during each flicking window, one of the boxes is marked by a yellow
    arrow indicating the box to be focused on by the subject. The Emotiv EPOC
    14 channel wireless EEG headset was used to capture the subjects' signals.

    Each subject underwent a single adaptation period before the first of their
    5 sessions (unlike experiment 1 in which each session began with its own
    adaptation period). In the adaptation period, the subject is exposed to ten
    5-second flickering windows from the five boxes simultaneously, with the
    target frequencies specified in the FREQUENCIES3.txt file under
    'adaptation'. The flickering windows are separated by 5 seconds of rest,
    and the 100s adaptation period precedes the first session by 30 seconds.

    Each session consisted of the following:
    For the series of frequencies specified in the FREQUENCIES3.txt file under
    'sessions':
    A 5 second window with all boxes flickering and the subject focusing on
    the specified frequency's marked box, followed by 5 seconds of rest.
    Between the 12th and 13th flickering window, there is a 30s resting
    period. This gives a total of 25 flickering windows for each session
    (not including the first adaptation period). Five minutes of rest
    before the next session (not including the 5th session).

    The order of chosen frequencies is the same for each session, although
    there are small-moderate variations in the actual frequencies of each
    individual window.

    **Note**: Each 'session' in experiment 1 includes an adaptation period, unlike
    experiment 2 and 3 where each subject undergoes only one adaptation period
    before their first 'session' [2]_.

    **Waveforms and Annotations**
    File names are in the form U0NNn, where NN is the subject number and n is
    a - e for the session letter or x for the adaptation period. In addition,
    session file names end with either i or ii, corresponding to the first 12
    or second 13 windows of the session respectively. Each session lasts in the
    order of several minutes and is sampled at 128Hz.
    Each session half and adaptation period has the following files:
    A waveform file of the EEG signals (.dat) along with its header file
    (.hea). An annotation file (.win) containing the locations of the beginning
    and end of each 5 second flickering window. The annotations are labeled as
    '(' for start and ')' for stop, along with auxiliary strings indicating the
    focal frequency of the flashing windows.

    The FREQUENCIES3.txt file indicates the approximate marked frequencies of
    the flickering windows, equal for each session, adaptation, and subject.
    These values are equal to those contained in the .win annotations.

    **Trial  manipulation**:
    The  trial  initiation  is  defined by  an  event  code  (32779)  and  the
    end by another (32780). There are five different labels that indicate the
    box subjects were instructed to focus  on  (1, 2, 3, 4 and 5) and
    correspond to frequencies 12.00, 10.00, 8.57, 7.50 and 6.66 Hz respectively.
    5 3 2 1 4 5 2 1 4 3 is the trial sequence for the adaptation and
    4 2 3 5 1 2 5 4 2 3 1 5 4 3 2 4 1 2 5 3 4 1 3 1 3 is the sequence for each
    session.

    **Observed  artifacts**:
    During  the  stimulus  presentation to  subject  S007  the  research staff
    noted that the subject had a tendency to eye blink. As a result the
    interference, in matters of artifacts, on the recorded signal is expected
    to be high.

    References
    ----------
    .. [1] MAMEM Steady State Visually Evoked Potential EEG Database
           `<https://archive.physionet.org/physiobank/database/mssvepdb/>`_
    .. [2] S. Nikolopoulos, 2016, DataAcquisitionDetails.pdf
           `<https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_III_14_channels_11_subjects_5_frequencies_presented_simultaneously_/3413851>`_  # noqa: E501
    """

    def __init__(self):
        super().__init__(
            events={
                "6.66": 33029,
                "7.50": 33028,
                "8.57": 33027,
                "10.00": 33026,
                "12.00": 33025,
            },
            sessions_per_subject=5,
            code="SSVEP MAMEM3",
            doi="https://arxiv.org/abs/1602.00904",
            figshare_id=3413851,
        )
