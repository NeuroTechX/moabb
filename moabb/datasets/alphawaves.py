#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import mne
import numpy as np
from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from scipy.io import loadmat

ALPHAWAVES_URL = 'https://zenodo.org/record/2348892/files/'


class AlphaWaves(BaseDataset):
    '''Dataset containing EEG recordings of subjects in a simple
    resting-state eyes open/closed experimental protocol. Data were recorded
    during a pilot experiment taking place in the GIPSA-lab, Grenoble,
    France, in 2017 [1].

    **Dataset Description**

    This dataset concerns an experiment carried out at GIPSA-lab
    (University of Grenoble Alpes, CNRS, Grenoble-INP) in 2017.
    Principal Investigators : Eng. Grégoire Cattan, Eng. Pedro L. C. Rodrigues
    Scientific Supervisor : Dr. Marco Congedo

    Introduction :

    The occipital dominant rhythm (commonly referred to as occipital ‘Alpha’)
    is prominent in occipital and parietal regions when a subject is exempt of
    visual stimulations, as in the case when keeping the eyes closed [2]. In
    normal subjects its peak frequency is in the range 8-12Hz. The detection of
    alpha waves on the ongoing electroencephalography (EEG) is a useful
    indicator of the subject’s level of stress, concentration, relaxation or
    mental load [3,4] and an easy marker to detect in the recorded signals
    because of its high signal-to-noise-ratio. This experiment was conducted to
    provide a simple yet reliable set of EEG signals carrying very distinct
    signatures on each experimental condition. It can be useful for researchers
    and students looking for an EEG dataset to perform tests with signal
    processing and machine learning algorithms. An example of application of
    this dataset can be seen in [5].

    I. Participants

    A total of 20 volunteers participated in the experiment (7 females), with
    mean (sd) age 25.8 (5.27) and median 25.5. 18 subjects were between 19 and
    28 years old. Two participants with age 33 and 44 were outside this range.

    II. Procedures

    EEG signals were acquired using a standard research grade amplifier
    (g.USBamp, g.tec, Schiedlberg, Austria) and the EC20 cap equipped with 16
    wet electrodes (EasyCap, Herrsching am Ammersee, Germany), placed according
    to the 10-20 international system. The locations of the electrodes were
    FP1, FP2, FC5, FC6, FZ, T7, CZ, T8, P7, P3, PZ, P4, P8, O1, Oz, and O2.
    The reference was placed on the right earlobe and the ground at the AFZ
    scalp location. The amplifier was linked by USB connection to the PC where
    the data were acquired by means of the software OpenVibe [6,7]. We acquired
    the data with no digital filter and a sampling frequency of 512 samples per
    second was used. For ensuing analyses, the experimenter was able to tag the
    EEG signal using an in-house application based on a C/C++ library [8]. The
    tag were sent by the application to the amplifier through the USB port of
    the PC. It was then recorded along with the EEG signal as a supplementary
    channel.

    For each recording we provide the age, genre and fatigue of each
    participant. Fatigue was evaluated by the subjects thanks to a scale
    ranging from 0 to 10, where 10 represents exhaustion. Each participant
    underwent one session consisting of ten blocks of ten seconds of EEG data
    recording. Five blocks were recorded while a subject was keeping his eyes
    closed (condition 1) and the others while his eyes were open (condition 2).
    The two conditions were alternated. Before the onset of each block, the
    subject was asked to close or open his eyes according to the experimental
    condition. The experimenter then tagged the EEG signal using the in-house
    application and started a 10-second countdown of a block.

    III. Organization of the dataset

    For each subject we provide a single .mat file containing the complete
    recording of the session. The file is a 2D-matrix where the rows contain
    the observations at each time sample. Columns 2 to 17 contain the
    recordings on each of the 16 EEG electrodes. The first column of the matrix
    represents the timestamp of each observation and column 18 and 19 contain
    the triggers for the experimental condition 1 and 2. The rows in column 18
    (resp. 19) are filled with zeros, except at the timestamp corresponding to
    the beginning of the block for condition 1 (resp. 2), when the row gets a
    value of one.

    We supply an online and open-source example working with Python [9].

    References
    ----------

    .. [1] Cattan G, Andreev A, Mendoza C, Congedo M. The Impact of Passive
           Head-Mounted Virtual Reality Devices on the Quality of EEG Signals.
           In Delft: The Eurographics Association; 2018 [cited 2018 Apr 16].

    .. [2] Pfurtscheller G, Stancák A, Neuper C. Event-related synchronization
           (ERS) in the alpha band — an electrophysiological correlate of
           cortical idling: A review. Int J Psychophysiol. 1996 Nov
           1;24(1):39–46.

    .. [3] Banquet JP. Spectral analysis of the EEG in meditation.
           Electroencephalogr Clin Neurophysiol. 1973 Aug 1;35(2):143–51.

    .. [4] Antonenko P, Paas F, Grabner R, van Gog T. Using
           Electroencephalography to Measure Cognitive Load.
           Educ Psychol Rev. 2010 Dec 1;22(4):425–38.

    .. [5] Rodrigues PLC, Congedo M, Jutten C. Multivariate Time-Series
           Analysis Via Manifold Learning. In: 2018 IEEE Statistical Signal
           Processing Workshop (SSP). 2018. p. 573–7.

    .. [6] Renard Y, Lotte F, Gibert G, Congedo M, Maby E, Delannoy V, et al.
           OpenViBE: An Open-Source Software Platform to Design, Test, and Use
           Brain–Computer Interfaces in Real and Virtual Environments.
           Presence Teleoperators Virtual Environ. 2010 Feb 1;19(1):35–53.

    .. [7] Arrouët C, Congedo M, Marvie J-E, Lamarche F, Lécuyer A, Arnaldi B.
           Open-ViBE: A Three Dimensional Platform for Real-Time Neuroscience.
           J Neurother. 2005 Jul 8;9(1):3–25.

    .. [8] Mandal MK. C++ Library for Serial Communication with Arduino
           [Internet]. 2016 [cited 2018 Dec 15]. Available from:
           https://github.com/manashmndl/SerialPort

    .. [9] Rodrigues PLC. Alpha-Waves-Dataset [Internet].
           Grenoble: GIPSA-lab; 2018. Available from:
           https://github.com/plcrodrigues/Alpha-Waves-Dataset

    '''

    def __init__(self):

        subject_list = list(range(1, 6+1)) + list(range(8, 20+1))
        self.subject_list = subject_list

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        filepath = self.data_path(subject)[0]
        data = loadmat(filepath)

        S = data['SIGNAL'][:, 1:17]
        stim_close = data['SIGNAL'][:, 17]
        stim_open = data['SIGNAL'][:, 18]
        stim = 1 * stim_close + 2 * stim_open

        chnames = [
            'Fp1',
            'Fp2',
            'Fc5',
            'Fz',
            'Fc6',
            'T7',
            'Cz',
            'T8',
            'P7',
            'P3',
            'Pz',
            'P4',
            'P8',
            'O1',
            'Oz',
            'O2',
            'stim']
        chtypes = ['eeg'] * 16 + ['stim']
        X = np.concatenate([S, stim[:, None]], axis=1).T

        info = mne.create_info(ch_names=chnames, sfreq=512,
                               ch_types=chtypes,
                               verbose=False)
        raw = mne.io.RawArray(data=X, info=info, verbose=False)

        return raw

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        url = '{:s}subject_{:02d}.mat'.format(ALPHAWAVES_URL, subject)
        file_path = dl.data_path(url, 'ALPHAWAVES')

        return [file_path]