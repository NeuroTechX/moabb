"""
BMI/OpenBMI dataset (Motor Imagery).
"""
from functools import partialmethod

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


Lee2019_URL = "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/"

class Lee2019(BaseDataset):
    """BMI/OpenBMI dataset.

    Dataset from Lee et al 2019 [1]_.

    **Dataset Description**

    EEG signals were recorded with a sampling rate of 1,000 Hz and
    collected with 62 Ag/AgCl electrodes. The EEG amplifier used
    in the experiment was a BrainAmp (Brain Products; Munich,
    Germany). The channels were nasion-referenced and grounded
    to electrode AFz. Additionally, an EMG electrode recorded from
    each flexor digitorum profundus muscle with the olecranon
    used as reference. The EEG/EMG channel configuration and
    indexing numbers are described in Fig. 1. The impedances of the
    EEG electrodes were maintained below 10 k during the entire
    experiment.

    MI paradigm
    The MI paradigm was designed following a well-established system protocol.
    For all blocks, the first 3 s of each trial began
    with a black fixation cross that appeared at the center of the
    monitor to prepare subjects for the MI task. Afterwards, the subject
    performed the imagery task of grasping with the appropriate
    hand for 4 s when the right or left arrow appeared as a visual cue.
    After each task, the screen remained blank for 6 s (± 1.5 s). The
    experiment consisted of training and test phases; each phase
    had 100 trials with balanced right and left hand imagery tasks.
    During the online test phase, the fixation cross appeared at the
    center of the monitor and moved right or left, according to the
    real-time classifier output of the EEG signal.

    ERP paradigm
    The interface layout of the speller followed the typical design
    of a row-column speller. The six rows and six columns were
    configured with 36 symbols (A to Z, 1 to 9, and _). Each symbol
    was presented equally spaced. To enhance the
    signal quality, two additional settings were incorporated into
    the original row-column speller design, namely, random-set
    presentation and face stimuli. These additional settings
    help to elicit stronger ERP responses by minimizing adjacency
    distraction errors and by presenting a familiar face image. The
    stimulus-time interval was set to 80 ms, and the inter-stimulus
    interval (ISI) to 135 ms. A single iteration of stimulus presentation
    in all rows and columns was considered a sequence. Therefore,
    one sequence consisted of 12 stimulus flashes. A maximum
    of five sequences (i.e., 60 flashes) was allotted without prolonged
    inter-sequence intervals for each target character. After the end
    of five sequences, 4.5 s were given to the user for identifying, locating,
    and gazing at the next target character. The participant
    was instructed to attend to the target symbol by counting the
    number of times each target character had been flashed.
        In the training session, subjects were asked to copy-spell
    a given sentence, "NEURAL NETWORKS AND DEEP LEARNING"
    (33 characters including spaces) by gazing at the target character
    on the screen. The training session was performed in the offline
    condition, and no feedback was provided to the subject during
    the EEG recording. In the test session, subjects were instructed to
    copy-spell "PATTERN RECOGNITION MACHINE LEARNING"
    (36 characters including spaces), and the real-time EEG data were
    analyzed based on the classifier that was calculated from the
    training session data. The selected character from the subject’s
    current EEG data was displayed in the top left area of the screen
    at the end of the presentation (i.e., after five sequences).
    Per participant, the collected EEG data for the ERP experiment consisted
    of 1,980 and 2,160 trials (samples) for training and test phase, respectively.

    SSVEP paradigm
    Four target SSVEP stimuli were designed to flicker at 5.45, 6.67,
    8.57, and 12 Hz and were presented in four positions (down,
    right, left, and up, respectively) on a monitor. The designed
    paradigm followed the conventional types of SSVEP-based BCI
    systems that require four-direction movements. Partici-
    pants were asked to fixate the center of a black screen and then
    to gaze in the direction where the target stimulus was high-
    lighted in a different color. Each SSVEP stimulus
    was presented for 4 s with an ISI of 6 s. Each target frequency
    was presented 25 times. Therefore, the corrected EEG data had
    100 trials (4 classes x 25 trials) in the offline training phase and
    another 100 trials in the online test phase. Visual feedback was
    presented in the test phase; the estimated target frequency was
    highlighted for 1 s with a red border at the end of each trial.

    Parameters
    ----------
    paradigm: (['MI','ERP','SSVEP'])
        the paradigm to load (see paper).

    train_run: bool (default True)
        if True, return runs corresponding to the training/offline phase (see paper).

    test_run: bool (default False)
        if True, return runs corresponding to the test/online phase (see paper).

    pre_rest_run: bool (default False)
        if True, return runs corresponding to the resting phases before recordings (see paper).

    post_rest_run: bool (default False)
        if True, return runs corresponding to the resting phases after recordings (see paper).

    sessions: list of int (default [1,2])
        the list of the sessions to load (2 available).

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
           Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
           toolbox for three BCI paradigms: An investigation into BCI
           illiteracy. GigaScience, 8(5), 1–16.
           https://doi.org/10.1093/gigascience/giz002
    """

    def __init__(self, paradigm, train_run=True, test_run=False, pre_rest_run=False, post_rest_run=False, sessions=[1, 2]):
        if paradigm.lower() in ['imagery', 'mi']:
            paradigm = 'imagery'
            code_suffix = 'MI'
            interval = [0., 4.] # [1.0, 3.5] is the interval used in paper for online prediction
            events = dict(left_hand=2, right_hand=1)
        elif paradigm.lower() in ['p300', 'erp']:
            paradigm = 'p300'
            code_suffix = 'ERP'
            interval = [0., 1.] # [-0.2, 0.8] is the interval used in paper for online prediction
            events = dict(Target=1, NonTarget=2)
        elif paradigm.lower() in ['ssvep',]:
            paradigm = 'ssvep'
            code_suffix = 'SSVEP'
            interval = [0., 4.]
            events = dict(up=1, left=2, right=3, down=4)
        else:
            raise ValueError('unknown paradigm "{}"'.format(paradigm))
        for s in sessions:
            if s not in [1,2]:
                raise ValueError('inexistant session {}'.format(s))
        self.sessions = sessions

        super().__init__(
            subjects=list(range(1, 55)),
            sessions_per_subject=2,
            events=events,
            code='Lee2019_'+code_suffix,
            interval=interval,
            paradigm=paradigm,
            doi="10.5524/100542",
        )
        self.code_suffix = code_suffix
        self.train_run = train_run
        self .test_run =  test_run
        self. pre_rest_run =  pre_rest_run
        self.post_rest_run = post_rest_run

    def _translate_class(self, c):
        if self.paradigm=='imagery':
            dictionary = dict(
                left_hand=['left'],
                right_hand=['right'],
            )
        elif self.paradigm=='p300':
            dictionary = dict(
                Target=['target'],
                NonTarget=['nontarget'],
            )
        elif self.paradigm=='ssvep':
            dictionary = dict(
                up=['up'],
                left=['left'],
                right=['right'],
                down=['down'],
            )
        for k,v in dictionary.items():
            if c.lower() in v:
                return k
        raise ValueError('unknown class "{}" for "{}" paradigm'.format(c, self.paradigm))

    def _check_mapping(self, file_mapping):
        def raise_error():
            raise ValueError('file_mapping ({}) different than events ({})'.format(file_mapping, self.event_id))
        if len(file_mapping)!=len(self.event_id):
            raise_error()
        for c,v in file_mapping.items():
            v2 = self.event_id.get(self._translate_class(c), None)
            if v!=v2 or v2 is None:
                raise_error()

    def _get_single_run(self, data):
        sfreq = data['fs'].item()
        file_mapping = {c.item(): int(v.item()) for v,c in data['class']}
        self._check_mapping(file_mapping)

        # Create RawArray
        ch_names = [c.item() for c in data['chan'].squeeze()]
        info = create_info(ch_names=ch_names, ch_types=['eeg']*len(ch_names), sfreq=sfreq)
        raw_data = data['x'].transpose(1,0)
        raw = RawArray(data=raw_data, info=info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Create EMG channels
        ch_names = [c.item() for c in data['EMG_index'].squeeze()]
        info = create_info(ch_names=ch_names, ch_types=['emg']*len(ch_names), sfreq=sfreq)
        raw_data = data['EMG'].transpose(1,0)
        emg_raw = RawArray(data=raw_data, info=info, verbose=False)

        # Now create stim chan
        event_times_in_samples = data['t'].squeeze()
        event_id = data['y_dec'].squeeze()
        stim_chan = np.zeros(len(raw))
        for i_sample, id_class in zip(event_times_in_samples, event_id):
            stim_chan[i_sample] += id_class
        info = create_info(ch_names=["STI 014"], sfreq=sfreq, ch_types=["stim"])
        stim_raw = RawArray(stim_chan[None], info, verbose="WARNING")

        # Add events
        event_arr = [
            event_times_in_samples,
            [0] * len(event_times_in_samples),
            event_id,
        ]
        raw.info["events"] = [dict(list=np.array(event_arr).T, channels=None), ]

        # Add EMG and stim channels
        raw = raw.add_channels([emg_raw, stim_raw])
        return raw

    def _get_single_rest_run(self, data, prefix):
        sfreq = data['fs'].item()
        ch_names = [c.item() for c in data['chan'].squeeze()]
        info = create_info(ch_names=ch_names, ch_types=['eeg']*len(ch_names), sfreq=sfreq)
        raw_data = data['{}_rest'.format(prefix)].transpose(1,0)
        raw = RawArray(data=raw_data, info=info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subejct"""

        sessions = {}
        file_path_list = self.data_path(subject)

        for session in self.sessions:
            if self.train_run or self.test_run:
                mat = loadmat(file_path_list[self.sessions.index(session)])

            session_name = "session_{}".format(session)
            sessions[session_name] = {}
            if self.train_run:
                sessions[session_name]['train'] = self._get_single_run(mat['EEG_{}_train'.format(self.code_suffix)][0,0])
            if self. test_run:
                sessions[session_name]['test']  = self._get_single_run(mat['EEG_{}_test' .format(self.code_suffix)][0,0])
            if self. pre_rest_run:
                prefix = 'pre'
                sessions[session_name][ 'test_{}_rest'.format(prefix)]  = self._get_single_rest_run(mat['EEG_{}_test' .format(self.code_suffix)][0,0], prefix)
                sessions[session_name]['train_{}_rest'.format(prefix)]  = self._get_single_rest_run(mat['EEG_{}_train'.format(self.code_suffix)][0,0], prefix)
            if self.post_rest_run:
                prefix = 'post'
                sessions[session_name][ 'test_{}_rest'.format(prefix)]  = self._get_single_rest_run(mat['EEG_{}_test' .format(self.code_suffix)][0,0], prefix)
                sessions[session_name]['train_{}_rest'.format(prefix)]  = self._get_single_rest_run(mat['EEG_{}_train'.format(self.code_suffix)][0,0], prefix)

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []
        for session in self.sessions:
            url = "{0}session{1}/s{2}/sess{1:02d}_subj{2:02d}_EEG_{3}.mat".format(
                Lee2019_URL, session, subject, self.code_suffix
            )
            data_path = dl.data_dl(url, self.code, path, force_update, verbose)
            subject_paths.append(data_path)

        return subject_paths

class Lee2019_MI   (Lee2019):
    __init__ = partialmethod(Lee2019.__init__, 'MI'   )
class Lee2019_ERP  (Lee2019):
    __init__ = partialmethod(Lee2019.__init__, 'ERP'  )
class Lee2019_SSVEP(Lee2019):
    __init__ = partialmethod(Lee2019.__init__, 'SSVEP')
