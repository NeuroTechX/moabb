"""
BMI/OpenBMI dataset (Motor Imagery).
"""
from functools import partial

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

    Parameters
    ----------
    paradigm: (['MI','ERP','SSVEP'])
        the paradigm to load (see paper).

    train_run: bool (default True)
        if True, return runs corresponding to the training/offline phase (see paper).

    test_run: bool (default False)
        if True, return runs corresponding to the test/online phase (see paper).

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
           Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
           toolbox for three BCI paradigms: An investigation into BCI
           illiteracy. GigaScience, 8(5), 1–16.
           https://doi.org/10.1093/gigascience/giz002
    """

    def __init__(self, paradigm, train_run=True, test_run=False):
        if paradigm.lower() in ['imagery', 'mi']:
            paradigm = 'imagery'
            code_suffix = 'MI'
            interval = [1.0, 3.5]
            events = dict(left_hand=2, right_hand=1),
        elif paradigm.lower() in ['p300', 'erp']:
            paradigm = 'p300'
            code_suffix = 'ERP'
            interval = [-0.2, 0.8]
            events = dict(Target=1, NonTarget=2) ## TODO: put real values and names
        elif paradigm.lower() in ['ssvep',]:
            paradigm = 'ssvep'
            code_suffix = 'SSVEP'
            interval = [0., 4.]
            events = dict(up=1, left=2, right=3, down=4) ## TODO: put real values and names
        else:
            raise ValueError('unknown paradigm "{}"'.format(paradigm))

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
        self.test_run  =  test_run

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

    def _get_single_run(self, data):
        sfreq = data['fs'].item()

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

    def _get_single_subject_data(self, subject):
        """return data for a single subejct"""

        sessions = {}
        file_path_list = self.data_path(subject)

        for session in range(1, 3):
            if self.train_run or self.test_run:
                mat = loadmat(file_path_list[session - 1])

            session_name = "session_{}".format(session)
            sessions[session_name] = {}
            if self.train_run:
                sessions[session_name]['train'] = self._get_single_run(mat['EEG_{}_train'.format(self.code_suffix)][0,0])
            if self. test_run:
                sessions[session_name]['test']  = self._get_single_run(mat['EEG_{}_test' .format(self.code_suffix)][0,0])

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

Lee2019_MI    = partial(Lee2019, paradigm='MI')
Lee2019_ERP   = partial(Lee2019, paradigm='ERP')
Lee2019_SSVEP = partial(Lee2019, paradigm='SSVEP')
