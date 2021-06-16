"""
BMI/OpenBMI dataset (Motor Imagery).
"""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


Lee2019_URL = "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/"


class Lee2019_MI(BaseDataset):
    """Motor Imagery BMI/OpenBMI dataset from BMI/OpenBMI dataset.

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

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
           Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI
           toolbox for three BCI paradigms: An investigation into BCI
           illiteracy. GigaScience, 8(5), 1–16.
           https://doi.org/10.1093/gigascience/giz002
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 55)),
            sessions_per_subject=2,
            events=dict(left_hand=2, right_hand=1),
            code="Lee2019_MI",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5524/100542",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subejct"""

        sessions = {}
        file_path_list = self.data_path(subject)

        for session in range(1, 3):
            data = loadmat(file_path_list[session - 1])

            # Create channel info and montage
            eeg_ch_names = data["EEG_MI_train"][0, 0][8][0]
            ch_names = [elem[0] for elem in eeg_ch_names] + ["stim"]
            ch_types = ["eeg"] * 62 + ["stim"]
            sfreq = data["EEG_MI_train"][0, 0][3][0, 0]
            info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
            montage = make_standard_montage("standard_1005")

            # Create raw_data
            raw_train_data = np.transpose(data["EEG_MI_train"][0, 0][0], (1, 2, 0))
            raw_test_data = np.transpose(data["EEG_MI_test"][0, 0][0], (1, 2, 0))
            raw_data = np.concatenate([raw_train_data, raw_test_data], axis=0)

            # Create raw_event
            train_event_id = data["EEG_MI_train"][0, 0][4].ravel()
            test_event_id = data["EEG_MI_test"][0, 0][4].ravel()
            event_id = np.concatenate([train_event_id, test_event_id], axis=0)
            raw_events = np.zeros((raw_data.shape[0], 1, raw_data.shape[2]))
            raw_events[:, 0, 0] = event_id

            # Zero pad the data
            data = np.concatenate([raw_data, raw_events], axis=1)
            zeroshape = (data.shape[0], data.shape[1], 50)
            data = np.concatenate(
                [np.zeros(zeroshape), data, np.zeros(zeroshape)], axis=2
            )

            # Create RawArray
            raw = RawArray(
                data=np.concatenate(list(data), axis=1), info=info, verbose=False
            )
            raw.set_montage(montage)

            # add the data to sessions
            session_name = "session_{}".format(session)
            sessions[session_name] = {"run_1": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []
        for session in range(1, 3):
            url = "{0}session{1}/s{2}/sess{1:02d}_subj{2:02d}_EEG_MI.mat".format(
                Lee2019_URL, session, subject
            )
            data_path = dl.data_dl(url, "Lee2019_MI", path, force_update, verbose)
            subject_paths.append(data_path)

        return subject_paths
