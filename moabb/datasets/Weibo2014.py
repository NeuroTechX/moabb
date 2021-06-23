"""
Simple and compound motor imagery
https://doi.org/10.1371/journal.pone.0114853
"""

import logging
import os
import shutil

import mne
import numpy as np
from mne.datasets.utils import _get_path
from pooch import Unzip, retrieve
from scipy.io import loadmat

from .base import BaseDataset


log = logging.getLogger(__name__)

FILES = []
FILES.append("https://dataverse.harvard.edu/api/access/datafile/2499178")
FILES.append("https://dataverse.harvard.edu/api/access/datafile/2499182")
FILES.append("https://dataverse.harvard.edu/api/access/datafile/2499179")


def eeg_data_path(base_path, subject):
    file1_subj = ["cl", "cyy", "kyf", "lnn"]
    file2_subj = ["ls", "ry", "wcf"]
    file3_subj = ["wx", "yyx", "zd"]

    def get_subjects(sub_inds, sub_names, ind):
        dataname = "data{}".format(ind)
        if not os.path.isfile(os.path.join(base_path, dataname + ".zip")):
            retrieve(FILES[ind], None, dataname + ".zip", base_path, processor=Unzip())

        for fname in os.listdir(os.path.join(base_path, dataname + ".zip.unzip")):
            for ind, prefix in zip(sub_inds, sub_names):
                if fname.startswith(prefix):
                    os.rename(
                        os.path.join(base_path, dataname + ".zip.unzip", fname),
                        os.path.join(base_path, "subject_{}.mat".format(ind)),
                    )
        os.remove(os.path.join(base_path, dataname + ".zip"))
        shutil.rmtree(os.path.join(base_path, dataname + ".zip.unzip"))

    if not os.path.isfile(os.path.join(base_path, "subject_{}.mat".format(subject))):
        if subject in range(1, 5):
            get_subjects(list(range(1, 5)), file1_subj, 0)
        elif subject in range(5, 8):
            get_subjects(list(range(5, 8)), file2_subj, 1)
        elif subject in range(8, 11):
            get_subjects(list(range(8, 11)), file3_subj, 2)
    return os.path.join(base_path, "subject_{}.mat".format(subject))


class Weibo2014(BaseDataset):
    """Motor Imagery dataset from Weibo et al 2014.

    Dataset from the article *Evaluation of EEG oscillatory patterns and
    cognitive process during simple and compound limb motor imagery* [1]_.

    It contains data recorded on 10 subjects, with 60 electrodes.

    This dataset was used to investigate the differences of the EEG patterns
    between simple limb motor imagery and compound limb motor
    imagery. Seven kinds of mental tasks have been designed, involving three
    tasks of simple limb motor imagery (left hand, right hand, feet), three
    tasks of compound limb motor imagery combining hand with hand/foot
    (both hands, left hand combined with right foot, right hand combined with
    left foot) and rest state.

    At the beginning of each trial (8 seconds), a white circle appeared at the
    center of the monitor. After 2 seconds, a red circle (preparation cue)
    appeared for 1 second to remind the subjects of paying attention to the
    character indication next. Then red circle disappeared and character
    indication (‘Left Hand’, ‘Left Hand & Right Foot’, et al) was presented on
    the screen for 4 seconds, during which the participants were asked to
    perform kinesthetic motor imagery rather than a visual type of imagery
    while avoiding any muscle movement. After 7 seconds, ‘Rest’ was presented
    for 1 second before next trial (Fig. 1(a)). The experiments were divided
    into 9 sections, involving 8 sections consisting of 60 trials each for six
    kinds of MI tasks (10 trials for each MI task in one section) and one
    section consisting of 80 trials for rest state. The sequence of six MI
    tasks was randomized. Intersection break was about 5 to 10 minutes.

    References
    -----------
    .. [1] Yi, Weibo, et al. "Evaluation of EEG oscillatory patterns and
           cognitive process during simple and compound limb motor imagery."
           PloS one 9.12 (2014). https://doi.org/10.1371/journal.pone.0114853
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=dict(
                left_hand=1,
                right_hand=2,
                hands=3,
                feet=4,
                left_hand_right_foot=5,
                right_hand_left_foot=6,
                rest=7,
            ),
            code="Weibo 2014",
            # Full trial w/ rest is 0-8
            interval=[3, 7],
            paradigm="imagery",
            doi="10.1371/journal.pone.0114853",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname = self.data_path(subject)
        # TODO: add 1s 0 buffer between trials and make continuous
        data = loadmat(
            fname,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )
        montage = mne.channels.make_standard_montage("standard_1005")

        # fmt: off
        ch_names = [
            "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
            "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
            "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2",
            "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
            "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2", "VEO", "HEO",
        ]
        # fmt: on

        ch_types = ["eeg"] * 62 + ["eog"] * 2
        # FIXME not sure what are those CB1 / CB2
        ch_types[57] = "misc"
        ch_types[61] = "misc"
        info = mne.create_info(
            ch_names=ch_names + ["STIM014"], ch_types=ch_types + ["stim"], sfreq=200
        )
        # until we get the channel names montage is None
        event_ids = data["label"].ravel()
        raw_data = np.transpose(data["data"], axes=[2, 0, 1])
        # de-mean each trial
        raw_data = raw_data - np.mean(raw_data, axis=2, keepdims=True)
        raw_events = np.zeros((raw_data.shape[0], 1, raw_data.shape[2]))
        raw_events[:, 0, 0] = event_ids
        data = np.concatenate([1e-6 * raw_data, raw_events], axis=1)
        # add buffer in between trials
        log.warning(
            "Trial data de-meaned and concatenated with a buffer to create " "cont data"
        )
        zeroshape = (data.shape[0], data.shape[1], 50)
        data = np.concatenate([np.zeros(zeroshape), data, np.zeros(zeroshape)], axis=2)
        raw = mne.io.RawArray(
            data=np.concatenate(list(data), axis=1), info=info, verbose=False
        )
        raw.set_montage(montage)
        return {"session_0": {"run_0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        key = "MNE_DATASETS_WEIBO2014_PATH"
        path = _get_path(path, key, "Weibo 2014")
        basepath = os.path.join(path, "MNE-weibo-2014")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)
        return eeg_data_path(basepath, subject)
