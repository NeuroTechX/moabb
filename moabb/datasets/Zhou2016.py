"""
Simple and compound motor imagery.
https://doi.org/10.1371/journal.pone.0114853
"""

import os
import shutil
import zipfile as z

import numpy as np
from mne.channels import make_standard_montage
from mne.datasets.utils import _get_path
from mne.io import read_raw_cnt
from pooch import retrieve

from .base import BaseDataset


DATA_PATH = "https://ndownloader.figshare.com/files/3662952"


def local_data_path(base_path, subject):
    if not os.path.isdir(os.path.join(base_path, "subject_{}".format(subject))):
        if not os.path.isdir(os.path.join(base_path, "data")):
            retrieve(DATA_PATH, None, fname="data.zip", path=base_path)
            with z.ZipFile(os.path.join(base_path, "data.zip"), "r") as f:
                f.extractall(base_path)
            os.remove(os.path.join(base_path, "data.zip"))
        datapath = os.path.join(base_path, "data")
        for i in range(1, 5):
            os.makedirs(os.path.join(base_path, "subject_{}".format(i)))
            for session in range(1, 4):
                for run in ["A", "B"]:
                    os.rename(
                        os.path.join(datapath, "S{}_{}{}.cnt".format(i, session, run)),
                        os.path.join(
                            base_path,
                            "subject_{}".format(i),
                            "{}{}.cnt".format(session, run),
                        ),
                    )
        shutil.rmtree(os.path.join(base_path, "data"))
    subjpath = os.path.join(base_path, "subject_{}".format(subject))
    return [
        [os.path.join(subjpath, "{}{}.cnt".format(y, x)) for x in ["A", "B"]]
        for y in ["1", "2", "3"]
    ]


class Zhou2016(BaseDataset):
    """Motor Imagery dataset from Zhou et al 2016.

    Dataset from the article *A Fully Automated Trial Selection Method for
    Optimization of Motor Imagery Based Brain-Computer Interface* [1]_.
    This dataset contains data recorded on 4 subjects performing 3 type of
    motor imagery: left hand, right hand and feet.

    Every subject went through three sessions, each of which contained two
    consecutive runs with several minutes inter-run breaks, and each run
    comprised 75 trials (25 trials per class). The intervals between two
    sessions varied from several days to several months.

    A trial started by a short beep indicating 1 s preparation time,
    and followed by a red arrow pointing randomly to three directions (left,
    right, or bottom) lasting for 5 s and then presented a black screen for
    4 s. The subject was instructed to immediately perform the imagination
    tasks of the left hand, right hand or foot movement respectively according
    to the cue direction, and try to relax during the black screen.

    References
    ----------

    .. [1] Zhou B, Wu X, Lv Z, Zhang L, Guo X (2016) A Fully Automated
           Trial Selection Method for Optimization of Motor Imagery Based
           Brain-Computer Interface. PLoS ONE 11(9).
           https://doi.org/10.1371/journal.pone.0162657
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 5)),
            sessions_per_subject=3,
            events=dict(left_hand=1, right_hand=2, feet=3),
            code="Zhou 2016",
            # MI 1-6s, prepare 0-1, break 6-10
            # boundary effects
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1371/journal.pone.0162657",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        files = self.data_path(subject)

        out = {}
        for sess_ind, runlist in enumerate(files):
            sess_key = "session_{}".format(sess_ind)
            out[sess_key] = {}
            for run_ind, fname in enumerate(runlist):
                run_key = "run_{}".format(run_ind)
                raw = read_raw_cnt(fname, preload=True, eog=["VEOU", "VEOL"])
                stim = raw.annotations.description.astype(np.dtype("<10U"))
                stim[stim == "1"] = "left_hand"
                stim[stim == "2"] = "right_hand"
                stim[stim == "3"] = "feet"
                raw.annotations.description = stim
                out[sess_key][run_key] = raw
                out[sess_key][run_key].set_montage(make_standard_montage("standard_1005"))
        return out

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        key = "MNE_DATASETS_ZHOU2016_PATH"
        path = _get_path(path, key, "Zhou 2016")
        basepath = os.path.join(path, "MNE-zhou-2016")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)
        return local_data_path(basepath, subject)
