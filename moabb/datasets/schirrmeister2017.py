import logging

import mne
from mne.channels import make_standard_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)

GIN_URL = (
    "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"  # noqa
)


class Schirrmeister2017(BaseDataset):
    """High-gamma dataset discribed in Schirrmeister et al. 2017

    .. admonition:: Dataset summary


        =================  =======  =======  ==========  =================  ============  ===============  ===========
        Name                 #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =================  =======  =======  ==========  =================  ============  ===============  ===========
        Schirrmeister2017       14      128           4                120  4s            500Hz                      1
        =================  =======  =======  ==========  =================  ============  ===============  ===========

    Dataset from [1]_

    Our “High-Gamma Dataset” is a 128-electrode dataset (of which we later only use
    44 sensors covering the motor cortex, (see Section 2.7.1), obtained from 14
    healthy subjects (6 female, 2 left-handed, age 27.2 ± 3.6 (mean ± std)) with
    roughly 1000 (963.1 ± 150.9, mean ± std) four-second trials of executed
    movements divided into 13 runs per subject.  The four classes of movements were
    movements of either the left hand, the right hand, both feet, and rest (no
    movement, but same type of visual cue as for the other classes).  The training
    set consists of the approx.  880 trials of all runs except the last two runs,
    the test set of the approx.  160 trials of the last 2 runs.  This dataset was
    acquired in an EEG lab optimized for non-invasive detection of high- frequency
    movement-related EEG components (Ball et al., 2008; Darvas et al., 2010).

    Depending on the direction of a gray arrow that was shown on black back-
    ground, the subjects had to repetitively clench their toes (downward arrow),
    perform sequential finger-tapping of their left (leftward arrow) or right
    (rightward arrow) hand, or relax (upward arrow).  The movements were selected
    to require little proximal muscular activity while still being complex enough
    to keep subjects in- volved.  Within the 4-s trials, the subjects performed the
    repetitive movements at their own pace, which had to be maintained as long as
    the arrow was showing.  Per run, 80 arrows were displayed for 4 s each, with 3
    to 4 s of continuous random inter-trial interval.  The order of presentation
    was pseudo-randomized, with all four arrows being shown every four trials.
    Ideally 13 runs were performed to collect 260 trials of each movement and rest.
    The stimuli were presented and the data recorded with BCI2000 (Schalk et al.,
    2004).  The experiment was approved by the ethical committee of the University
    of Freiburg.

    References
    ----------

    .. [1] Schirrmeister, Robin Tibor, et al. "Deep learning with convolutional
           neural networks for EEG decoding and visualization." Human brain mapping 38.11
           (2017): 5391-5420.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events=dict(right_hand=1, left_hand=2, rest=3, feet=4),
            code="Schirrmeister2017",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1002/hbm.23730",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        def _url(prefix):
            return "/".join([GIN_URL, prefix, "{:d}.edf".format(subject)])

        return [
            dl.data_dl(_url(t), "SCHIRRMEISTER2017", path, force_update, verbose)
            for t in ["train", "test"]
        ]

    def _get_single_subject_data(self, subject):
        train_raw, test_raw = [
            mne.io.read_raw_edf(path, infer_types=True, preload=True)
            for path in self.data_path(subject)
        ]

        # Select only EEG sensors (remove EOG, EMG),
        # and also set montage for visualizations
        montage = make_standard_montage("standard_1005")
        train_raw, test_raw = [
            raw.pick_types(eeg=True).set_montage(montage) for raw in (train_raw, test_raw)
        ]
        sessions = {
            "session_0": {"train": train_raw, "test": test_raw},
        }
        return sessions
