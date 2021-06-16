"""
Munich MI dataset
"""

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


DOWNLOAD_URL = "https://zenodo.org/record/1217449/files/"


class MunichMI(BaseDataset):
    """Munich Motor Imagery dataset.

    Motor imagery dataset from Grosse-Wentrup et al. 2009 [1]_.

    A trial started with the central display of a white fixation cross. After 3
    s, a white arrow was superimposed on the fixation cross, either pointing to
    the left or the right.
    Subjects were instructed to perform haptic motor imagery of the
    left or the right hand during display of the arrow, as indicated by the
    direction of the arrow. After another 7 s, the arrow was removed,
    indicating the end of the trial and start of the next trial. While subjects
    were explicitly instructed to perform haptic motor imagery with the
    specified hand, i.e., to imagine feeling instead of visualizing how their
    hands moved, the exact choice of which type of imaginary movement, i.e.,
    moving the fingers up and down, gripping an object, etc., was left
    unspecified.
    A total of 150 trials per condition were carried out by each subject,
    with trials presented in pseudorandomized order.

    Ten healthy subjects (S1–S10) participated in the experimental
    evaluation. Of these, two were females, eight were right handed, and their
    average age was 25.6 years with a standard deviation of 2.5 years. Subject
    S3 had already participated twice in a BCI experiment, while all other
    subjects were naive to BCIs. EEG was recorded at M=128 electrodes placed
    according to the extended 10–20 system. Data were recorded at 500 Hz with
    electrode Cz as reference. Four BrainAmp amplifiers were used for this
    purpose, using a temporal analog high-pass filter with a time constant of
    10 s. The data were re-referenced to common average reference
    offline. Electrode impedances were below 10 kΩ for all electrodes and
    subjects at the beginning of each recording session. No trials were
    rejected and no artifact correction was performed. For each subject, the
    locations of the 128 electrodes were measured in three dimensions using a
    Zebris ultrasound tracking system and stored for further offline analysis.


    References
    ----------
    .. [1] Grosse-Wentrup, Moritz, et al. "Beamforming in noninvasive
           brain–computer interfaces." IEEE Transactions on Biomedical
           Engineering 56.4 (2009): 1209-1219.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=dict(right_hand=2, left_hand=1),
            code="Grosse-Wentrup 2009",
            interval=[0, 7],
            paradigm="imagery",
            doi="10.1109/TBME.2008.2009768",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        raw = mne.io.read_raw_eeglab(
            self.data_path(subject), preload=True, verbose="ERROR"
        )
        stim = raw.annotations.description.astype(np.dtype("<10U"))

        stim[stim == "20"] = "right_hand"
        stim[stim == "10"] = "left_hand"
        raw.annotations.description = stim
        return {"session_0": {"run_0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # download .set
        _set = "{:s}subject{:d}.set".format(DOWNLOAD_URL, subject)
        set_local = dl.data_dl(_set, "MUNICHMI", path, force_update, verbose)
        # download .fdt
        _fdt = "{:s}subject{:d}.fdt".format(DOWNLOAD_URL, subject)
        dl.data_dl(_fdt, "MUNICHMI", path, force_update, verbose)
        return set_local
