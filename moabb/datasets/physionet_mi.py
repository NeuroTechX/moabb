"""
Physionet Motor imagery dataset.
"""

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import read_raw_edf

from moabb.datasets.base import BaseDataset
from moabb.datasets.download import get_dataset_path


BASE_URL = "http://archive.physionet.org/pn4/eegmmidb/"


class PhysionetMI(BaseDataset):
    """Physionet Motor Imagery dataset.

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers [2]_.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org) [1]_.
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
       The subject opens and closes the corresponding fist until the target
       disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
       The subject imagines opening and closing the corresponding fist until
       the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
       The subject opens and closes either both fists (if the target is on top)
       or both feet (if the target is on the bottom) until the target
       disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
       The subject imagines opening and closing either both fists
       (if the target is on top) or both feet (if the target is on the bottom)
       until the target disappears. Then the subject relaxes.

    Parameters
    ----------

    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.

    executed: bool (default False)
        if True, return runs corresponding to motor execution.

    references
    ----------

    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
           Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
           interface (BCI) system. IEEE Transactions on biomedical engineering,
           51(6), pp.1034-1043.

    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
           P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
           H.E. and PhysioBank, P., PhysioNet: components of a new research
           resource for complex physiologic signals Circulation 2000 Volume
           101 Issue 23 pp. E215–E220.
    """

    def __init__(self, imagined=True, executed=False):
        super().__init__(
            subjects=list(range(1, 110)),
            sessions_per_subject=1,
            events=dict(left_hand=2, right_hand=3, feet=5, hands=4, rest=1),
            code="Physionet Motor Imagery",
            # website does not specify how long the trials are, but the
            # interval between 2 trial is 4 second.
            interval=[0, 3],
            paradigm="imagery",
            doi="10.1109/TBME.2004.827072",
        )

        self.feet_runs = []
        self.hand_runs = []

        if imagined:
            self.feet_runs += [6, 10, 14]
            self.hand_runs += [4, 8, 12]

        if executed:
            self.feet_runs += [5, 9, 13]
            self.hand_runs += [3, 7, 11]

    def _load_one_run(self, subject, run, preload=True):
        raw_fname = eegbci.load_data(
            subject, runs=[run], verbose="ERROR", base_url=BASE_URL
        )[0]
        raw = read_raw_edf(raw_fname, preload=preload, verbose="ERROR")
        raw.rename_channels(lambda x: x.strip("."))
        raw.rename_channels(lambda x: x.upper())
        # fmt: off
        renames = {
            "AFZ": "AFz", "PZ": "Pz", "FPZ": "Fpz", "FCZ": "FCz", "FP1": "Fp1", "CZ": "Cz",
            "OZ": "Oz", "POZ": "POz", "IZ": "Iz", "CPZ": "CPz", "FP2": "Fp2", "FZ": "Fz",
        }
        # fmt: on
        raw.rename_channels(renames)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        data = {}
        sign = "EEGBCI"
        get_dataset_path(sign, None)

        # hand runs
        for run in self.hand_runs:
            raw = self._load_one_run(subject, run)
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "left_hand"
            stim[stim == "T2"] = "right_hand"
            raw.annotations.description = stim
            data["run_%d" % run] = raw

        # feet runs
        for run in self.feet_runs:
            raw = self._load_one_run(subject, run)
            # modify stim channels to match new event ids. for feets runs,
            # hand = 2 modified to 4, and feet = 3, modified to 5
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "hands"
            stim[stim == "T2"] = "feet"
            raw.annotations.description = stim
            data["run_%d" % run] = raw

        return {"session_0": data}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sign = "EEGBCI"
        get_dataset_path(sign, None)
        paths = eegbci.load_data(
            subject, runs=[1, 2] + self.hand_runs + self.feet_runs, verbose=verbose
        )
        return paths
