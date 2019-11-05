"""
Physionet Motor imagery dataset.
"""

from .base import BaseDataset
from mne.io import read_raw_edf
import mne
from mne.datasets import eegbci

BASE_URL = 'http://www.physionet.org/pn4/eegmmidb/'


class PhysionetMI(BaseDataset):
    """Physionet Motor Imagery dataset.

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org).
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

    parameters
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
            code='Physionet Motor Imagery',
            # website does not specify how long the trials are, but the
            # interval between 2 trial is 4 second.
            interval=[0, 3],
            paradigm='imagery',
            doi='10.1109/TBME.2004.827072')

        self.feet_runs = []
        self.hand_runs = []

        if imagined:
            self.feet_runs += [6, 10, 14]
            self.hand_runs += [4, 8, 12]

        if executed:
            self.feet_runs += [5, 9, 13]
            self.hand_runs += [3, 7, 11]

    def _load_one_run(self, subject, run, preload=True):
        raw_fname = eegbci.load_data(subject, runs=[run], verbose='ERROR',
                                     base_url=BASE_URL)[0]
        raw = read_raw_edf(raw_fname, preload=preload, verbose='ERROR')
        raw.rename_channels(lambda x: x.strip('.'))
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
        return raw

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        data = {}

        # baseline runs
        data['baseline_eye_open'] = self._load_one_run(subject, 1)
        data['baseline_eye_closed'] = self._load_one_run(subject, 2)

        # hand runs
        for run in self.hand_runs:
            data['run_%d' % run] = self._load_one_run(subject, run)

        # feet runs
        for run in self.feet_runs:
            raw = self._load_one_run(subject, run)

            # modify stim channels to match new event ids. for feets runs,
            # hand = 2 modified to 4, and feet = 3, modified to 5
            stim = raw._data[-1]
            raw._data[-1, stim == 2] = 4
            raw._data[-1, stim == 3] = 5
            data['run_%d' % run] = raw

        return {"session_0": data}

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        paths = eegbci.load_data(subject,
                                 runs=[1, 2] + self.hand_runs + self.feet_runs,
                                 verbose=verbose)
        return paths
