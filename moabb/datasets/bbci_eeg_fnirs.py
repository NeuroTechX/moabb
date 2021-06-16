"""
BBCI EEG fNIRS Motor imagery dataset.
"""

import os
import os.path as op
import zipfile as z

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.datasets.utils import _get_path
from mne.io import RawArray
from mne.utils import _fetch_file
from scipy.io import loadmat

from .base import BaseDataset


SHIN_URL = "http://doc.ml.tu-berlin.de/hBCI"


def eeg_data_path(base_path, subject, accept):
    datapath = op.join(
        base_path, "EEG", "subject {:02d}".format(subject), "with occular artifact"
    )
    if not op.isfile(op.join(datapath, "cnt.mat")):
        if not op.isdir(op.join(base_path, "EEG")):
            os.makedirs(op.join(base_path, "EEG"))
        intervals = [[1, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 29]]
        for low, high in intervals:
            if subject >= low and subject <= high:
                if not op.isfile(op.join(base_path, "EEG.zip")):
                    if not accept:
                        raise AttributeError(
                            "You must accept licence term to download this dataset,"
                            "set accept=True when instanciating the dataset."
                        )
                    _fetch_file(
                        "{}/EEG/EEG_{:02d}-{:02d}.zip".format(SHIN_URL, low, high),
                        op.join(base_path, "EEG.zip"),
                        print_destination=False,
                    )
                with z.ZipFile(op.join(base_path, "EEG.zip"), "r") as f:
                    f.extractall(op.join(base_path, "EEG"))
                os.remove(op.join(base_path, "EEG.zip"))
                break
    assert op.isfile(op.join(datapath, "cnt.mat")), op.join(datapath, "cnt.mat")
    return [op.join(datapath, fn) for fn in ["cnt.mat", "mrk.mat"]]


def fnirs_data_path(path, subject, accept):
    datapath = op.join(path, "NIRS", "subject {:02d}".format(subject))
    if not op.isfile(op.join(datapath, "mrk.mat")):
        # fNIRS
        if not op.isfile(op.join(path, "fNIRS.zip")):
            if not accept:
                raise AttributeError(
                    "You must accept licence term to download this dataset,"
                    "set accept=True when instanciating the dataset."
                )
            _fetch_file(
                "http://doc.ml.tu-berlin.de/hBCI/NIRS/NIRS_01-29.zip",
                op.join(path, "fNIRS.zip"),
                print_destination=False,
            )
        if not op.isdir(op.join(path, "NIRS")):
            os.makedirs(op.join(path, "NIRS"))
        with z.ZipFile(op.join(path, "fNIRS.zip"), "r") as f:
            f.extractall(op.join(path, "NIRS"))
        os.remove(op.join(path, "fNIRS.zip"))
    return [op.join(datapath, fn) for fn in ["cnt.mat", "mrk.mat"]]


class Shin2017(BaseDataset):
    """Not to be used."""

    def __init__(
        self, fnirs=False, motor_imagery=True, mental_arithmetic=False, accept=False
    ):
        if not any([motor_imagery, mental_arithmetic]):
            raise (
                ValueError(
                    "at least one of motor_imagery or" " mental_arithmetic must be true"
                )
            )
        events = dict()
        paradigms = []
        n_sessions = 0
        if motor_imagery:
            events.update(dict(left_hand=1, right_hand=2))
            paradigms.append("imagery")
            n_sessions += 3

        if mental_arithmetic:
            events.update(dict(substraction=3, rest=4))
            paradigms.append("arithmetic")
            n_sessions += 3

        self.motor_imagery = motor_imagery
        self.mental_arithmetic = mental_arithmetic
        self.accept = accept

        super().__init__(
            subjects=list(range(1, 30)),
            sessions_per_subject=n_sessions,
            events=events,
            code="Shin2017",
            # marker is for *task* start not cue start
            interval=[0, 10],
            paradigm=("/").join(paradigms),
            doi="10.1109/TNSRE.2016.2628057",
        )

        if fnirs:
            raise (NotImplementedError("Fnirs not implemented."))
        self.fnirs = fnirs  # TODO: actually incorporate fNIRS somehow

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        fname, fname_mrk = self.data_path(subject)
        data = loadmat(fname, squeeze_me=True, struct_as_record=False)["cnt"]
        mrk = loadmat(fname_mrk, squeeze_me=True, struct_as_record=False)["mrk"]

        sessions = {}
        # motor imagery
        if self.motor_imagery:
            for ii in [0, 2, 4]:
                session = self._convert_one_session(data, mrk, ii, trig_offset=0)
                sessions["session_%d" % ii] = session

        # arithmetic/rest
        if self.mental_arithmetic:
            for ii in [1, 3, 5]:
                session = self._convert_one_session(data, mrk, ii, trig_offset=2)
                sessions["session_%d" % ii] = session

        return sessions

    def _convert_one_session(self, data, mrk, session, trig_offset=0):
        eeg = data[session].x.T * 1e-6
        trig = np.zeros((1, eeg.shape[1]))
        idx = (mrk[session].time - 1) // 5
        trig[0, idx] = mrk[session].event.desc // 16 + trig_offset
        eeg = np.vstack([eeg, trig])
        ch_names = list(data[session].clab) + ["Stim"]
        ch_types = ["eeg"] * 30 + ["eog"] * 2 + ["stim"]

        montage = make_standard_montage("standard_1005")
        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=200.0)
        raw = RawArray(data=eeg, info=info, verbose=False)
        raw.set_montage(montage)
        return {"run_0": raw}

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        accept=False,
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        if accept:
            self.accept = True

        key = "MNE_DATASETS_BBCIFNIRS_PATH"
        path = _get_path(path, key, "BBCI EEG-fNIRS")
        if not op.isdir(op.join(path, "MNE-eegfnirs-data")):
            os.makedirs(op.join(path, "MNE-eegfnirs-data"))
        if self.fnirs:
            return fnirs_data_path(
                op.join(path, "MNE-eegfnirs-data"), subject, self.accept
            )
        else:
            return eeg_data_path(op.join(path, "MNE-eegfnirs-data"), subject, self.accept)


class Shin2017A(Shin2017):
    """Motor Imagey Dataset from Shin et al 2017

    Dataset from [1]_.

    You should accept the licence term [2]_ to download this dataset, using::
        Shin2017A(accept=True)

    **Data Acquisition**

    EEG and NIRS data was collected in an ordinary bright room. EEG data was
    recorded by a multichannel BrainAmp EEG amplifier with thirty active
    electrodes (Brain Products GmbH, Gilching, Germany) with linked mastoids
    reference at 1000 Hz sampling rate. The EEG amplifier was also used to
    measure the electrooculogram (EOG), electrocardiogram (ECG) and respiration
    with a piezo based breathing belt. Thirty EEG electrodes were placed on a
    custom-made stretchy fabric cap (EASYCAP GmbH, Herrsching am Ammersee,
    Germany) and placed according to the international 10-5 system (AFp1, AFp2,
    AFF1h, AFF2h, AFF5h, AFF6h, F3, F4, F7, F8, FCC3h, FCC4h, FCC5h, FCC6h, T7,
    T8, Cz, CCP3h, CCP4h, CCP5h, CCP6h, Pz, P3, P4, P7, P8, PPO1h, PPO2h, POO1,
    POO2 and Fz for ground electrode).

    NIRS data was collected by NIRScout (NIRx GmbH, Berlin, Germany) at 12.5 Hz
    sampling rate. Each adjacent source-detector pair creates one physiological
    NIRS channel. Fourteen sources and sixteen detectors resulting in
    thirty-six
    physiological channels were placed at frontal (nine channels around Fp1,
    Fp2, and Fpz), motor (twelve channels around C3 and C4, respectively) and
    visual areas (three channels around Oz). The inter-optode distance was 30
    mm. NIRS optodes were fixed on the same cap as the EEG electrodes. Ambient
    lights were sufficiently blocked by a firm contact between NIRS optodes and
    scalp and use of an opaque cap.

    EOG was recorded using two vertical (above and below left eye) and two
    horizontal (outer canthus of each eye) electrodes. ECG was recorded based
    on
    Einthoven triangle derivations I and II, and respiration was measured using
    a respiration belt on the lower chest. EOG, ECG and respiration were
    sampled
    at the same sampling rate of the EEG. ECG and respiration data were not
    analyzed in this study, but are provided along with the other signals.

    **Experimental Procedure**

    The subjects sat on a comfortable armchair in front of a 50-inch white
    screen. The distance between their heads and the screen was 1.6 m. They
    were
    asked not to move any part of the body during the data recording. The
    experiment consisted of three sessions of left and right hand MI (dataset
    A)and MA and baseline tasks (taking a rest without any thought) (dataset B)
    each. Each session comprised a 1 min pre-experiment resting period, 20
    repetitions of the given task and a 1 min post-experiment resting
    period. The task started with 2 s of a visual introduction of the task,
    followed by 10 s of a task period and resting period which was given
    randomly from 15 to 17 s. At the beginning and end of the task period, a
    short beep (250 ms) was played. All instructions were displayed on the
    white
    screen by a video projector. MI and MA tasks were performed in separate
    sessions but in alternating order (i.e., sessions 1, 3 and 5 for MI
    (dataset
    A) and sessions 2, 4 and 6 for MA (dataset B)). Fig. 2 shows the schematic
    diagram of the experimental paradigm. Five sorts of motion artifacts
    induced
    by eye and head movements (dataset C) were measured. The motion artifacts
    were recorded after all MI and MA task recordings. The experiment did not
    include the pre- and post-experiment resting state periods.

    **Motor Imagery (Dataset A)**

    For motor imagery, subjects were instructed to perform haptic motor imagery
    (i.e. to imagine the feeling of opening and closing their hands as they
    were
    grabbing a ball) to ensure that actual motor imagery, not visual imagery,
    was performed. All subjects were naive to the MI experiment. For the visual
    instruction, a black arrow pointing to either the left or right side
    appeared at the center of the screen for 2 s. The arrow disappeared with a
    short beep sound and then a black fixation cross was displayed during the
    task period. The subjects were asked to imagine hand gripping (opening and
    closing their hands) in a 1 Hz pace. This pace was shown to and repeated by
    the subjects by performing real hand gripping before the experiment. Motor
    imagery was performed continuously over the task period. The task period
    was finished with a short beep sound and a 'STOP' displayed for 1s on the
    screen. The fixation cross was displayed again during the rest period and
    the subjects were asked to gaze at it to minimize their eye movements. This
    process was repeated twenty times in a single session (ten trials per
    condition in a single session; thirty trials in the whole sessions). In a
    single session, motor imagery tasks were performed on the basis of ten
    subsequent blocks randomly consisting of one of two conditions: Either
    first left and then right hand motor imagery or vice versa.

    References
    ----------

    .. [1] Shin, J., von Lühmann, A., Blankertz, B., Kim, D.W., Jeong, J.,
           Hwang, H.J. and Müller, K.R., 2017. Open access dataset for EEG+NIRS
           single-trial classification. IEEE Transactions on Neural Systems
           and Rehabilitation Engineering, 25(10), pp.1735-1745.
    .. [2] GNU General Public License, Version 3
           `<https://www.gnu.org/licenses/gpl-3.0.txt>`_
    """

    def __init__(self, accept=False):
        super().__init__(
            fnirs=False, motor_imagery=True, mental_arithmetic=False, accept=accept
        )
        self.code = "Shin2017A"


class Shin2017B(Shin2017):
    """Mental Arithmetic Dataset from Shin et al 2017

    Dataset from [1]_.

    You should accept the licence term [2]_ to download this dataset, using::
        Shin2017A(accept=True)

    **Data Acquisition**

    EEG and NIRS data was collected in an ordinary bright room. EEG data was
    recorded by a multichannel BrainAmp EEG amplifier with thirty active
    electrodes (Brain Products GmbH, Gilching, Germany) with linked mastoids
    reference at 1000 Hz sampling rate. The EEG amplifier was also used to
    measure the electrooculogram (EOG), electrocardiogram (ECG) and respiration
    with a piezo based breathing belt. Thirty EEG electrodes were placed on a
    custom-made stretchy fabric cap (EASYCAP GmbH, Herrsching am Ammersee,
    Germany) and placed according to the international 10-5 system (AFp1, AFp2,
    AFF1h, AFF2h, AFF5h, AFF6h, F3, F4, F7, F8, FCC3h, FCC4h, FCC5h, FCC6h, T7,
    T8, Cz, CCP3h, CCP4h, CCP5h, CCP6h, Pz, P3, P4, P7, P8, PPO1h, PPO2h, POO1,
    POO2 and Fz for ground electrode).

    NIRS data was collected by NIRScout (NIRx GmbH, Berlin, Germany) at 12.5 Hz
    sampling rate. Each adjacent source-detector pair creates one physiological
    NIRS channel. Fourteen sources and sixteen detectors resulting in
    thirty-six
    physiological channels were placed at frontal (nine channels around Fp1,
    Fp2, and Fpz), motor (twelve channels around C3 and C4, respectively) and
    visual areas (three channels around Oz). The inter-optode distance was 30
    mm. NIRS optodes were fixed on the same cap as the EEG electrodes. Ambient
    lights were sufficiently blocked by a firm contact between NIRS optodes and
    scalp and use of an opaque cap.

    EOG was recorded using two vertical (above and below left eye) and two
    horizontal (outer canthus of each eye) electrodes. ECG was recorded based
    on
    Einthoven triangle derivations I and II, and respiration was measured using
    a respiration belt on the lower chest. EOG, ECG and respiration were
    sampled
    at the same sampling rate of the EEG. ECG and respiration data were not
    analyzed in this study, but are provided along with the other signals.

    **Experimental Procedure**

    The subjects sat on a comfortable armchair in front of a 50-inch white
    screen. The distance between their heads and the screen was 1.6 m. They
    were
    asked not to move any part of the body during the data recording. The
    experiment consisted of three sessions of left and right hand MI (dataset
    A)and MA and baseline tasks (taking a rest without any thought) (dataset B)
    each. Each session comprised a 1 min pre-experiment resting period, 20
    repetitions of the given task and a 1 min post-experiment resting
    period. The task started with 2 s of a visual introduction of the task,
    followed by 10 s of a task period and resting period which was given
    randomly from 15 to 17 s. At the beginning and end of the task period, a
    short beep (250 ms) was played. All instructions were displayed on the
    white
    screen by a video projector. MI and MA tasks were performed in separate
    sessions but in alternating order (i.e., sessions 1, 3 and 5 for MI
    (dataset
    A) and sessions 2, 4 and 6 for MA (dataset B)). Fig. 2 shows the schematic
    diagram of the experimental paradigm. Five sorts of motion artifacts
    induced
    by eye and head movements (dataset C) were measured. The motion artifacts
    were recorded after all MI and MA task recordings. The experiment did not
    include the pre- and post-experiment resting state periods.

    **Mental Arithmetic (Dataset B)**

    For the visual instruction of the MA task, an initial subtraction such as
    'three-digit number minus one-digit number' (e.g., 384-8) appeared at the
    center of the screen for 2 s. The subjects were instructed to memorize the
    numbers while the initial subtraction was displayed on the screen. The
    initial subtraction disappeared with a short beep sound and a black
    fixation cross was displayed during the task period in which the subjects
    were asked
    to repeatedly perform to subtract the one-digit number from the result of
    the previous subtraction. For the baseline task, no specific sign but the
    black fixation cross was displayed on the screen, and the subjects were
    instructed to take a rest. Note that there were other rest periods between
    the MA and baseline task periods, as same with the MI paradigm. Both task
    periods were finished with a short beep sound and a 'STOP' displayed for
    1 s on the screen. The fixation cross was displayed again during the rest
    period. MA and baseline trials were randomized in the same way as MI.

    References
    ----------
    .. [1] Shin, J., von Lühmann, A., Blankertz, B., Kim, D.W., Jeong, J.,
           Hwang, H.J. and Müller, K.R., 2017. Open access dataset for EEG+NIRS
           single-trial classification. IEEE Transactions on Neural Systems
           and Rehabilitation Engineering, 25(10), pp.1735-1745.
    .. [2] GNU General Public License, Version 3
           `<https://www.gnu.org/licenses/gpl-3.0.txt>`_
    """

    def __init__(self, accept=False):
        super().__init__(
            fnirs=False, motor_imagery=False, mental_arithmetic=True, accept=accept
        )
        self.code = "Shin2017B"
