import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_gdf

from moabb.datasets.base import BaseDataset

from . import download as dl


UPPER_LIMB_URL = "https://zenodo.org/record/834976/files/"


class Ofner2017(BaseDataset):
    """Motor Imagery ataset from Ofner et al 2017.

    Upper limb Motor imagery dataset from the paper [1]_.

    **Dataset description**

    We recruited 15 healthy subjects aged between 22 and 40 years with a mean
    age of 27 years (standard deviation 5 years). Nine subjects were female,
    and all the subjects except s1 were right-handed.

    We measured each subject in two sessions on two different days, which were
    not separated by more than one week. In the first session the subjects
    performed ME, and MI in the second session. The subjects performed six
    movement types which were the same in both sessions and comprised of
    elbow flexion/extension, forearm supination/pronation and hand open/close;
    all with the right upper limb. All movements started at a
    neutral position: the hand half open, the lower arm extended to 120
    degree and in a neutral rotation, i.e. thumb on the inner side.
    Additionally to the movement classes, a rest class was recorded in which
    subjects were instructed to avoid any movement and to stay in the starting
    position. In the ME session, we instructed subjects to execute sustained
    movements. In the MI session, we asked subjects to perform kinesthetic MI
    of the movements done in the ME session (subjects performed one ME run
    immediately before the MI session to support kinesthetic MI).

    The paradigm was trial-based and cues were displayed on a computer screen
    in front of the subjects, Fig 2 shows the sequence of the paradigm.
    At second 0, a beep sounded and a cross popped up on the computer screen
    (subjects were instructed to fixate their gaze on the cross). Afterwards,
    at second 2, a cue was presented on the computer screen, indicating the
    required task (one out of six movements or rest) to the subjects. At the
    end of the trial, subjects moved back to the starting position. In every
    session, we recorded 10 runs with 42 trials per run. We presented 6
    movement classes and a rest class and recorded 60 trials per class in a
    session.

    References
    ----------
    .. [1] Ofner, P., Schwarz, A., Pereira, J. and Müller-Putz, G.R., 2017.
           Upper limb movements can be decoded from the time-domain of
           low-frequency EEG. PloS one, 12(8), p.e0182578.
           https://doi.org/10.1371/journal.pone.0182578

    """

    def __init__(self, imagined=True, executed=False):
        self.imagined = imagined
        self.executed = executed
        event_id = {
            "right_elbow_flexion": 1536,
            "right_elbow_extension": 1537,
            "right_supination": 1538,
            "right_pronation": 1539,
            "right_hand_close": 1540,
            "right_hand_open": 1541,
            "rest": 1542,
        }

        n_sessions = int(imagined) + int(executed)
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=n_sessions,
            events=event_id,
            code="Ofner2017",
            interval=[0, 3],  # according to paper 2-5
            paradigm="imagery",
            doi="10.1371/journal.pone.0182578",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        sessions = []
        if self.imagined:
            sessions.append("imagination")

        if self.executed:
            sessions.append("execution")

        out = {}
        for session in sessions:
            paths = self.data_path(subject, session=session)

            eog = ["eog-l", "eog-m", "eog-r"]
            montage = make_standard_montage("standard_1005")
            data = {}
            for ii, path in enumerate(paths):
                raw = read_raw_gdf(
                    path, eog=eog, misc=range(64, 96), preload=True, verbose="ERROR"
                )
                raw.set_montage(montage)
                # there is nan in the data
                raw._data[np.isnan(raw._data)] = 0
                # Modify the annotations to match the name of the command
                stim = raw.annotations.description.astype(np.dtype("<21U"))
                stim[stim == "1536"] = "right_elbow_flexion"
                stim[stim == "1537"] = "right_elbow_extension"
                stim[stim == "1538"] = "right_supination"
                stim[stim == "1539"] = "right_pronation"
                stim[stim == "1540"] = "right_hand_close"
                stim[stim == "1541"] = "right_hand_open"
                stim[stim == "1542"] = "rest"
                raw.annotations.description = stim
                data["run_%d" % ii] = raw

            out[session] = data
        return out

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        session=None,
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        paths = []

        if session is None:
            sessions = []
            if self.imagined:
                sessions.append("imagination")

            if self.executed:
                sessions.append("execution")
        else:
            sessions = [session]

        # FIXME check the value are in V and not uV.
        for session in sessions:
            for run in range(1, 11):
                url = (
                    f"{UPPER_LIMB_URL}motor{session}_subject{subject}" + f"_run{run}.gdf"
                )
                p = dl.data_dl(url, "UPPERLIMB", path, force_update, verbose)
                paths.append(p)

        return paths
