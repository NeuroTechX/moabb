"""
SSVEP Exoskeleton dataset.
"""

from mne.io import Raw

from . import download as dl
from .base import BaseDataset


SSVEPEXO_URL = "https://zenodo.org/record/2392979/files/"


class SSVEPExo(BaseDataset):
    """SSVEP Exo dataset

    SSVEP dataset from E. Kalunga PhD in University of Versailles [1]_.

    The datasets contains recording from 12 male and female subjects aged
    between 20 and 28 years. Informed consent was obtained from all subjects,
    each one has signed a form attesting her or his consent. The subject sits
    in an electric wheelchair, his right upper limb is resting on the
    exoskeleton. The exoskeleton is functional but is not used during the
    recording of this experiment.

    A panel of size 20x30 cm is attached on the left side of the chair, with
    3 groups of 4 LEDs blinking at different frequencies. Even if the panel
    is on the left side, the user could see it without moving its head. The
    subjects were asked to sit comfortably in the wheelchair and to follow the
    auditory instructions, they could move and blink freely.

    A sequence of trials is proposed to the user. A trial begin by an audio cue
    indicating which LED to focus on, or to focus on a fixation point set at an
    equal distance from all LEDs for the reject class. A trial lasts 5 seconds
    and there is a 3 second pause between each trial. The evaluation is
    conducted during a session consisting of 32 trials, with 8 trials for each
    frequency (13Hz, 17Hz and 21Hz) and 8 trials for the reject class, i.e.
    when the subject is not focusing on any specific blinking LED.

    There is between 2 and 5 sessions for each user, recorded on different
    days, by the same operators, on the same hardware and in the same
    conditions.

    references
    ----------
    .. [1] Emmanuel K. Kalunga, Sylvain Chevallier, Quentin Barthelemy. "Online
           SSVEP-based BCI using Riemannian Geometry". Neurocomputing, 2016.
           arXiv report: https://arxiv.org/abs/1501.03227
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events={"13": 2, "17": 3, "21": 4, "rest": 1},
            code="SSVEP Exoskeleton",
            interval=[2, 4],
            paradigm="ssvep",
            doi="10.1016/j.neucom.2016.01.007",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject"""

        out = {}
        paths = self.data_path(subject, update_path=True, verbose=False)
        for ii, path in enumerate(paths):
            raw = Raw(path, preload=True, verbose="ERROR")
            out["run_%d" % ii] = raw
        return {"session_0": out}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):

        runs = {s + 1: n for s, n in enumerate([2] * 6 + [3] + [2] * 2 + [4, 2, 5])}

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        paths = []
        for run in range(runs[subject]):
            url = "{:s}subject{:02d}_run{:d}_raw.fif".format(
                SSVEPEXO_URL, subject, run + 1
            )
            p = dl.data_dl(url, "SSVEPEXO", path, force_update, verbose)
            paths.append(p)
        return paths
