import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets.base import BaseDataset
from moabb.datasets.braininvaders import VirtualReality


class FakeDataset(BaseDataset):
    """Fake Dataset for test purpose.

    By default, the dataset has 2 sessions, 10 subjects, and 3 classes.

    Parameters
    ----------
    event_list: list or tuple of str
        List of event to generate, default: ("fake_c1", "fake_c2", "fake_c3")
    n_sessions: int, default 2
        Number of session to generate
    n_runs: int, default 2
        Number of runs to generate
    n_subjects: int, default 10
        Number of subject to generate
    paradigm: ['p300','imagery', 'ssvep']
        Defines what sort of dataset this is
    channels: list or tuple of str
        List of channels to generate, default ("C3", "Cz", "C4")

        .. versionadded:: 0.4.3
    """

    def __init__(
        self,
        event_list=("fake_c1", "fake_c2", "fake_c3"),
        n_sessions=2,
        n_runs=2,
        n_subjects=10,
        code="FakeDataset",
        paradigm="imagery",
        channels=("C3", "Cz", "C4"),
    ):
        self.n_runs = n_runs
        event_id = {ev: ii + 1 for ii, ev in enumerate(event_list)}
        self.channels = channels
        super().__init__(
            subjects=list(range(1, n_subjects + 1)),
            sessions_per_subject=n_sessions,
            events=event_id,
            code=code,
            interval=[0, 3],
            paradigm=paradigm,
        )

    def _get_single_subject_data(self, subject):
        data = dict()
        for session in range(self.n_sessions):
            data[f"session_{session}"] = {
                f"run_{ii}": self._generate_raw() for ii in range(self.n_runs)
            }
        return data

    def _generate_raw(self):
        montage = make_standard_montage("standard_1005")
        sfreq = 128
        duration = len(self.event_id) * 60
        eeg_data = 2e-5 * np.random.randn(duration * sfreq, len(self.channels))
        y = np.zeros((duration * sfreq))
        for ii, ev in enumerate(self.event_id):
            start_idx = (1 + 5 * ii) * 128
            jump = 5 * len(self.event_id) * 128
            y[start_idx::jump] = self.event_id[ev]

        ch_types = ["eeg"] * len(self.channels) + ["stim"]
        ch_names = list(self.channels) + ["stim"]

        eeg_data = np.c_[eeg_data, y]

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = RawArray(data=eeg_data.T, info=info, verbose=False)
        raw.set_montage(montage)
        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        pass


class FakeVirtualRealityDataset(FakeDataset):
    """Fake VirtualReality dataset for test purpose.

    .. versionadded:: 0.5.0
    """

    def __init__(self):
        self.n_blocks = 5
        self.n_repetitions = 12
        super().__init__(
            n_sessions=1,
            n_runs=self.n_blocks * self.n_repetitions,
            n_subjects=21,
            code="FakeVirtualRealityDataset",
            event_list=dict(Target=2, NonTarget=1),
            paradigm="p300",
        )

    def _get_single_subject_data(self, subject):
        data = dict()
        for session in range(self.n_sessions):
            data[f"{session}"] = {}
            for block in range(self.n_blocks):
                for repetition in range(self.n_repetitions):
                    data[f"{session}"][
                        f"block_{block}-repetition_{repetition}"
                    ] = self._generate_raw()
        return data

    def get_block_repetition(self, paradigm, subjects, block_list, repetition_list):
        """Select data for all provided subjects, blocks and repetitions.
        Each subject has 5 blocks of 12 repetitions.

        The returned data is a dictionary with the folowing structure::

            data = {'subject_id' :
                        {'session_id':
                            {'run_id': raw}
                        }
                    }

        See also
        --------
        BaseDataset.get_data
        VirtualReality.get_block_repetition

        Parameters
        ----------
        subjects: List of int
            List of subject number
        block_list: List of int
            List of block number (from 1 to 5)
        repetition_list: List of int
            List of repetition number inside a block (from 1 to 12)

        Returns
        -------
        data: Dict
            dict containing the raw data
        """
        return VirtualReality.get_block_repetition(
            self, paradigm, subjects, block_list, repetition_list
        )
