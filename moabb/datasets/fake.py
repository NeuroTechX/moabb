import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets.base import BaseDataset


class FakeDataset(BaseDataset):
    """Fake Dataset for test purpose.

    The dataset has 2 sessions, 10 subjects, and 3 classes.

    """

    def __init__(
        self,
        event_list=("fake_c1", "fake_c2", "fake_c3"),
        n_sessions=2,
        n_runs=2,
        n_subjects=10,
        paradigm="imagery",
    ):
        self.n_runs = n_runs
        event_id = {ev: ii + 1 for ii, ev in enumerate(event_list)}
        super().__init__(
            list(range(1, n_subjects + 1)),
            n_sessions,
            event_id,
            "FakeDataset",
            [0, 3],
            paradigm,
        )

    def _get_single_subject_data(self, subject):

        data = dict()
        for session in range(self.n_sessions):
            data[f"session_{session}"] = {
                f"run_{ii}": self._generate_raw() for ii in range(self.n_runs)
            }
        return data

    def _generate_raw(self):

        ch_names = ["C3", "Cz", "C4"]

        montage = make_standard_montage("standard_1005")
        sfreq = 128
        duration = len(self.event_id) * 60
        eeg_data = 2e-5 * np.random.randn(duration * sfreq, len(ch_names))
        y = np.zeros((duration * sfreq))
        for ii, ev in enumerate(self.event_id):
            start_idx = (1 + 5 * ii) * 128
            jump = 5 * len(self.event_id) * 128
            y[start_idx::jump] = self.event_id[ev]

        ch_types = ["eeg"] * len(ch_names) + ["stim"]
        ch_names = ch_names + ["stim"]

        eeg_data = np.c_[eeg_data, y]

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = RawArray(data=eeg_data.T, info=info, verbose=False)
        raw.set_montage(montage)
        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        pass
