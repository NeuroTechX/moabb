from mne import create_info
from mne.io import RawArray
from mne.channels import read_montage
import numpy as np

from moabb.datasets.base import BaseDataset
from moabb.paradigms.motor_imagery import BaseMotorImagery


class FakeDataset(BaseDataset):
    """Fake Dataset for test purpose.

    The dataset has 2 sessions, 10 subjects, and 3 classes.

    """

    def __init__(self, event_list=['fake_c1', 'fake_c2', 'fake_c3']):
        event_id = {ev: ii + 1 for ii, ev in enumerate(event_list)}
        super().__init__(range(1, 3), 2, event_id,
                         'FakeDataset', [1, 3], 'imagery')

    def _get_single_subject_data(self, subject, stack_sessions):
        data = [self._generate_raw(), self._generate_raw()]
        if stack_sessions:
            return [data]
        else:
            return [[data]]

    def _generate_raw(self):

        ch_names = ['C3', 'Cz', 'C4']

        montage = read_montage('standard_1005')
        sfreq = 128
        duration = len(self.event_id)*60
        eeg_data = 2e-5 * np.random.randn(duration * sfreq, len(ch_names))
        y = np.zeros((duration * sfreq))
        for ii, ev in enumerate(self.event_id):
            y[((1 + 5*ii)*128)::(5*len(self.event_id)*128)] = self.event_id[ev]

        ch_types = ['eeg'] * len(ch_names) + ['stim']
        ch_names = ch_names + ['stim']

        eeg_data = np.c_[eeg_data, y]

        info = create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=sfreq, montage=montage)
        raw = RawArray(data=eeg_data.T, info=info, verbose=False)
        return raw


class FakeImageryParadigm(BaseMotorImagery):
    """fake Imagery for left hand/right hand classification

    Metric is 'roc_auc'

    """

    def verify(self, d):
        events = ['left_hand', 'right_hand']
        super().verify(d)
        assert set(events) <= set(d.event_id.keys())
        d.selected_events = dict(
            zip(events, [d.event_id[s] for s in events]))

    @property
    def scoring(self):
        return 'roc_auc'

    @property
    def datasets(self):
        return [FakeDataset(['left_hand', 'right_hand'])]
