"""P300 Paradigms"""

import abc
import mne
import numpy as np
import pandas as pd
import logging

from moabb.paradigms.base import BaseParadigm
from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset

log = logging.getLogger()


class BaseP300(BaseParadigm):
    """Base P300 paradigm.

    Please use one of the child classes

    Parameters
    ----------

    filters: list of list (defaults [[7, 35]])
        bank of bandpass filter to apply.

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the begining of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the begining of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the begining of the task as defined in the dataset. If
        None, use the dataset value.

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    """

    def __init__(self, filters=([1, 24],), events=None, tmin=0.0, tmax=None,
                 channels=None, resample=None):
        super().__init__()
        self.filters = filters
        self.channels = channels
        self.events = events
        self.resample = resample

        if (tmax is not None):
            if tmin >= tmax:
                raise(ValueError("tmax must be greater than tmin"))

        self.tmin = tmin
        self.tmax = tmax

    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == 'p300'):
            ret = False

        # check if dataset has required events
        if self.events:
            if not set(self.events) <= set(dataset.event_id.keys()):
                ret = False

        # we should verify list of channels, somehow
        return ret

    @abc.abstractmethod
    def used_events(self, dataset):
        pass

    def process_raw(self, raw, dataset):
        # find the events
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        channels = () if self.channels is None else self.channels

        # picks channels
        picks = mne.pick_types(raw.info, eeg=True, stim=False,
                               include=channels)

        # get event id
        event_id = self.used_events(dataset)

        # pick events, based on event_id
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        # get interval
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter data
            raw_f = raw.copy().filter(fmin, fmax, method='iir',
                                      picks=picks, verbose=False)
            # epoch data
            epochs = mne.Epochs(raw_f, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, proj=False,
                                baseline=None, preload=True,
                                verbose=False, picks=picks,
                                on_missing='ignore')
            if self.resample is not None:
                epochs = epochs.resample(self.resample)
            # MNE is in V, rescale to have uV
            X.append(1e6 * epochs.get_data())

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])

        # if only one band, return a 3D array, otherwise return a 4D
        if len(self.filters) == 1:
            X = X[0]
        else:
            X = np.array(X).transpose((1, 2, 3, 0))

        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata

    @property
    def datasets(self):
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(paradigm='p300',
                                    events=self.events,
                                    interval=interval,
                                    has_all_events=True)

    @property
    def scoring(self):
        return 'roc_auc'


class SinglePass(BaseP300):
    """Single Bandpass filter P300

    P300 paradigm with only one bandpass filter (default 1 to 24 Hz)

    Parameters
    ----------
    fmin: float (default 1)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 24)
        cutoff frequency (Hz) for the low pass filter

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the begining of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the begining of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the begining of the task as defined in the dataset. If
        None, use the dataset value.

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.

    """
    def __init__(self, fmin=1, fmax=24, **kwargs):
        if 'filters' in kwargs.keys():
            raise(ValueError("P300 does not take argument filters"))
        super().__init__(filters=[[fmin, fmax]], **kwargs)


class P300(SinglePass):
    """P300 for Target/NonTarget classification

    Metric is 'roc_auc'

    """

    def __init__(self, **kwargs):
        if 'events' in kwargs.keys():
            raise(ValueError('P300 dont accept events'))
        super().__init__(events=['Target', 'NonTarget'], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return 'roc_auc'


class FakeP300Paradigm(P300):
    """Fake P300 for Target/NonTarget classification.
    """

    @property
    def datasets(self):
        return [FakeDataset(['Target', 'NonTarget'], paradigm='p300')]
