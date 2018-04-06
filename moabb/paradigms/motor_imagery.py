"""Motor Imagery Paradigms"""

import abc
import mne
import numpy as np
import pandas as pd
import logging

from moabb.paradigms.base import BaseParadigm
from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset

log = logging.getLogger()


class BaseMotorImagery(BaseParadigm):
    """Base Motor imagery paradigm. Please use one of the child classes

    Parameters
    ----------

    filters: List of list (defaults [[7, 35]])
        bank of filter to apply
    interval: list | None, (default None)
        time interval to epoch trial, shifted to imagery start (e.g. 
        [0,2] becomes [3,5] if imagery starts 3s after the cue). If None, 
        defaults to the dataset-defined interval
    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.
    """

    def __init__(self, filters=[[7, 35]], channels=None, interval=None,
                 events=None, resample=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.channels = channels
        self.events = events
        self.resample = resample

        if interval is not None:
            if not isinstance(interval, list):
                raise(ValueError("interval must be a list"))

            if len(interval) != 2:
                raise(ValueError("interval must be a list of 2 elements"))

            if interval[0] >= interval[1]:
                raise(ValueError("first element of interval must be greater"
                                 "than the second element"))

        self.interval = interval

    def verify(self, dataset):
        assert dataset.paradigm == 'imagery'

        # check if dataset has required events
        if self.events:
            assert set(self.events) <= set(dataset.event_id.keys())

        # we should verify list of channels, somehow

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
        except RuntimeError as r:
            # skip raw if no event found
            return

        # get interval
        if self.interval is None:
            tmin, tmax = dataset.interval
        else:
            tmin, tmax = [t + dataset.interval[0] for t in self.interval]

        if self.resample is not None:
            raw = raw.copy().resample(self.resample)

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
        if self.interval is None:
            interval = None
        else:
            interval = self.interval[1]-self.interval[0]
        return utils.dataset_search(paradigm='imagery',
                                    events=self.events,
                                    interval=interval,
                                    has_all_events=True)

    @property
    def scoring(self):
        return 'accuracy'


class SinglePass(BaseMotorImagery):

    def __init__(self, fmin=8, fmax=32, **kwargs):
        """
        Single-pass MI. Takes arguments fmin and fmax and not filters
        """
        if 'filters' in kwargs.keys():
            raise(ValueError('MotorImagery does not take argument \'filters\''))
        super().__init__(filters=[[fmin, fmax]], **kwargs)


class FilterBankMotorImagery(BaseMotorImagery):
    """Filter Bank MI."""

    def __init__(self, filters=[[8, 12], [12, 16], [16, 20], [20, 24],
                                [24, 28], [28, 32]], **kwargs):
        """init"""
        super().__init__(filters=filters, **kwargs)


class LeftRightImagery(SinglePass):
    """Motor Imagery for left hand/right hand classification

    Metric is 'roc_auc'

    """

    def __init__(self, **kwargs):
        if 'events' in kwargs.keys():
            raise(ValueError('LeftRightImagery dont accept events'))
        super().__init__(events=['left_hand', 'right_hand'], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return 'roc_auc'


class FilterBankLeftRightImagery(FilterBankMotorImagery):
    """Filter Bank Motor Imagery for left hand/right hand classification

    Metric is 'roc_auc'

    """

    def __init__(self, **kwargs):
        if 'events' in kwargs.keys():
            raise(ValueError('LeftRightImagery dont accept events'))
        super().__init__(events=['left_hand', 'right_hand'], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return 'roc_auc'


class FilterBankMotorImagery(FilterBankMotorImagery):
    """
    Filter bank n-class imagery.

    Metric is 'roc-auc' if 2 classes and 'accuracy' if more

    Parameters
    -----------

    events: List of str
        event labels used to filter datasets (e.g. if only motor imagery is
        desired).

    n_classes: int,
        number of classes each dataset must have. If events is given,
        requires all imagery sorts to be within the events list.
    """

    def __init__(self, n_classes=2, **kwargs):
        "docstring"
        super().__init__(**kwargs)
        self.n_classes = n_classes
        assert n_classes <= len(
            self.events), 'More classes than events specified'

        if self.events is None:
            log.warning("Choosing from all possible events")

    def verify(self, dataset):
        assert dataset.paradigm == 'imagery'
        if self.events is None:
            assert len(dataset.event_id) >= self.n_classes
        else:
            overlap = len(set(self.events) & set(dataset.event_id.keys()))
            assert overlap >= self.n_classes

    def used_events(self, dataset):
        out = {}
        if self.events is None:
            for k, v in dataset.event_id.items():
                out[k] = v
                if len(out) == self.n_classes:
                    break
        else:
            for event in self.events:
                if event in dataset.event_id.keys():
                    out[event] = dataset.event_id[event]
                if len(out) == self.n_classes:
                    break
        if len(out) < self.n_classes:
            raise ValueError("Dataset {} did not have enough events in {} to run analysis".format(
                dataset.code, self.events))
        return out

    @property
    def datasets(self):
        if self.interval is None:
            interval = None
        else:
            interval = self.interval[1]-self.interval[0]
        return utils.dataset_search(paradigm='imagery',
                                    events=self.events,
                                    total_classes=self.n_classes,
                                    interval=interval,
                                    has_all_events=False)

    @property
    def scoring(self):
        if self.n_classes == 2:
            return 'roc_auc'
        else:
            return 'accuracy'


class MotorImagery(SinglePass):
    """
    N-class motor imagery.

    Metric is 'roc-auc' if 2 classes and 'accuracy' if more

    Parameters
    -----------

    events: List of str
        event labels used to filter datasets (e.g. if only motor imagery is
        desired).

    n_classes: int,
        number of classes each dataset must have. If events is given,
        requires all imagery sorts to be within the events list.

    """

    def __init__(self, n_classes=2, **kwargs):
        "docstring"
        super().__init__(**kwargs)
        self.n_classes = n_classes
        assert n_classes <= len(
            self.events), 'More classes than events specified'

        if self.events is None:
            log.warning("Choosing from all possible events")

    def verify(self, dataset):
        assert dataset.paradigm == 'imagery'
        if self.events is None:
            assert len(dataset.event_id) >= self.n_classes
        else:
            overlap = len(set(self.events) & set(dataset.event_id.keys()))
            assert overlap >= self.n_classes

    def used_events(self, dataset):
        out = {}
        if self.events is None:
            for k, v in dataset.event_id.items():
                out[k] = v
                if len(out) == self.n_classes:
                    break
        else:
            for event in self.events:
                if event in dataset.event_id.keys():
                    out[event] = dataset.event_id[event]
                if len(out) == self.n_classes:
                    break
        if len(out) < self.n_classes:
            raise ValueError("Dataset {} did not have enough events in {} to run analysis".format(
                dataset.code, self.events))
        return out

    @property
    def datasets(self):
        if self.interval is None:
            interval = None
        else:
            interval = self.interval[1]-self.interval[0]
        return utils.dataset_search(paradigm='imagery',
                                    events=self.events,
                                    total_classes=self.n_classes,
                                    interval=interval,
                                    has_all_events=False)

    @property
    def scoring(self):
        if self.n_classes == 2:
            return 'roc_auc'
        else:
            return 'accuracy'


class FakeImageryParadigm(LeftRightImagery):
    """fake Imagery for left hand/right hand classification
    """

    @property
    def datasets(self):
        return [FakeDataset(['left_hand', 'right_hand'])]
