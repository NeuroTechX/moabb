"""Steady-State Visually Evoked Paradigms"""

import mne
import numpy as np
import pandas as pd
import logging

from moabb.paradigms.base import BaseParadigm
from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset

log = logging.getLogger()


class BaseSSVEP(BaseParadigm):
    """Base SSVEP Paradigm

    Parameters
    ----------
    filters: list of list | None (default [7, 45])
        Bank of bandpass filter to apply. 

    events: list of str | None (default None)
        List of stimulation frequencies. If None, use all stimulus
        found in the dataset.

    n_classes: int or None (default 2)
        Number of classes each dataset must have. All dataset classes if None

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
        List of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    """
    def __init__(self, filters=[7, 45], events=None, n_classes=2, tmin=0.0,
                 tmax=None, channels=None, resample=None):
        super().__init__()
        self.filters = filters
        self.events = events
        self.n_classes = n_classes
        self.channels = channels
        self.resample = resample

        if tmax is not None and  tmin >= tmax:
            raise(ValueError("tmax must be greater than tmin"))
        self.tmin = tmin
        self.tmax = tmax
        
        if self.events is None:
            log.warning("Choosing the first " + str(n_classes) + " classes"
                        + "from all possible events")
        else:
            assert n_classes <= len(
                self.events), 'More classes than events specified'
        

    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == 'ssvep'):
            ret = False
            
        if self.events is None and self.n_classes is None:
            return ret

        # if self.events is None and self.n_classes is not None:
        #     if not len(dataset.event_id) >= self.n_classes:
        #         ret = False
        # else:
        #     overlap = len(set(self.events) & set(dataset.event_id.keys()))
        #     if not overlap >= self.n_classes:
        #         ret = False
                
        return ret

    def used_events(self, dataset):
        out = {}
        if self.events is None:
            for k, v in dataset.event_id.items():
                out[k] = v
                if self.n_classes and len(out) == self.n_classes:
                    break
        else:
            for event in self.events:
                if event in dataset.event_id.keys():
                    out[event] = dataset.event_id[event]
                if self.n_classes and len(out) == self.n_classes:
                    break
        if self.n_classes and len(out) < self.n_classes:
            raise(ValueError(f"Dataset {dataset.code} did not have enough "
                             f"freqs in {self.events} to run analysis"))
        return out

    def process_raw(self, raw, dataset):
        # find the events
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        channels = () if self.channels is None else self.channels

        # picks channels
        picks = mne.pick_types(raw.info, eeg=True, stim=False,
                               include=channels)

        # get events id
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

        # get filters
        if self.filters is None:
            self.filters = [[float(f) - 0.5, float(f) + 0.5]
                            for f in event_id.keys()
                            if f.replace('.', '', 1).isnumeric()]

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
        return utils.dataset_search(paradigm='ssvep',
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


class SSVEP(BaseSSVEP):
    """Single bandpass filter SSVEP

    SSVEP paradigm with only one bandpass filter (default 7 to 45 Hz)
    Metric is 'roc-auc' if 2 classes and 'accuracy' if more

    Parameters
    ----------
    fmin: float (default 7)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 45)
        cutoff frequency (Hz) for the low pass filter

    events: list of str | None (default None)
        List of stimulation frequencies. If None, use all stimulus
        found in the dataset.

    n_classes: int or None (default 2)
        Number of classes each dataset must have. All dataset classes if None

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
        List of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    """
    def __init__(self, fmin=7, fmax=45, **kwargs):
        if 'filters' in kwargs.keys():
            raise(ValueError("MotorImagery does not take argument filters"))
        super().__init__(filters=[(fmin, fmax)], **kwargs)
        

class FilterBankSSVEP(BaseSSVEP):
    """ Filtered bank n-class SSVEP paradigm

    SSVEP paradigm with multiple narrow bandpass filters, centered around the
    frequencies of considered events. 
    Metric is 'roc-auc' if 2 classes and 'accuracy' if more.

    Parameters
    -----------

    filters: list of list | None (default None)
        If None, bandpass set around freqs of events with [f_n-0.5, f_n+0.5]

    events: List of str,
        List of stimulation frequencies. If None, use all stimulus
        found in the dataset.

    n_classes: int or None (default 2)
        Number of classes each dataset must have. All dataset classes if None

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
        List of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    """
    def __init__(self, filters=None, **kwargs):
        super().__init__(filters=filters, **kwargs)


class FakeSSVEPParadigm(BaseSSVEP):
    """Fake SSVEP classification.
    """

    @property
    def datasets(self):
        return [FakeDataset(events=['13', '15'], paradigm='ssvep')]
