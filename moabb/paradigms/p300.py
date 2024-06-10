"""P300 Paradigms."""

import logging

from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset
from moabb.datasets.preprocessing import RawToEventsP300
from moabb.paradigms.base import BaseParadigm


log = logging.getLogger(__name__)


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
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline
            correction. If None, do not apply baseline correction.
            If a tuple (a, b), the interval is between a and b (in seconds),
            including the endpoints.
            Correction is applied by computing the mean of the baseline period
            and subtracting it from the data (see mne.Epochs)

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    """

    def __init__(
        self,
        filters=([1, 24],),
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
    ):
        super().__init__(
            filters=filters,
            events=events,
            channels=channels,
            baseline=baseline,
            resample=resample,
            tmin=tmin,
            tmax=tmax,
        )

    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == "p300"):
            ret = False

        # check if dataset has required events
        if self.events:
            if not set(self.events) <= set(dataset.event_id.keys()):
                ret = False

        # we should verify list of channels, somehow
        return ret

    def _get_events_pipeline(self, dataset):
        event_id = self.used_events(dataset)
        return RawToEventsP300(event_id=event_id, interval=dataset.interval)

    @property
    def datasets(self):
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm="p300", events=self.events, interval=interval, has_all_events=True
        )

    @property
    def scoring(self):
        return "roc_auc"


class SinglePass(BaseP300):
    """Single Bandpass filter P300.

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
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline
            correction. If None, do not apply baseline correction.
            If a tuple (a, b), the interval is between a and b (in seconds),
            including the endpoints.
            Correction is applied by computing the mean of the baseline period
            and subtracting it from the data (see mne.Epochs)

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    """

    def __init__(self, fmin=1, fmax=24, **kwargs):
        if "filters" in kwargs.keys():
            raise (ValueError("P300 does not take argument filters"))
        super().__init__(filters=[[fmin, fmax]], **kwargs)

    @property
    def fmax(self):
        return self.filters[0][1]

    @property
    def fmin(self):
        return self.filters[0][0]


class P300(SinglePass):
    """P300 for Target/NonTarget classification.

    Metric is 'roc_auc'
    """

    def __init__(self, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("P300 dont accept events"))
        super().__init__(events=["Target", "NonTarget"], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return "roc_auc"


class FakeP300Paradigm(P300):
    """Fake P300 for Target/NonTarget classification."""

    @property
    def datasets(self):
        return [FakeDataset(["Target", "NonTarget"], paradigm="p300")]

    def is_valid(self, dataset):
        return dataset.paradigm == "p300"
