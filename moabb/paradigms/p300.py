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

    filters: list of list (defaults [[1, 24]])
        bank of bandpass filter to apply.

    ignore_relabelling: bool (default False)
        If True, ignore the relabelling of the events. This is useful for
        datasets where the events are already in the correct format.
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
        ignore_relabelling=False,
        scorer=None,
    ):
        self.ignore_relabelling = ignore_relabelling
        super().__init__(
            filters=filters,
            events=events,
            channels=channels,
            baseline=baseline,
            resample=resample,
            tmin=tmin,
            tmax=tmax,
            scorer=scorer,
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
        return RawToEventsP300(
            event_id=event_id,
            interval=dataset.interval,
            ignore_relabelling=self.ignore_relabelling,
        )

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
    def fmax(self):
        return self.filters[0][1]

    @property
    def fmin(self):
        return self.filters[0][0]

    @property
    def scoring(self):
        if self.scorer is not None:
            return self.scorer
        return "roc_auc"


class P300(BaseP300):
    """P300 for Target/NonTarget classification.

    Metric is 'roc_auc' by default

    Parameters
    ----------

    fmin: float (default 1)
        cutoff frequency (Hz) for the high pass filter.

    fmax: float (default 24)
        cutoff frequency (Hz) for the low pass filter.

    events: List of str (default ["Target", "NonTarget"])
        event to use for epoching.
    """

    def __init__(
        self,
        fmin=1,
        fmax=24,
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        ignore_relabelling=False,
        scorer=None,
    ):
        if events is None:
            events = ["Target", "NonTarget"]
        super().__init__(
            filters=[[fmin, fmax]],
            events=events,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=channels,
            resample=resample,
            ignore_relabelling=ignore_relabelling,
            scorer=scorer,
        )

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        if self.scorer is not None:
            return self.scorer
        return "roc_auc"


class FakeP300Paradigm(P300):
    """Fake P300 for Target/NonTarget classification."""

    @property
    def datasets(self):
        return [FakeDataset(["Target", "NonTarget"], paradigm="p300")]

    def is_valid(self, dataset):
        return dataset.paradigm == "p300"
