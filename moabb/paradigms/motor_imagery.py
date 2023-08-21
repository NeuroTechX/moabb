"""Motor Imagery Paradigms."""

import abc
import logging

from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.base import BaseParadigm

log = logging.getLogger(__name__)


class BaseMotorImagery(BaseParadigm):
    """Base Motor imagery paradigm.

    Not to be instantiated.

    """

    def __init__(
        self,
        filters=([7, 35],),
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

        ret = dataset.paradigm == "imagery"
        if not ret:
            return ret

        # check if dataset has required events
        if self.events:
            if not set(self.events) <= set(dataset.event_id.keys()):
                ret = False

        # we should verify list of channels, somehow
        return ret

    @abc.abstractmethod
    def used_events(self, dataset):
        pass

    @property
    def datasets(self):
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm="imagery", events=self.events, interval=interval, has_all_events=True
        )

    @property
    def scoring(self):
        return "accuracy"


class LeftRightImagery(BaseMotorImagery):
    """Motor Imagery for left hand/right hand classification.

    Attributes
    -----------

    fmin: float (default 8)
        cutoff frequency (Hz) for the high pass filter.

    fmax: float (default 32)
        cutoff frequency (Hz) for the low pass filter.

    """

    def __init__(self, fmin=8, fmax=32, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("LeftRightImagery dont accept events"))
        if "filters" in kwargs.keys():
            raise (ValueError("LeftRightImagery does not take argument filters"))
        super().__init__(
            filters=[[fmin, fmax]], events=["left_hand", "right_hand"], **kwargs
        )

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return "roc_auc"


class FilterBankLeftRightImagery(LeftRightImagery):
    """Filter Bank Motor Imagery for left/right hand classification."""

    def __init__(
        self,
        filters=([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]),
        **kwargs,
    ):
        if "events" in kwargs.keys():
            raise (ValueError("LeftRightImagery dont accept events"))
        super(LeftRightImagery, self).__init__(
            filters=filters, events=["left_hand", "right_hand"], **kwargs
        )


class MotorImagery(BaseMotorImagery):
    """N-class Motor Imagery.

    Attributes
    -----------

    fmin: float (default 8)
        cutoff frequency (Hz) for the high pass filter.

    fmax: float (default 32)
        cutoff frequency (Hz) for the low pass filter.

    n_classes: int (default number of available classes)
        number of MotorImagery classes/events to select.

    """

    def __init__(self, fmin=8, fmax=32, n_classes=None, **kwargs):
        if "filters" in kwargs.keys():
            raise (ValueError("MotorImagery does not take argument filters"))
        super().__init__(filters=[[fmin, fmax]], **kwargs)
        self.n_classes = n_classes
        if self.events is None:
            log.warning("Choosing from all possible events")
        elif self.n_classes is not None:
            assert n_classes <= len(self.events), "More classes than events specified"

    def is_valid(self, dataset):

        ret = dataset.paradigm == "imagery"
        if not ret:
            return ret

        if self.events is None and self.n_classes:
            ret = len(dataset.event_id) >= self.n_classes
        elif self.events and self.n_classes:
            overlap = len(set(self.events) & set(dataset.event_id.keys()))
            ret = overlap >= self.n_classes

        return ret

    def used_events(self, dataset):
        out = {}
        if self.events is None:
            for k, v in dataset.event_id.items():
                out[k] = v
            if self.n_classes is None:
                self.n_classes = len(out)
        else:
            for event in self.events:
                if event in dataset.event_id.keys():
                    out[event] = dataset.event_id[event]
                if len(out) == self.n_classes:
                    break
        if len(out) < self.n_classes:
            raise (
                ValueError(
                    f"Dataset {dataset.code} did not have enough "
                    f"events in {self.events} to run analysis"
                )
            )
        return out

    @property
    def datasets(self):
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm="imagery",
            events=self.events,
            interval=interval,
            has_all_events=False,
        )

    @property
    def scoring(self):
        if self.n_classes == 2:
            return "roc_auc"
        else:
            return "accuracy"


class FilterBankMotorImagery(MotorImagery):
    """Filter bank N-class motor imagery."""

    def __init__(
        self,
        filters=([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]),
        **kwargs,
    ):
        super(MotorImagery, self).__init__(filters=filters, **kwargs)
        self.n_classes = n_classes
        if self.events is None:
            log.warning("Choosing from all possible events")
        elif self.n_classes is not None:
            assert n_classes <= len(self.events), "More classes than events specified"

class FakeImageryParadigm(LeftRightImagery):
    """Fake Imagery for left hand/right hand classification."""

    @property
    def datasets(self):
        return [FakeDataset(["left_hand", "right_hand"], paradigm="imagery")]

    def is_valid(self, dataset):
        return dataset.paradigm == "imagery"
