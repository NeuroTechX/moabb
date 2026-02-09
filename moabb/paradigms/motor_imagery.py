"""Motor Imagery Paradigms."""

import abc
import logging

from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.base import BaseParadigm


log = logging.getLogger(__name__)


class BaseMotorImagery(BaseParadigm):
    """Base Motor imagery paradigm.

    Please use one of the child classes
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
        scorer=None,
        overlap=None,
    ):
        if overlap is not None and not (0 <= overlap < 100):
            raise ValueError("overlap must be in [0, 100)")

        super().__init__(
            filters=filters,
            events=events,
            channels=channels,
            baseline=baseline,
            resample=resample,
            tmin=tmin,
            tmax=tmax,
            overlap=overlap,
            scorer=scorer,
        )

    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == "imagery"):
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
        if self.scorer is not None:
            return self.scorer
        return "accuracy"


class LeftRightImagery(BaseMotorImagery):
    """Motor Imagery for left hand/right hand classification.

    Metric is 'roc_auc' by default

    Parameters
    ----------

    fmin: float (default 8)
        cutoff frequency (Hz) for the high pass filter.

    fmax: float (default 32)
        cutoff frequency (Hz) for the low pass filter.

    """

    def __init__(
        self,
        fmin=8,
        fmax=32,
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        scorer=None,
        overlap=None,
    ):
        if events is not None:
            raise ValueError("LeftRightImagery dont accept events")
        super().__init__(
            filters=[[fmin, fmax]],
            events=["left_hand", "right_hand"],
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=channels,
            resample=resample,
            overlap=overlap,
            scorer=scorer,
        )

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        if self.scorer is not None:
            return self.scorer
        return "roc_auc"


class FilterBankLeftRightImagery(LeftRightImagery):
    """Filter Bank Motor Imagery for left hand/right hand classification.

    Metric is 'roc_auc' by default

    Parameters
    ----------

    filters: list of list (defaults ([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]))
        bank of bandpass filter to apply.

    """

    def __init__(
        self,
        filters=([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]),
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        scorer=None,
        overlap=None,
    ):
        if events is not None:
            raise ValueError("LeftRightImagery dont accept events")
        BaseMotorImagery.__init__(
            self,
            filters=filters,
            events=["left_hand", "right_hand"],
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=channels,
            resample=resample,
            overlap=overlap,
            scorer=scorer,
        )

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        if self.scorer is not None:
            return self.scorer
        return "roc_auc"


class MotorImagery(BaseMotorImagery):
    """N-class motor imagery.

    By default, metric is 'roc-auc' if 2 classes and 'accuracy' if more

    Parameters
    ----------

    events: List of str
        event labels used to filter datasets (e.g. if only motor imagery is
        desired).

    n_classes: int,
        number of classes each dataset must have. If events is given,
        requires all imagery sorts to be within the events list.

    fmin: float (default 8)
        cutoff frequency (Hz) for the high pass filter.

    fmax: float (default 32)
        cutoff frequency (Hz) for the low pass filter.

    """

    def __init__(
        self,
        n_classes=None,
        fmin=8,
        fmax=32,
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        scorer=None,
        overlap=None,
    ):
        super().__init__(
            filters=[[fmin, fmax]],
            events=events,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=channels,
            resample=resample,
            overlap=overlap,
            scorer=scorer,
        )
        self.n_classes = n_classes

        if self.events is None:
            log.warning("Choosing from all possible events")
        elif self.n_classes is not None:
            assert n_classes <= len(self.events), "More classes than events specified"

    def is_valid(self, dataset):
        ret = True
        if not dataset.paradigm == "imagery":
            ret = False
        elif self.n_classes is None and self.events is None:
            pass
        elif self.events is None:
            if not len(dataset.event_id) >= self.n_classes:
                ret = False
        else:
            overlap = len(set(self.events) & set(dataset.event_id.keys()))
            if self.n_classes is not None and not overlap >= self.n_classes:
                ret = False
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
        if self.scorer is not None:
            return self.scorer
        if self.n_classes == 2:
            return "roc_auc"
        return "accuracy"


class FilterBankMotorImagery(MotorImagery):
    """Filter bank n-class motor imagery.

    By default, metric is 'roc-auc' if 2 classes and 'accuracy' if more

    Parameters
    ----------

    events: List of str
        event labels used to filter datasets (e.g. if only motor imagery is
        desired).

    n_classes: int,
        number of classes each dataset must have. If events is given,
        requires all imagery sorts to be within the events list.
    """

    def __init__(
        self,
        n_classes=2,
        filters=([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]),
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        scorer=None,
        overlap=None,
    ):
        BaseMotorImagery.__init__(
            self,
            filters=filters,
            events=events,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=channels,
            resample=resample,
            overlap=overlap,
            scorer=scorer,
        )
        self.n_classes = n_classes

        if self.events is None:
            log.warning("Choosing from all possible events")
        else:
            assert n_classes <= len(self.events), "More classes than events specified"

    def is_valid(self, dataset):
        ret = True
        if not dataset.paradigm == "imagery":
            ret = False
        if self.events is None:
            if not len(dataset.event_id) >= self.n_classes:
                ret = False
        else:
            overlap = len(set(self.events) & set(dataset.event_id.keys()))
            if not overlap >= self.n_classes:
                ret = False
        return ret

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
        if self.scorer is not None:
            return self.scorer
        if self.n_classes == 2:
            return "roc_auc"
        return "accuracy"


class FakeImageryParadigm(LeftRightImagery):
    """Fake Imagery for left hand/right hand classification."""

    @property
    def datasets(self):
        return [FakeDataset(["left_hand", "right_hand"], paradigm="imagery")]

    def is_valid(self, dataset):
        return dataset.paradigm == "imagery"
