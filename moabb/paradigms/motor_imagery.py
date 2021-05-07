"""Motor Imagery Paradigms"""

import abc
import logging

from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.base import BaseParadigm


log = logging.getLogger(__name__)


class BaseMotorImagery(BaseParadigm):
    """Base Motor imagery paradigm.

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
        filters=([7, 35],),
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
    ):
        super().__init__()
        self.filters = filters
        self.events = events
        self.channels = channels
        self.baseline = baseline
        self.resample = resample

        if tmax is not None:
            if tmin >= tmax:
                raise (ValueError("tmax must be greater than tmin"))

        self.tmin = tmin
        self.tmax = tmax

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
        return "accuracy"


class SinglePass(BaseMotorImagery):
    """Single Bandpass filter motot Imagery.

    Motor imagery paradigm with only one bandpass filter (default 8 to 32 Hz)

    Parameters
    ----------
    fmin: float (default 8)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 32)
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

    def __init__(self, fmin=8, fmax=32, **kwargs):
        if "filters" in kwargs.keys():
            raise (ValueError("MotorImagery does not take argument filters"))
        super().__init__(filters=[[fmin, fmax]], **kwargs)


class FilterBank(BaseMotorImagery):
    """Filter Bank MI."""

    def __init__(
        self,
        filters=([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]),
        **kwargs,
    ):
        """init"""
        super().__init__(filters=filters, **kwargs)


class LeftRightImagery(SinglePass):
    """Motor Imagery for left hand/right hand classification

    Metric is 'roc_auc'

    """

    def __init__(self, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("LeftRightImagery dont accept events"))
        super().__init__(events=["left_hand", "right_hand"], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return "roc_auc"


class FilterBankLeftRightImagery(FilterBank):
    """Filter Bank Motor Imagery for left hand/right hand classification

    Metric is 'roc_auc'

    """

    def __init__(self, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("LeftRightImagery dont accept events"))
        super().__init__(events=["left_hand", "right_hand"], **kwargs)

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    @property
    def scoring(self):
        return "roc_auc"


class FilterBankMotorImagery(FilterBank):
    """
    Filter bank n-class motor imagery.

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
            total_classes=self.n_classes,
            interval=interval,
            has_all_events=False,
        )

    @property
    def scoring(self):
        if self.n_classes == 2:
            return "roc_auc"
        else:
            return "accuracy"


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

    fmin: float (default 8)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 32)
        cutoff frequency (Hz) for the low pass filter

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the begining of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the begining of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the begining of the task as defined in the dataset. If
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

    def __init__(self, n_classes=2, **kwargs):
        super().__init__(**kwargs)
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
        if self.n_classes == 2:
            return "roc_auc"
        else:
            return "accuracy"


class FakeImageryParadigm(LeftRightImagery):
    """Fake Imagery for left hand/right hand classification."""

    @property
    def datasets(self):
        return [FakeDataset(["left_hand", "right_hand"], paradigm="imagery")]
