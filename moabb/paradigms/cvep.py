"""c-VEP Paradigms"""

import logging

from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.base import BaseParadigm


log = logging.getLogger(__name__)


class BaseCVEP(BaseParadigm):
    """Base c-VEP paradigm for epoch-level decoding.

    Please use one of the child classes.

    This paradigm is meant to be used for epoch-level decoding of c-VEP
    datasets; this means that the goal of the classifiers with this paradigm
    is to predict, for every stimulation one by one, the amplitude of
    the target code. The code value is between 0  (i.e., stimulation off)
    and 1 (i.e., maximal stimulation).

    Parameters
    ----------

    filters: tuple of tuple (defaults ((1.0, 45.0),))
        Bank of bandpass filter to apply.

    events: List of str | None (default None)
        Event to use for epoching. Note, we stick to a convention where the
        intensity level is encoded as float. For example, a binary sequence
        would have events 1.0 (i.e., on) and 0.0 (i.e., off). If None, default
        to all events defined in the dataset.

    n_classes: int or None (default None)
        Number of classes each dataset must have. All dataset classes if None.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2 (default None)
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

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    def __init__(
        self,
        filters=((1.0, 45.0),),
        events=None,
        n_classes=None,
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

        self.n_classes = n_classes
        if self.events is None:
            log.warning(
                f"Choosing the first {n_classes} classes from all possible events."
            )
        else:
            assert n_classes <= len(self.events), "More classes than events specified"

    def is_valid(self, dataset):
        """Check if dataset is valid for the c-VEP paradigm."""
        ret = True
        if not (dataset.paradigm == "cvep"):
            ret = False

        # check if dataset has required events
        if self.events:
            if not set(self.events) <= set(dataset.event_id.keys()):
                ret = False

        return ret

    def used_events(self, dataset):
        """Return the mne events used for the dataset."""
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
            raise (
                ValueError(
                    f"Dataset {dataset.code} did not have enough "
                    f"events in {self.events} to run analysis"
                )
            )
        return out

    @property
    def datasets(self):
        """List of datasets valid for the paradigm."""
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm="cvep", events=self.events, interval=interval, has_all_events=True
        )

    @property
    def scoring(self):
        """Return the default scoring method for this paradigm.

        If n_classes==2 use the roc_auc, else use accuracy. More details
        about this default scoring method can be found in the original
        moabb paper.
        """
        if self.n_classes and self.n_classes == 2:
            return "roc_auc"
        else:
            return "accuracy"


class CVEP(BaseCVEP):
    """Single bandpass c-VEP paradigm for epoch-level decoding.

    c-VEP paradigm with only one bandpass filter (default 1 to 45 Hz)
    Metric is 'roc-auc' if 2 classes and 'accuracy' if more.
    This paradigm is meant to be used for epoch-level decoding of c-VEP
    datasets; this means that the goal of the classifiers with this paradigm
    is to predict, for every stimulation one by one, the amplitude of
    the target code. The code value is between 0  (i.e., stimulation off)
    and 1 (i.e., maximal stimulation).


    Parameters
    ----------
    fmin: float (default 1.0)
        cutoff frequency (Hz) for the highpass filter

    fmax: float (default 45.0)
        cutoff frequency (Hz) for the lowpass filter

    events: List of str | None (default None)
        Event to use for epoching. Note, we stick to a convention where the
        intensity level is encoded as float. For example, a binary sequence
        would have events 1.0 (i.e., on) and 0.0 (i.e., off). If None, default
        to all events defined in the dataset.

    n_classes: int or None (default None)
        Number of classes each dataset must have. All dataset classes if None

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2 (default None)
        The time interval to consider as “baseline” when applying baseline
        correction. If None, do not apply baseline correction.
        If a tuple (a, b), the interval is between a and b (in seconds),
        including the endpoints.
        Correction is applied by computing the mean of the baseline period
        and subtracting it from the data (see mne.Epochs)

    channels: list of str | None (default None)
        List of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    def __init__(self, fmin=1.0, fmax=45.0, **kwargs):
        if "filters" in kwargs.keys():
            raise (ValueError("c-VEP does not take argument filters"))
        super().__init__(filters=((fmin, fmax),), **kwargs)


class FilterBankCVEP(BaseCVEP):
    """Filterbank c-VEP paradigm for epoch-level decoding.

    c-VEP paradigm with multiple bandpass filters.
    Metric is 'roc-auc' if 2 classes and 'accuracy' if more.
    This paradigm is meant to be used for epoch-level decoding of c-VEP
    datasets; this means that the goal of the classifiers with this paradigm
    is to predict, for every stimulation one by one, the amplitude of
    the target code. The code value is between 0  (i.e., stimulation off)
    and 1 (i.e., maximal stimulation).

    Parameters
    ----------
    filters: tuple of tuple | None (default ((1., 45.), (12., 45.), (30., 45.)))
        Bank of bandpass filter to apply.
    events: List of str | None (default None)
        Event to use for epoching. Note, we stick to a convention where the
        intensity level is encoded as float. For example, a binary sequence
        would have events 1.0 (i.e., on) and 0.0 (i.e., off). If None, default
        to all events defined in the dataset.
    n_classes: int or None (default None)
        Number of classes each dataset must have. All dataset classes if None
    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.
    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.
    baseline: None | tuple of length 2 (default None)
        The time interval to consider as “baseline” when applying baseline
        correction. If None, do not apply baseline correction.
        If a tuple (a, b), the interval is between a and b (in seconds),
        including the endpoints.
        Correction is applied by computing the mean of the baseline period
        and subtracting it from the data (see mne.Epochs)
    channels: list of str | None (default None)
        List of channel to select. If None, use all EEG channels available in
        the dataset.
    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    def __init__(self, filters=((1.0, 45.0), (12.0, 45.0), (30.0, 45.0)), **kwargs):
        super().__init__(filters=filters, **kwargs)


class FakeCVEPParadigm(BaseCVEP):
    """Fake c-VEP paradigm."""

    @property
    def datasets(self):
        """Return a fake dataset with event list 1.0 and 0.0."""
        return [FakeDataset(event_list=["1.0", "0.0"], paradigm="cvep")]

    def is_valid(self, dataset):
        """Overwrite the original function, always True in FakeDataset."""
        return dataset.paradigm == "cvep"
