from moabb.datasets import utils
from moabb.datasets.preprocessing import RawToFixedIntervalEvents
from moabb.paradigms.base import BaseProcessing


class BaseFixedIntervalWindowsProcessing(BaseProcessing):
    """Base class for fixed interval windows processing.

    Paradigm for creating epochs at fixed interval,
    ignoring the stim channel and events of the dataset.

    Parameters
    ----------

    filters: list of list (default [[7, 45]])
        bank of bandpass filter to apply.

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

    length: float (default 5.0)
        Length of the epochs in seconds.

    stride: float (default 10.0)
        Stride between epochs in seconds.

    start_offset: float (default 0.0)
        Start from the beginning of the raw recordings in seconds.

    stop_offset: float | None (default None)
        Stop offset from beginning of raw recordings in seconds.
        If None, set to be the end of the recording.

    marker: int (default -1)
        Marker to use for the events created.

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    def __init__(
        self,
        filters=None,
        baseline=None,
        channels=None,
        resample=None,
        length: float = 5.0,
        stride: float = 10.0,
        start_offset=0.0,
        stop_offset=None,
        marker=-1,
    ):
        if not filters:
            raise ValueError("filters must be specified")
        tmin = 0.0
        tmax = length
        super().__init__(
            filters=filters,
            channels=channels,
            baseline=baseline,
            resample=resample,
            tmin=tmin,
            tmax=tmax,
        )
        self.length = length
        self.stride = stride
        self.start_offset = start_offset
        self.stop_offset = stop_offset
        self.marker = marker

    def _to_samples(self, key):
        value = getattr(self, key)
        if self.resample is None:
            raise ValueError(f"{key}_samples: must be specified")
        if value is None:
            raise ValueError(f"{key}_samples: {key} must be specified")
        return int(value * self.resample)

    @property
    def length_samples(self):
        return self._to_samples("length")

    @property
    def stride_samples(self):
        return self._to_samples("stride")

    @property
    def start_offset_samples(self):
        return self._to_samples("start_offset")

    @property
    def stop_offset_samples(self):
        return self._to_samples("stop_offset")

    def used_events(self, dataset):
        return {"Window": self.marker}

    def is_valid(self, dataset):
        return True

    @property
    def datasets(self):
        return utils.dataset_search(paradigm=None)

    def _get_events_pipeline(self, dataset):
        return RawToFixedIntervalEvents(
            length=self.length,
            stride=self.stride,
            start_offset=self.start_offset,
            stop_offset=self.stop_offset,
            marker=self.marker,
        )


class FixedIntervalWindowsProcessing(BaseFixedIntervalWindowsProcessing):
    """Fixed interval windows processing.

    Paradigm for creating epochs at fixed interval,
    ignoring the stim channel and events of the dataset.

     Parameters
     ----------

    fmin: float (default 7)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 45)
        cutoff frequency (Hz) for the low pass filter

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

    length: float (default 5.0)
        Length of the epochs in seconds.

    stride: float (default 10.0)
        Stride between epochs in seconds.

    start_offset: float (default 0.0)
        Start from the beginning of the raw recordings in seconds.

    stop_offset: float | None (default None)
        Stop offset from beginning of raw recordings in seconds.
        If None, set to be the end of the recording.

    marker: int (default -1)
        Marker to use for the events created.
    """

    def __init__(
        self,
        fmin=7,
        fmax=45,
        baseline=None,
        channels=None,
        resample=None,
        length: float = 5.0,
        stride: float = 10.0,
        start_offset=0.0,
        stop_offset=None,
        marker=-1,
    ):
        super().__init__(
            filters=[(fmin, fmax)],
            baseline=baseline,
            channels=channels,
            resample=resample,
            length=length,
            stride=stride,
            start_offset=start_offset,
            stop_offset=stop_offset,
            marker=marker,
        )


class FilterBankFixedIntervalWindowsProcessing(BaseFixedIntervalWindowsProcessing):
    """Filter bank fixed interval windows processing.

    Paradigm for creating epochs at fixed interval
    with multiple narrow bandpass filters,
    ignoring the stim channel and events of the dataset.

    Parameters
    ----------

    filters: list of list (default ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)))
        bank of bandpass filter to apply.

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

    length: float (default 5.0)
        Length of the epochs in seconds.

    stride: float (default 10.0)
        Stride between epochs in seconds.

    start_offset: float (default 0.0)
        Start from the beginning of the raw recordings in seconds.

    stop_offset: float | None (default None)
        Stop offset from beginning of raw recordings in seconds.
        If None, set to be the end of the recording.

    marker: int (default -1)
        Marker to use for the events created.
    """

    def __init__(
        self,
        filters=((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)),
        baseline=None,
        channels=None,
        resample=None,
        length: float = 5.0,
        stride: float = 10.0,
        start_offset=0.0,
        stop_offset=None,
        marker=-1,
    ):
        super().__init__(
            filters=filters,
            baseline=baseline,
            channels=channels,
            resample=resample,
            length=length,
            stride=stride,
            start_offset=start_offset,
            stop_offset=stop_offset,
            marker=marker,
        )
