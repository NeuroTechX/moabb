from moabb.datasets import utils
from moabb.datasets.preprocessing import RawToFixedIntervalEvents
from moabb.paradigms.base import BaseProcessing


class FixedIntervalWindowsProcessing(BaseProcessing):
    """Paradigm for creating epochs at fixed interval,
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

    marker: int (default 1)
        Marker to use for the events created.
    """

    def __init__(
        self,
        filters=((7, 45)),
        baseline=None,
        channels=None,
        resample=None,
        length: float = 5.0,
        stride: float = 10.0,
        start_offset=0.0,
        stop_offset=None,
        marker=1,
    ):
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

    def _to_samples(self, key, dataset=None):
        value = getattr(self, key)
        if dataset is None and self.resample is None:
            raise ValueError(f"{key}_samples: dataset or resample must be specified")
        return int(value * self.resample)

    def length_samples(self, dataset=None):
        return self._to_samples("length", dataset)

    def stride_samples(self, dataset=None):
        return self._to_samples("stride", dataset)

    def start_offset_samples(self, dataset=None):
        return self._to_samples("start_offset", dataset)

    def stop_offset_samples(self, dataset=None):
        if self.stop_offset is None:
            return None
        return self._to_samples("stop_offset", dataset)

    def used_events(self, dataset):
        return {"Window": self.marker}

    def is_valid(self, dataset):
        return True

    @property
    def datasets(self):
        return utils.dataset_search(paradigm=None)

    def _get_events_pipeline(self, dataset):
        return RawToFixedIntervalEvents(
            length_samples=self.length_samples(dataset),
            stride_samples=self.stride_samples(dataset),
            start_offset_samples=self.start_offset_samples(dataset),
            stop_offset_samples=self.stop_offset_samples(dataset),
            marker=self.marker,
        )
