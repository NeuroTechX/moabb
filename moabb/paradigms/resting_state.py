"""Resting state Paradigms.

Regroups paradigms for experience where we record the EEG and the
participant is not doing an active task, such as focusing, counting or
speaking.

Typically, a open/close eye experiment, where we record the EEG of a
subject while he is having the eye open or close is a resting state
experiment.
"""

from scipy.signal import welch

from moabb.paradigms.p300 import SinglePass


class RestingStateToP300Adapter(SinglePass):
    """Adapter to the P300 paradigm for resting state experiments.

    It implements a SinglePass processing as for P300, except that:
    - the name of the event is free (it is not enforced to Target/NonTarget as for P300)
    - the default values are different. In particular, the length of the epochs is larger.

    Parameters
    ----------
    fmin: float (default 1)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 35)
        cutoff frequency (Hz) for the low pass filter

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.

    tmin: float (default 10s)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default 50s)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    resample: float | None (default 128)
        If not None, resample the eeg data with the sampling rate provided.

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
    """

    def __init__(self, fmin=1, fmax=35, tmin=10, tmax=50, resample=128, **kwargs):
        super().__init__(
            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=resample, **kwargs
        )

    def used_events(self, dataset):
        return {ev: dataset.event_id[ev] for ev in self.events}

    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == "rstate"):
            ret = False

        if self.events:
            if not set(self.events) <= set(dataset.event_id.keys()):
                ret = False

        return ret

    def psd(self, subject, dataset):
        # power spectrum density for ease of use
        X, y, _ = self.get_data(dataset, [subject])
        f, S = welch(X, axis=-1, nperseg=1024, fs=self.resample)
        return (f, S, X, y)

    @property
    def scoring(self):
        return "roc_auc"
