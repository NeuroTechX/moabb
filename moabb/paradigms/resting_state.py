"""Resting state Paradigms.

Regroups paradigms for experience where we record the EEG and the
participant is not doing an active task, such as focusing, counting or
speaking.

Typically, a open/close eye experiment, where we record the EEG of a
subject while he is having the eye open or close is a resting state
experiment.
"""

from scipy.signal import welch

from moabb.paradigms.p300 import BaseP300


class RestingStateToP300Adapter(BaseP300):
    """Adapter to the P300 paradigm for resting state experiments.

    It implements a single bandpass processing as for P300, except that:
    - the name of the event is free (it is not enforced to Target/NonTarget as for P300)
    - the default values are different. In particular, the length of the epochs is larger.

    Parameters
    ----------
    fmin: float (default 1)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 35)
        cutoff frequency (Hz) for the low pass filter

    tmin: float (default 10)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default 50)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    resample: float | None (default 128)
        If not None, resample the eeg data with the sampling rate provided.
    """

    def __init__(
        self,
        fmin=1,
        fmax=35,
        events=None,
        tmin=10,
        tmax=50,
        baseline=None,
        channels=None,
        resample=128,
        ignore_relabelling=False,
        scorer=None,
    ):
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
        if self.scorer is not None:
            return self.scorer
        return "roc_auc"
