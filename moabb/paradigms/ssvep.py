"""Steady-State Visually Evoked Potentials Paradigms."""

import logging

from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.base import BaseParadigm


log = logging.getLogger(__name__)


class BaseSSVEP(BaseParadigm):
    """Base SSVEP Paradigm.

    Parameters
    ----------
    filters: list of list | None (default ((7, 45),))
        Bank of bandpass filter to apply.

    n_classes: int or None (default None)
        Number of classes each dataset must have. All dataset classes if None.
    """

    def __init__(
        self,
        filters=((7, 45),),
        events=None,
        n_classes=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        scorer=None,
    ):
        """Init the BaseSSVEP function."""

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

        self.n_classes = n_classes
        if self.events is None:
            log.warning(
                "Choosing the first "
                + str(n_classes)
                + " classes"
                + " from all possible events"
            )
        else:
            assert n_classes <= len(self.events), "More classes than events specified"

    def is_valid(self, dataset):
        """Check if dataset is valid for the SSVEP paradigm."""
        ret = True
        if not (dataset.paradigm == "ssvep"):
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
                    f"freqs in {self.events} to run analysis"
                )
            )
        return out

    def prepare_process(self, dataset):
        """Prepare dataset for processing, and using events if needed.

        This function is called before the processing function, and is used to
        prepare the dataset for processing. This includes:
        get the events used for the paradigm, and set the filters if needed.

        Parameters
        ----------
        dataset: moabb.datasets.base.BaseDataset
            Dataset to prepare.
        """
        event_id = self.used_events(dataset)

        # get filters
        if self.filters is None:
            self.filters = [
                [float(f) - 0.5, float(f) + 0.5]
                for f in event_id.keys()
                if f.replace(".", "", 1).isnumeric()
            ]

    @property
    def datasets(self):
        """List of datasets valid for the paradigm."""
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm="ssvep",
            events=self.events,
            # total_classes=self.n_classes,
            interval=interval,
            has_all_events=True,
        )

    @property
    def scoring(self):
        """Return the scoring method for this paradigm.

        By default, if n_classes use the roc_auc, else use accuracy. More details
        about this default scoring method can be found in the original
        moabb paper.
        """
        if self.scorer is not None:
            return self.scorer
        if self.n_classes == 2:
            return "roc_auc"
        return "accuracy"


class SSVEP(BaseSSVEP):
    """Single bandpass filter SSVEP.

    SSVEP paradigm with only one bandpass filter (default 7 to 45 Hz)
    Metric is 'roc-auc' if 2 classes and 'accuracy' if more

    Parameters
    ----------
    fmin: float (default 7)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 45)
        cutoff frequency (Hz) for the low pass filter
    """

    def __init__(
        self,
        fmin=7,
        fmax=45,
        filters=None,
        events=None,
        n_classes=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        scorer=None,
    ):
        if filters is not None:
            raise ValueError("SSVEP does not take argument filters")
        super().__init__(
            filters=[(fmin, fmax)],
            events=events,
            n_classes=n_classes,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=channels,
            resample=resample,
            scorer=scorer,
        )


class FilterBankSSVEP(BaseSSVEP):
    """Filtered bank n-class SSVEP paradigm.

    SSVEP paradigm with multiple narrow bandpass filters, centered around the
    frequencies of considered events.
    Metric is 'roc-auc' if 2 classes and 'accuracy' if more.

    Parameters
    ----------
    filters: list of list | None (default None)
        If None, bandpass set around freqs of events with [f_n-0.5, f_n+0.5]
    """

    def __init__(
        self,
        filters=None,
        events=None,
        n_classes=None,
        tmin=0.0,
        tmax=None,
        baseline=None,
        channels=None,
        resample=None,
        scorer=None,
    ):
        super().__init__(
            filters=filters,
            events=events,
            n_classes=n_classes,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=channels,
            resample=resample,
            scorer=scorer,
        )


class FakeSSVEPParadigm(BaseSSVEP):
    """Fake SSVEP classification."""

    @property
    def datasets(self):
        """Return a fake dataset with event list 13 and 15."""
        return [FakeDataset(event_list=["13", "15"], paradigm="ssvep")]

    def is_valid(self, dataset):
        """Overwrite the original function, always True in FakeDataset."""
        return dataset.paradigm == "ssvep"
