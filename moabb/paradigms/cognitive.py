"""Cognitive task paradigms."""

import logging

from moabb.datasets import utils
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.base import BaseParadigm


log = logging.getLogger(__name__)


class CognitiveTask(BaseParadigm):
    """N-class cued cognitive task.

    Standalone paradigm for non-motor endogenous tasks decoded from
    trial-wise EEG: handwriting/stroke imagery, mental arithmetic,
    mental rotation / spatial navigation imagery, word generation /
    verbal fluency, music or melody imagery. Follows the ``Cued
    Cognitive Tasks`` branch of the #1056 discussion (jarod1212),
    distinct from motor imagery (ERD/ERS) and from evoked ERP/P300.

    Decoded with the same trial-epoch machinery as motor imagery;
    defaults match :class:`~moabb.paradigms.MotorImagery` (8-32 Hz
    bandpass), so handwriting-imagery scripts can swap names without
    behavior changes. Valid on both ``paradigm="cognitive"`` datasets
    and the existing ``paradigm="imagery"`` tag, so current loaders
    such as Wang2026Handwriting work today while future loaders can
    adopt the ``"cognitive"`` tag.

    By default, metric is 'roc-auc' if 2 classes and 'accuracy' if more

    Parameters
    ----------

    events: List of str
        event labels used to filter datasets.

    n_classes: int,
        number of classes each dataset must have. If events is given,
        requires all task sorts to be within the events list.

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
        reject_by_annotation=True,
    ):
        if overlap is not None and not (0 <= overlap < 100):
            raise ValueError("overlap must be in [0, 100)")
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
            reject_by_annotation=reject_by_annotation,
        )
        self.n_classes = n_classes

        if self.events is not None and len(self.events) == 0:
            raise ValueError("events must be None or a non-empty list of event names")

        if self.events is None:
            log.warning("Choosing from all possible events")
        elif self.n_classes is not None:
            if n_classes > len(self.events):
                raise ValueError(
                    f"n_classes ({n_classes}) exceeds number of events ({len(self.events)})"
                )

    def is_valid(self, dataset):
        ret = True
        if getattr(dataset, "paradigm", None) not in ("cognitive", "imagery"):
            ret = False
        elif self.n_classes is None and self.events is None:
            pass
        elif self.events is None:
            if not len(dataset.event_id) >= self.n_classes:
                ret = False
        else:
            n_overlap = len(set(self.events) & set(dataset.event_id.keys()))
            if self.n_classes is not None and not n_overlap >= self.n_classes:
                ret = False
            elif self.n_classes is None and n_overlap == 0:
                ret = False
        return ret

    def used_events(self, dataset):
        out = {}
        if self.events is None:
            for k, v in dataset.event_id.items():
                out[k] = v
        else:
            for event in self.events:
                if event in dataset.event_id.keys():
                    out[event] = dataset.event_id[event]
                if self.n_classes is not None and len(out) == self.n_classes:
                    break
        if self.n_classes is not None and len(out) < self.n_classes:
            raise (
                ValueError(
                    f"Dataset {dataset.code} did not have enough "
                    f"events in {self.events} to run analysis"
                )
            )
        return out

    @property
    def datasets(self):
        # Transitional bridge: imagery search keeps current loaders
        # (e.g. Wang2026Handwriting) discoverable until they adopt
        # paradigm="cognitive". Costs ~2x catalog scan; scope down
        # once the migration lands.
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        found = utils.dataset_search(
            paradigm="imagery",
            events=self.events,
            interval=interval,
            has_all_events=False,
        )
        found += utils.dataset_search(
            paradigm="cognitive",
            events=self.events,
            interval=interval,
            has_all_events=False,
        )
        return found

    @property
    def scoring(self):
        if self.scorer is not None:
            return self.scorer
        if self.n_classes == 2:
            return "roc_auc"
        return "accuracy"


class FakeCognitiveParadigm(CognitiveTask):
    """Fake cognitive task for tests."""

    @property
    def datasets(self):
        return [FakeDataset(["subtraction", "rest"], paradigm="cognitive")]

    def is_valid(self, dataset):
        return getattr(dataset, "paradigm", None) in ("cognitive", "imagery")
