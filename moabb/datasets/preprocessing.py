import logging
from collections import OrderedDict
from operator import methodcaller
from typing import Dict, List, Tuple, Union

import mne
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline, _name_estimators


# Handle different scikit-learn versions for _VisualBlock import
# sklearn >= 1.6 moved _VisualBlock to sklearn.utils._repr_html.estimator
# sklearn < 1.6 has it in sklearn.utils._estimator_html_repr
try:
    from sklearn.utils._repr_html.estimator import _VisualBlock
except (ImportError, ModuleNotFoundError):
    try:
        from sklearn.utils._estimator_html_repr import _VisualBlock
    except (ImportError, ModuleNotFoundError):
        # Fallback: create a dummy _VisualBlock for older sklearn versions
        # that don't have HTML representation support
        _VisualBlock = None


log = logging.getLogger(__name__)


class FixedPipeline(Pipeline):
    """A Pipeline that is always considered fitted.

    This is useful for pre-processing pipelines that don't require fitting,
    as they only apply fixed transformations (e.g., filtering, epoching).
    This avoids the FutureWarning from sklearn 1.8+ about unfitted pipelines.
    """

    def __sklearn_is_fitted__(self):
        """Return True to indicate this pipeline is always considered fitted."""
        return True


def make_fixed_pipeline(*steps, memory=None, verbose=False):
    """Create a FixedPipeline that is always considered fitted.

    This is a drop-in replacement for sklearn's make_pipeline that creates
    a pipeline marked as fitted, suitable for fixed transformations.

    Parameters
    ----------
    *steps : list of estimators
        List of (name, transform) tuples that are chained in the pipeline.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed.

    Returns
    -------
    p : FixedPipeline
        A FixedPipeline object.
    """

    return FixedPipeline(_name_estimators(steps), memory=memory, verbose=verbose)


def _is_none_pipeline(pipeline):
    """Check if a pipeline is the result of make_pipeline(None)"""
    return (
        isinstance(pipeline, Pipeline)
        and pipeline.steps[0][1] is None
        and len(pipeline) == 1
    )


def _unsafe_pick_events(events, include):
    try:
        return mne.pick_events(events, include=include)
    except RuntimeError as e:
        if str(e) == "No events found":
            return np.zeros((0, 3), dtype="int32")
        raise e


_REST_LABEL = -1


def _insert_rest_events(events, task_duration_samples, interval_start_samples=0):
    """Insert rest events in gaps between consecutive task trials.

    Adds rest events between trials and after the last trial so that
    the sliding window can cover the trial timeline.

    Parameters
    ----------
    events : ndarray of shape (n_events, 3)
        Task events sorted by onset (cue onsets).
    task_duration_samples : int
        Duration of each task trial in samples.
    interval_start_samples : int
        Offset from cue onset to actual task start, in samples.
        For datasets where ``interval[0] != 0`` (e.g. ``interval=(2, 6)``),
        this shifts the task segment to ``[cue + offset, cue + offset + duration]``.

    Returns
    -------
    all_events : ndarray of shape (n_all_events, 3)
        Merged task + rest events, sorted by onset.
    """
    rest = []
    for i in range(len(events)):
        # Task segment starts at cue + interval_start_samples
        task_start = events[i, 0] + interval_start_samples
        task_end = task_start + task_duration_samples

        # Pre-task rest: from cue onset to task start (if interval_start > 0)
        if interval_start_samples > 0:
            rest.append([events[i, 0], 0, _REST_LABEL])

        # Post-task rest: from task end to next trial's task start
        if i + 1 < len(events):
            next_task_start = events[i + 1, 0] + interval_start_samples
        else:
            next_task_start = task_end + 1
        if next_task_start > task_end:
            rest.append([task_end, 0, _REST_LABEL])

    # Move task event onsets to actual task start positions
    events = events.copy()
    events[:, 0] += interval_start_samples

    if not rest:
        return events
    rest = np.array(rest, dtype=events.dtype)
    merged = np.vstack([events, rest])
    return merged[merged[:, 0].argsort()]


def _generate_sliding_window_events(
    events, window_length, overlap, sfreq, interval, tmin=0.0
):
    """Generate sliding window events from original trial events.

    Simulates a pseudo-online BCI scenario by sliding a fixed-size window
    across the trial timeline starting from the first event onset (task + rest
    periods). Each window
    is assigned the label of whichever class occupies the majority of the
    window. Rest-labeled windows are kept in the output and expected to be
    filtered out downstream by ``_unsafe_pick_events``.

    The timeline is reconstructed as a continuous sequence of segments::

        [task_1] [rest] [task_2] [rest] ... [task_N] [rest]

    where rest periods are inferred from the gaps between consecutive
    task trials using the dataset ``interval``.

    Parameters
    ----------
    events : ndarray of shape (n_events, 3)
        Original MNE-style events array (task events only, sorted by onset).
    window_length : float
        Window length in seconds (typically ``tmax - tmin``).
    overlap : float
        Overlap percentage (0-100). Controls how much consecutive windows
        overlap. For example, 50 means each window shares half its length
        with the previous one.
    sfreq : float
        Sampling frequency in Hz.
    interval : tuple of (float, float)
        Dataset interval ``(tmin, tmax)`` in seconds, used to compute task
        trial duration and infer where rest periods fall.
    tmin : float
        Start time of the epoch relative to the dataset interval, in seconds.
        Used together with ``interval[0]`` to compute the epoch offset so that
        the label voting window aligns with the actual extracted data.

    Returns
    -------
    events_new : ndarray of shape (n_new_events, 3)
        New events array with sliding window onsets and majority-vote labels.
        Contains both task-labeled and rest-labeled (``_REST_LABEL``) windows.
    """
    if len(events) == 0:
        return np.zeros((0, 3), dtype="int32")

    window_samples = int(round(window_length * sfreq))
    if window_samples <= 0:
        raise ValueError("Window length must be strictly positive")

    # Compute task trial duration from the dataset interval and determine
    # the end of the last trial (used to bound the sliding window range).
    task_duration_samples = int(round((interval[1] - interval[0]) * sfreq))
    interval_start_samples = int(round(interval[0] * sfreq))

    # The epoch offset accounts for the fact that epoch extraction uses
    # tmin + interval[0] as the actual start time. This offset aligns the
    # label voting window with the data that will actually be extracted.
    epoch_offset_samples = int(round((tmin + interval[0]) * sfreq))

    last_task_end = int(events[-1, 0]) + interval_start_samples + task_duration_samples

    # Insert synthetic rest events in the gaps between trials (and after
    # the last trial) so the sliding window can traverse rest periods too.
    events = _insert_rest_events(events, task_duration_samples, interval_start_samples)

    # Stride = how far the window advances each step.
    # overlap=50 with a 750-sample window gives stride=375.
    stride_samples = int(round(window_samples * (1.0 - overlap / 100.0)))
    stride_samples = max(1, stride_samples)

    # Generate all candidate window start positions. Windows are allowed
    # to start anywhere from the first event up to one stride before the
    # last task trial ends, so the last trial is fully covered.
    first_onset = int(events[0, 0])
    max_start = last_task_end - stride_samples
    if max_start < first_onset:
        return np.zeros((0, 3), dtype="int32")

    onsets = np.arange(first_onset, max_start + 1, stride_samples, dtype="int32")

    # Build arrays of segment boundaries (transitions) and their labels.
    # Each event marks the start of a new segment that extends until the
    # next event. Example: transitions=[0, 1000, 1500, 2500] means
    # segment 0 runs from sample 0 to 999, segment 1 from 1000 to 1499, etc.
    transitions = events[:, 0].astype(np.int64, copy=False)
    labels = events[:, 2].astype(np.int32, copy=False)

    kept_onsets = []
    kept_labels = []

    for start in onsets:
        # The voting window is shifted by the epoch offset so that
        # label assignment matches the actual data extracted by epoching
        # (which uses tmin + interval[0] as start).
        vote_start = int(start + epoch_offset_samples)
        vote_end = int(vote_start + window_samples)

        # Find which segment the window starts in using binary search.
        # searchsorted(right) - 1 gives the index of the last transition
        # that is <= vote_start.
        seg_idx = np.searchsorted(transitions, vote_start, side="right") - 1
        durations_by_label = {}

        if seg_idx < 0:
            # vote_start is before the first transition -- treat as rest
            pre_samples = int(transitions[0]) - vote_start
            durations_by_label[_REST_LABEL] = pre_samples
            seg_start = int(transitions[0])
            seg_idx = 0
        else:
            seg_start = int(vote_start)

        # Walk through all segment boundaries that fall inside this window,
        # accumulating how many samples each label occupies.
        while seg_idx + 1 < len(transitions) and transitions[seg_idx + 1] < vote_end:
            current_label = int(labels[seg_idx])
            next_transition = int(transitions[seg_idx + 1])
            durations_by_label[current_label] = durations_by_label.get(
                current_label, 0
            ) + (next_transition - seg_start)
            seg_start = next_transition
            seg_idx += 1

        # Account for the final segment (from last transition to window end).
        last_label = int(labels[seg_idx])
        durations_by_label[last_label] = durations_by_label.get(last_label, 0) + (
            vote_end - seg_start
        )

        # Pick the label that occupies the most samples in this window.
        # On ties, favor task labels over rest so that boundary windows
        # are more likely to retain a meaningful class label.
        label = max(
            durations_by_label,
            key=lambda lbl: (durations_by_label[lbl], lbl != _REST_LABEL),
        )

        kept_onsets.append(start)
        kept_labels.append(label)

    events_new = np.zeros((len(kept_onsets), 3), dtype="int32")
    if len(kept_onsets) > 0:
        events_new[:, 0] = np.asarray(kept_onsets, dtype="int32")
        events_new[:, 2] = np.asarray(kept_labels, dtype="int32")

    return events_new


class ForkPipelines(TransformerMixin, BaseEstimator):
    def __init__(self, transformers: List[Tuple[str, Union[Pipeline, TransformerMixin]]]):
        for _, t in transformers:
            assert hasattr(t, "transform")
        self.transformers = transformers
        self._is_fitted = True

    def transform(self, X, y=None):
        return OrderedDict([(n, t.transform(X)) for n, t in self.transformers])

    def fit(self, X, y=None):
        for _, t in self.transformers:
            t.fit(X)
        return self

    def __sklearn_is_fitted__(self):
        """Return True to indicate this transformer is always considered fitted."""
        return True

    def _sk_visual_block_(self):
        """Tell sklearn's diagrammer to lay us out in parallel."""
        if _VisualBlock is None:
            return NotImplemented
        names, estimators = zip(*self.transformers)
        return _VisualBlock(
            kind="parallel",
            estimators=list(estimators),
            names=list(names),
            name_caption=self.__class__.__name__,
            dash_wrapped=True,
        )


class FixedTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self._is_fitted = True
        # fixing transformers that are not fitted
        # to avoid the warning "This estimator has not been fitted yet"
        # when using the pipeline

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        """Return True to indicate this transformer is always considered fitted."""
        return True

    def _sk_visual_block_(self):
        """Tell sklearn's diagrammer to lay us out in parallel."""
        if _VisualBlock is None:
            return NotImplemented
        return _VisualBlock(
            kind="parallel",
            name_caption=str(self.__class__.__name__),
            estimators=[str(self.get_params())],
            name_details=str(self.__class__.__name__),
            dash_wrapped=True,
        )


def _get_event_id_values(event_id):
    event_id_values = list(event_id.values())
    if len(event_id_values) == 0:
        return []
    arrays = [np.atleast_1d(val) for val in event_id_values]
    return np.concatenate(arrays).tolist()


def _compute_events_desc(event_id):
    ret = {}
    for ev in event_id:
        codes = event_id[ev]
        if not isinstance(codes, list):
            codes = [codes]
        for code in codes:
            ret[code] = ev
    return ret


class SetRawAnnotations(FixedTransformer):
    """
    Always sets the annotations, even if the events list is empty
    """

    def __init__(self, event_id, interval: Tuple[float, float]):
        super().__init__()
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id
        values = _get_event_id_values(self.event_id)
        if len(np.unique(values)) != len(values):
            raise ValueError("Duplicate event code")
        self.event_desc = _compute_events_desc(self.event_id)
        self.interval = interval

    def transform(self, raw, y=None):
        duration = self.interval[1] - self.interval[0]
        offset = int(self.interval[0] * raw.info["sfreq"])
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        has_annotation_extras = False
        if len(stim_channels) == 0:
            if raw.annotations is None:
                log.warning(
                    "No stim channel nor annotations found, skipping setting annotations."
                )
                return raw
            if not all(isinstance(mrk, int) for mrk in self.event_id.values()):
                raise ValueError(
                    "When no stim channel is present, event_id values must be integers (not lists)."
                )
            # Build lookup of extras by sample position before events extraction
            # destroys the original annotations
            orig_extras = getattr(raw.annotations, "extras", None)
            if orig_extras is not None and any(orig_extras):
                has_annotation_extras = True
                sfreq = raw.info["sfreq"]
                extras_by_sample = {}
                for ann, extra in zip(raw.annotations, orig_extras):
                    sample = int(round(ann["onset"] * sfreq)) + raw.first_samp
                    extras_by_sample[sample] = extra

            events, _ = mne.events_from_annotations(
                raw, event_id=self.event_id, verbose=False
            )
        else:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        events = _unsafe_pick_events(events, include=_get_event_id_values(self.event_id))
        events[:, 0] += offset
        if len(events) != 0:
            annotations = mne.annotations_from_events(
                events,
                raw.info["sfreq"],
                self.event_desc,
                first_samp=raw.first_samp,
                verbose=False,
            )
            annotations.set_durations(duration)

            # Transfer extras from original annotations to new ones
            if has_annotation_extras:
                new_extras = []
                for event in events:
                    orig_sample = event[0] - offset
                    new_extras.append(extras_by_sample.get(orig_sample, {}))
                annotations.extras = new_extras

            raw.set_annotations(annotations)
        else:
            log.warning("No events found, skipping setting annotations.")
        return raw


class RawToEvents(FixedTransformer):
    """
    Always returns an array for shape (n_events, 3), even if no events found.

    When ``overlap`` and ``window_length`` are provided, generates overlapping
    sliding window events from the original trial events instead.

    Parameters
    ----------
    event_id : dict
        Mapping of event names to codes.
    interval : tuple of (float, float)
        Dataset interval.
    overlap : float | None
        Overlap percentage (0-100). If None, no sliding window is applied.
    window_length : float | None
        Window length in seconds. Required when overlap is not None.
    """

    def __init__(
        self,
        event_id: dict[str, int],
        interval: Tuple[float, float],
        overlap=None,
        window_length=None,
        tmin=0.0,
    ):
        super().__init__()
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id
        self.interval = interval
        self.overlap = overlap
        self.window_length = window_length
        self.tmin = tmin
        if self.overlap is not None and self.window_length is None:
            raise ValueError("window_length must be provided when overlap is set")
        if self.overlap is not None:
            try:
                overlap_value = float(self.overlap)
            except (TypeError, ValueError):
                raise TypeError(
                    f"overlap must be a number in [0, 100), got {self.overlap!r}"
                )
            if not (0.0 <= overlap_value < 100.0):
                raise ValueError(f"overlap must be in [0, 100), got {self.overlap!r}")
            self.overlap = overlap_value

    def _find_events(self, raw):
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            # returns empty array if none found
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            try:
                events, _ = mne.events_from_annotations(
                    raw, event_id=self.event_id, verbose=False
                )
                offset = int(self.interval[0] * raw.info["sfreq"])
                events[:, 0] -= offset  # return the original events onset
            except ValueError as e:
                if str(e) == "Could not find any of the events you specified.":
                    return np.zeros((0, 3), dtype="int32")
                raise e
        return events

    def transform(self, raw, y=None):
        events = self._find_events(raw)
        events = _unsafe_pick_events(events, list(self.event_id.values()))
        if self.overlap is not None and len(events) >= 1:
            sfreq = raw.info["sfreq"]
            events = _generate_sliding_window_events(
                events,
                self.window_length,
                self.overlap,
                sfreq,
                self.interval,
                tmin=self.tmin,
            )
            events = _unsafe_pick_events(events, list(self.event_id.values()))
        return events


class RawToEventsP300(RawToEvents):
    def __init__(self, event_id, interval, ignore_relabelling=False):
        self.ignore_relabelling = ignore_relabelling
        super().__init__(event_id, interval)

    def transform(self, raw, y=None):
        events = self._find_events(raw)
        event_id = self.event_id
        if (
            not self.ignore_relabelling
            and "Target" in event_id
            and "NonTarget" in event_id
            and isinstance(event_id["Target"], list)
            and isinstance(event_id["NonTarget"], list)
        ):
            event_id_new = dict(Target=1, NonTarget=0)
            events = mne.merge_events(events, event_id["Target"], 1)
            events = mne.merge_events(events, event_id["NonTarget"], 0)
            event_id = event_id_new
        ret = _unsafe_pick_events(events, _get_event_id_values(self.event_id))
        return ret


class RawToFixedIntervalEvents(FixedTransformer):
    def __init__(
        self,
        length,
        stride,
        start_offset,
        stop_offset,
        marker=1,
    ):
        super().__init__()
        self.length = length
        self.stride = stride
        self.start_offset = start_offset
        self.stop_offset = stop_offset
        self.marker = marker

    def transform(self, raw: mne.io.BaseRaw, y=None):
        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError
        sfreq = raw.info["sfreq"]
        length_samples = int(self.length * sfreq)
        stride_samples = int(self.stride * sfreq)
        start_offset_samples = int(self.start_offset * sfreq)
        stop_offset_samples = (
            raw.n_times if self.stop_offset is None else int(self.stop_offset * sfreq)
        )
        stop_samples = stop_offset_samples - length_samples + raw.first_samp
        onset = np.arange(
            raw.first_samp + start_offset_samples,
            stop_samples,
            stride_samples,
        )
        if len(onset) == 0:
            # skip raw if no event found
            return
        events = np.empty((len(onset), 3), dtype=int)
        events[:, 0] = onset
        events[:, 1] = length_samples
        events[:, 2] = self.marker
        return events


class EpochsToEvents(FixedTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, epochs, y=None):
        return epochs.events


class EventsToLabels(FixedTransformer):
    def __init__(self, event_id):
        super().__init__()
        self.event_id = event_id

    def transform(self, events, y=None):
        inv_events = _compute_events_desc(self.event_id)
        labels = [inv_events[e] for e in events[:, -1]]
        return labels


class RawToEpochs(FixedTransformer):
    def __init__(
        self,
        event_id: Dict[str, int],
        tmin: float,
        tmax: float,
        baseline: Tuple[float, float],
        channels: List[str] = None,
        interpolate_missing_channels: bool = False,
    ):
        super().__init__()
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.channels = channels
        self.interpolate_missing_channels = interpolate_missing_channels

    def transform(self, X, y=None):
        raw = X["raw"]
        events = X["events"]
        if len(events) == 0:
            raise ValueError("No events found")
        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("raw must be a mne.io.BaseRaw")

        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            available_channels = raw.info["ch_names"]
            if self.interpolate_missing_channels:
                missing_channels = list(set(self.channels).difference(available_channels))

                # add missing channels (contains only zeros by default)
                try:
                    raw.add_reference_channels(missing_channels)
                except IndexError:
                    # Index error can occurs if the channels we add are not part of this epoch montage
                    # Then log a warning
                    montage = raw.info["dig"]
                    log.warning(
                        f"Montage disabled as one of these channels, {missing_channels}, is not part of the montage {montage}"
                    )
                    # and disable the montage
                    raw.info.pop("dig")
                    # run again with montage disabled
                    raw.add_reference_channels(missing_channels)

                # Trick: mark these channels as bad
                raw.info["bads"].extend(missing_channels)
                # ...and use mne bad channel interpolation to generate the value of the missing channels
                try:
                    raw.interpolate_bads(origin="auto")
                except ValueError:
                    # use default origin if montage info not available
                    raw.interpolate_bads(origin=(0, 0, 0.04))
                # update the name of the available channels
                available_channels = self.channels

            picks = mne.pick_channels(
                available_channels, include=self.channels, ordered=True
            )
            assert len(picks) == len(self.channels)

        epochs = mne.Epochs(
            raw,
            events,
            event_id=_get_event_id_values(self.event_id),
            tmin=self.tmin,
            tmax=self.tmax,
            proj=False,
            baseline=self.baseline,
            preload=True,
            verbose=False,
            picks=picks,
            event_repeated="drop",
            on_missing="ignore",
        )
        return epochs


class NamedFunctionTransformer(FunctionTransformer):
    def __init__(self, func, *, display_name=None, validate=False, **kwargs):
        super().__init__(func=func, validate=validate, **kwargs)
        self.display_name = display_name
        self._display_name = display_name or getattr(func, "__name__", "<func>")
        self._kwargs = {"name": getattr(func, "__name__", "<func>")}

    def __repr__(self):
        return self._display_name

    def _sk_visual_block_(self):
        if _VisualBlock is None:
            return NotImplemented
        return _VisualBlock(
            kind="single",
            estimators=self,
            names=self._display_name,
            name_details=str(self._kwargs),
            dash_wrapped=False,
        )


def get_filter_pipeline(fmin, fmax):
    return NamedFunctionTransformer(
        func=methodcaller(
            "filter",
            l_freq=fmin,
            h_freq=fmax,
            method="iir",
            picks="data",
            verbose=False,
        ),
        display_name=f"Band Pass Filter ({fmin}–{fmax} Hz)",
    )


def get_crop_pipeline(tmin, tmax):
    return NamedFunctionTransformer(
        func=methodcaller(
            "crop",
            tmin=tmin,
            tmax=tmax,
            verbose=False,
        ),
        display_name=f"Crop ({tmin}–{tmax} s)",
    )


def get_resample_pipeline(sfreq):
    return NamedFunctionTransformer(
        func=methodcaller(
            "resample",
            sfreq=sfreq,
            verbose=False,
        ),
        display_name=f"Resample ({sfreq} Hz)",
    )
