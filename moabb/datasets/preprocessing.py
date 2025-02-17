import logging
from collections import OrderedDict
from operator import methodcaller
from typing import Dict, List, Tuple, Union
from warnings import warn

import mne
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline


log = logging.getLogger(__name__)


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


class ForkPipelines(TransformerMixin, BaseEstimator):
    def __init__(self, transformers: List[Tuple[str, Union[Pipeline, TransformerMixin]]]):
        for _, t in transformers:
            assert hasattr(t, "transform")
        self.transformers = transformers

    def transform(self, X, y=None):
        return OrderedDict([(n, t.transform(X)) for n, t in self.transformers])

    def fit(self, X, y=None):
        for _, t in self.transformers:
            t.fit(X)


class FixedTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        pass


class SetRawAnnotations(FixedTransformer):
    """
    Always sets the annotations, even if the events list is empty
    """

    def __init__(self, event_id, interval: Tuple[float, float]):
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id
        if len(set(event_id.values())) != len(event_id):
            raise ValueError("Duplicate event code")
        self.event_desc = dict((code, desc) for desc, code in self.event_id.items())
        self.interval = interval

    def transform(self, raw, y=None):
        duration = self.interval[1] - self.interval[0]
        offset = int(self.interval[0] * raw.info["sfreq"])
        if raw.annotations:
            return raw
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) == 0:
            log.warning(
                "No stim channel nor annotations found, skipping setting annotations."
            )
            return raw
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        events = _unsafe_pick_events(events, include=list(self.event_id.values()))
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
            raw.set_annotations(annotations)
        else:
            log.warning("No events found, skipping setting annotations.")
        return raw


class RawToEvents(FixedTransformer):
    """
    Always returns an array for shape (n_events, 3), even if no events found
    """

    def __init__(self, event_id: dict[str, int], interval: Tuple[float, float]):
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id
        self.interval = interval

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
        return _unsafe_pick_events(events, list(self.event_id.values()))


class RawToEventsP300(RawToEvents):
    def transform(self, raw, y=None):
        events = self._find_events(raw)
        event_id = self.event_id
        if (
            "Target" in event_id
            and "NonTarget" in event_id
            and isinstance(event_id["Target"], list)
            and isinstance(event_id["NonTarget"], list)
        ):
            event_id_new = dict(Target=1, NonTarget=0)
            events = mne.merge_events(events, event_id["Target"], 1)
            events = mne.merge_events(events, event_id["NonTarget"], 0)
            event_id = event_id_new
        return _unsafe_pick_events(events, list(event_id.values()))


class RawToFixedIntervalEvents(FixedTransformer):
    def __init__(
        self,
        length,
        stride,
        start_offset,
        stop_offset,
        marker=1,
    ):
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
    def transform(self, epochs, y=None):
        return epochs.events


class EventsToLabels(FixedTransformer):
    def __init__(self, event_id):
        self.event_id = event_id

    def transform(self, events, y=None):
        inv_events = {k: v for v, k in self.event_id.items()}
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
                    warn(
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
            event_id=self.event_id,
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
        warn(f"warnEpochs {epochs}")
        return epochs


def get_filter_pipeline(fmin, fmax):
    return FunctionTransformer(
        methodcaller(
            "filter",
            l_freq=fmin,
            h_freq=fmax,
            method="iir",
            picks="eeg",
            verbose=False,
        ),
    )


def get_crop_pipeline(tmin, tmax):
    return FunctionTransformer(
        methodcaller("crop", tmin=tmin, tmax=tmax, verbose=False),
    )


def get_resample_pipeline(sfreq):
    return FunctionTransformer(
        methodcaller("resample", sfreq=sfreq, verbose=False),
    )
