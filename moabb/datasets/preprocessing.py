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
    def __init__(self, event_id):
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id
        if len(set(event_id.values())) != len(event_id):
            raise ValueError("Duplicate event code")
        self.event_desc = dict((code, desc) for desc, code in self.event_id.items())

    def transform(self, raw, y=None):
        if raw.annotations:
            return raw
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) == 0:
            raise ValueError("Need either a stim channel or annotations")
        events = mne.find_events(raw, shortest_event=0, verbose=False)
        # we don't catch the error if no event found:
        events = mne.pick_events(events, include=list(self.event_id.values()))
        annotations = mne.annotations_from_events(
            events,
            raw.info["sfreq"],
            self.event_desc,
            first_samp=raw.first_samp,
            verbose=False,
        )
        raw.set_annotations(annotations)
        return raw


class RawToEvents(FixedTransformer):
    def __init__(self, event_id):
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id

    def transform(self, raw, y=None):
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events, _ = mne.events_from_annotations(
                raw, event_id=self.event_id, verbose=False
            )
        try:
            events = mne.pick_events(events, include=list(self.event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return
        return events


class RawToEventsP300(FixedTransformer):
    def __init__(self, event_id):
        assert isinstance(event_id, dict)  # not None
        self.event_id = event_id

    def transform(self, raw, y=None):
        event_id = self.event_id
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        try:
            if "Target" in event_id and "NonTarget" in event_id:
                if (
                    type(event_id["Target"]) is list
                    and type(event_id["NonTarget"]) == list
                ):
                    event_id_new = dict(Target=1, NonTarget=0)
                    events = mne.merge_events(events, event_id["Target"], 1)
                    events = mne.merge_events(events, event_id["NonTarget"], 0)
                    event_id = event_id_new
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return
        return events


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
        interpolate_missing_channels: bool = False
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
        if events is None or len(events) == 0:
            raise ValueError("No events found")
        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("raw must be a mne.io.BaseRaw")

        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            raw = raw.copy()
            available_channels = raw.info["ch_names"]
            if self.interpolate_missing_channels:
                missing_channels = list(set(self.channels).difference(available_channels))

                # add missing channels (contains only zeros by default)
                try:
                    raw.add_reference_channels(missing_channels)
                except IndexError:
                    # Index error can occurs if the channels we add are not part of this epoch montage
                    # Then log a warning
                    montage = raw.info['dig']
                    warn(f'Montage disabled as one of these channels, {missing_channels}, is not part of the montage {montage}')
                    # and disable the montage
                    raw.info.pop('dig')
                    # run again with montage disabled
                    raw.add_reference_channels(missing_channels)

                # Trick: mark these channels as bad
                raw.info['bads'].extend(missing_channels)
                # ...and use mne bad channel interpolation to generate the value of the missing channels
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
        methodcaller("crop", tmin=tmax, tmax=tmin, verbose=False),
    )


def get_resample_pipeline(sfreq):
    return FunctionTransformer(
        methodcaller("resample", sfreq=sfreq, verbose=False),
    )
