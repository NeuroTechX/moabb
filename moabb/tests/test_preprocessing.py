"""Tests for moabb.datasets.preprocessing module."""

from unittest.mock import patch

import mne
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer, Pipeline

from moabb.datasets.bids_interface import StepType
from moabb.datasets.preprocessing import (
    _REST_LABEL,
    EpochsToEvents,
    EventsToLabels,
    FixedPipeline,
    FixedTransformer,
    ForkPipelines,
    NamedFunctionTransformer,
    RawToEpochs,
    RawToEvents,
    RawToEventsP300,
    RawToFixedIntervalEvents,
    SetRawAnnotations,
    _compute_events_desc,
    _generate_sliding_window_events,
    _get_event_id_values,
    _insert_rest_events,
    _is_none_pipeline,
    _unsafe_pick_events,
    get_crop_pipeline,
    get_filter_pipeline,
    get_resample_pipeline,
    make_fixed_pipeline,
)


R, E, A = StepType.RAW, StepType.EPOCHS, StepType.ARRAY


# ── Helpers ───────────────────────────────────────────────────────────────────


class _Dummy(TransformerMixin, BaseEstimator):
    def __init__(self, name="dummy"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


def _pipe(*step_types):
    return FixedPipeline(
        [(st, _Dummy(f"{st.value}_{i}")) for i, st in enumerate(step_types)]
    )


def _raw(sfreq=250, n_channels=3, duration=10.0, stim_events=None, ch_names=None):
    if ch_names is None:
        ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
    ch_types = ["eeg"] * len(ch_names)
    if stim_events is not None:
        ch_names = [*ch_names, "STI 014"]
        ch_types.append("stim")
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.RandomState(42).randn(len(ch_names), int(sfreq * duration)) * 1e-6
    if stim_events is not None:
        data[-1, :] = 0
        for sample, label in stim_events:
            if sample < data.shape[1]:
                data[-1, sample] = label
    return mne.io.RawArray(data, info, verbose=False)


def _ann_raw(descriptions, onsets, event_id):
    """Raw with annotations only (no stim channel)."""
    raw = _raw(stim_events=None)
    raw.set_annotations(
        mne.Annotations(
            onset=onsets, duration=[0.0] * len(onsets), description=descriptions
        )
    )
    return raw


# ── FixedPipeline & make_fixed_pipeline ───────────────────────────────────────


def test_fixed_pipeline_is_fitted():
    assert _pipe(R).__sklearn_is_fitted__() is True


@pytest.mark.parametrize("n_steps", [1, 3])
def test_make_fixed_pipeline(n_steps):
    pipe = make_fixed_pipeline(*[_Dummy() for _ in range(n_steps)])
    assert isinstance(pipe, FixedPipeline)
    assert len(pipe.steps) == n_steps


# ── find_steps ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "steps, search, expected_indices",
    [((R, E, R), R, [0, 2]), ((R, E, A), E, [1]), ((R, E), A, [])],
    ids=["multiple", "single", "none"],
)
def test_find_steps(steps, search, expected_indices):
    result = _pipe(*steps).find_steps(search)
    assert [i for i, _ in result] == expected_indices


def test_find_steps_returns_correct_transformers():
    pipe = _pipe(R, E)
    assert pipe.find_steps(E)[0][1] is pipe.steps[1][1]


# ── insert_step ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "steps, kwargs, expected_pos",
    [
        ((R, E, A), {"after": E}, 2),
        ((R, E, E, A), {"after": E}, 3),
        ((R, E, A), {"before": E}, 1),
        ((R, E, E, A), {"before": E}, 1),
        ((R, E), {"index": 0}, 0),
        ((R, E), {"index": 2}, 2),
    ],
    ids=["after", "after-last", "before", "before-first", "index-0", "index-end"],
)
def test_insert_step(steps, kwargs, expected_pos):
    pipe = _pipe(*steps)
    t = _Dummy("new")
    result = pipe.insert_step(A, t, **kwargs)
    assert pipe.steps[expected_pos] == (A, t)
    assert len(pipe.steps) == len(steps) + 1
    assert result is pipe  # chaining


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"after": A}, "No steps of type"),
        ({"before": A}, "No steps of type"),
        ({}, "Exactly one"),
        ({"after": R, "before": E}, "Exactly one"),
    ],
    ids=["after-missing", "before-missing", "no-arg", "multi-arg"],
)
def test_insert_step_errors(kwargs, match):
    pipe = _pipe(R, E)
    with pytest.raises(ValueError, match=match):
        pipe.insert_step(R, _Dummy(), **kwargs)


# ── remove_step ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "steps, kwargs, remaining_types",
    [
        ((R, E, A), {"index": 1}, [R, A]),
        ((R, E, A), {"step_type": E}, [R, A]),
        ((R, E, R, A), {"step_type": R}, [E, A]),
    ],
    ids=["by-index", "by-type-single", "by-type-multi"],
)
def test_remove_step(steps, kwargs, remaining_types):
    pipe = _pipe(*steps)
    result = pipe.remove_step(**kwargs)
    assert [st for st, _ in pipe.steps] == remaining_types
    assert result is pipe  # chaining


@pytest.mark.parametrize(
    "steps, kwargs, match",
    [
        ((R, E), {"step_type": A}, "No steps of type"),
        ((R, R), {"step_type": R}, "Cannot remove all"),
        ((R,), {"index": 0}, "Cannot remove all"),
        ((R, E), {}, "Exactly one"),
        ((R, E), {"index": 0, "step_type": R}, "Exactly one"),
    ],
    ids=["not-found", "empty-by-type", "empty-by-index", "no-arg", "multi-arg"],
)
def test_remove_step_errors(steps, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _pipe(*steps).remove_step(**kwargs)


def test_remove_then_insert_chaining():
    pipe = _pipe(R, E, A)
    t = _Dummy("new")
    pipe.remove_step(index=1).insert_step(E, t, index=1)
    assert pipe.steps[1] == (E, t) and len(pipe.steps) == 3


# ── _is_none_pipeline ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "obj, expected",
    [
        (Pipeline([("none", None)]), True),
        (Pipeline([("t", _Dummy())]), False),
        ("not a pipeline", False),
        (Pipeline([("none", None), ("t", _Dummy())]), False),
    ],
    ids=["none", "non-none", "not-pipeline", "multi-step"],
)
def test_is_none_pipeline(obj, expected):
    assert _is_none_pipeline(obj) is expected


# ── _unsafe_pick_events ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "labels, include, expected_count",
    [([1, 2, 1], [1], 2), ([1, 2], [99], 0), ([5, 5], [5], 2)],
    ids=["partial", "none", "all"],
)
def test_unsafe_pick_events(labels, include, expected_count):
    events = np.array([[i * 100, 0, lab] for i, lab in enumerate(labels)], dtype="int32")
    assert _unsafe_pick_events(events, include).shape[0] == expected_count


def test_unsafe_pick_events_reraises():
    with patch(
        "moabb.datasets.preprocessing.mne.pick_events", side_effect=RuntimeError("other")
    ):
        with pytest.raises(RuntimeError, match="other"):
            _unsafe_pick_events(np.array([[0, 0, 1]], dtype="int32"), [1])


# ── _insert_rest_events ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "onsets, task_dur, interval_start, min_rest",
    [
        ([0, 1000], 500, 0, 1),
        ([0, 500], 500, 0, 1),
        ([0, 2000], 500, 100, 1),
        ([0], 500, 0, 1),
        ([0, 50, 100], 500, 0, 1),
    ],
    ids=["gap", "contiguous", "interval-start", "single", "overlapping"],
)
def test_insert_rest_events(onsets, task_dur, interval_start, min_rest):
    events = np.array([[o, 0, i + 1] for i, o in enumerate(onsets)], dtype="int32")
    result = _insert_rest_events(events, task_dur, interval_start)
    assert np.sum(result[:, 2] == _REST_LABEL) >= min_rest
    assert np.all(result[:-1, 0] <= result[1:, 0])


# ── _generate_sliding_window_events ──────────────────────────────────────────


def test_sliding_window_empty():
    assert _generate_sliding_window_events(
        np.zeros((0, 3), dtype="int32"), 1.0, 50, 250, (0, 4)
    ).shape == (0, 3)


def test_sliding_window_zero_length_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        _generate_sliding_window_events(
            np.array([[0, 0, 1]], dtype="int32"), 0, 50, 250, (0, 4)
        )


@pytest.mark.parametrize(
    "events, wl, overlap, interval, tmin",
    [
        (np.array([[0, 0, 1]], dtype="int32"), 1.0, 50, (0, 4), 0.0),
        (np.array([[0, 0, 1]], dtype="int32"), 1.0, 0, (0, 4), 0.0),
        (np.array([[0, 0, 1], [2000, 0, 2]], dtype="int32"), 1.0, 50, (0, 4), 0.0),
        (np.array([[0, 0, 1], [2000, 0, 2]], dtype="int32"), 1.0, 50, (0, 4), -0.5),
        (np.array([[0, 0, 1], [3000, 0, 2]], dtype="int32"), 1.0, 50, (2, 6), 0.0),
        (np.array([[500, 0, 1]], dtype="int32"), 1.0, 0, (0, 4), -3.0),
    ],
    ids=["basic", "no-overlap", "multi", "tmin", "nonzero-interval", "vote-before-first"],
)
def test_sliding_window_produces_events(events, wl, overlap, interval, tmin):
    result = _generate_sliding_window_events(events, wl, overlap, 250, interval, tmin)
    assert len(result) > 0


def test_sliding_window_max_start_too_small():
    events = np.array([[1000, 0, 1]], dtype="int32")
    assert _generate_sliding_window_events(events, 10.0, 0, 250, (0, 1)).shape == (0, 3)


# ── _get_event_id_values & _compute_events_desc ──────────────────────────────


@pytest.mark.parametrize(
    "event_id, expected", [({"left": 1, "right": 2}, [1, 2]), ({}, []), ({"a": 5}, [5])]
)
def test_get_event_id_values(event_id, expected):
    assert _get_event_id_values(event_id) == expected


def test_get_event_id_values_list():
    assert sorted(_get_event_id_values({"t": [1, 2], "n": [3]})) == [1, 2, 3]


@pytest.mark.parametrize(
    "event_id, expected",
    [
        ({"left": 1, "right": 2}, {1: "left", 2: "right"}),
        ({"t": [1, 2], "n": [3]}, {1: "t", 2: "t", 3: "n"}),
        ({}, {}),
    ],
)
def test_compute_events_desc(event_id, expected):
    assert _compute_events_desc(event_id) == expected


# ── ForkPipelines ─────────────────────────────────────────────────────────────


def test_fork_pipelines():
    fork = ForkPipelines([("a", _Dummy()), ("b", _Dummy())])
    result = fork.transform("X")
    assert list(result.keys()) == ["a", "b"] and result["a"] == "X"
    assert fork.fit("X") is fork
    assert fork.__sklearn_is_fitted__() is True
    assert fork._sk_visual_block_() is not NotImplemented


# ── FixedTransformer ──────────────────────────────────────────────────────────


def test_fixed_transformer():
    t = FixedTransformer()
    assert t._is_fitted is True
    assert t.fit("X") is t
    assert t.__sklearn_is_fitted__() is True
    assert t._sk_visual_block_() is not NotImplemented


# ── SetRawAnnotations ─────────────────────────────────────────────────────────


def test_set_raw_annotations_duplicate_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        SetRawAnnotations(event_id={"a": 1, "b": 1}, interval=(0, 4))


@pytest.mark.parametrize(
    "stim, event_id, n_ann",
    [([(500, 1), (1500, 2)], {"left": 1, "right": 2}, 2), ([(500, 99)], {"left": 1}, 0)],
    ids=["found", "no-match"],
)
def test_set_raw_annotations_stim(stim, event_id, n_ann):
    raw = _raw(stim_events=stim)
    SetRawAnnotations(event_id=event_id, interval=(0, 4)).transform(raw)
    if n_ann:
        assert len(raw.annotations) == n_ann


def test_set_raw_annotations_from_annotations():
    raw = _ann_raw(["left", "right"], [1.0, 3.0], {"left": 1, "right": 2})
    SetRawAnnotations(event_id={"left": 1, "right": 2}, interval=(0, 4)).transform(raw)


def test_set_raw_annotations_none_annotations():
    raw = _raw(stim_events=None)
    raw._annotations = None
    result = SetRawAnnotations(event_id={"left": 1}, interval=(0, 4)).transform(raw)
    assert result is raw


def test_set_raw_annotations_non_int_raises():
    raw = _ann_raw(["left"], [1.0], {"left": [1, 2]})
    with pytest.raises(ValueError, match="integers"):
        SetRawAnnotations(event_id={"left": [1, 2]}, interval=(0, 4)).transform(raw)


def test_set_raw_annotations_extras():
    raw = _raw(stim_events=None)
    ann = mne.Annotations(
        onset=[1.0, 3.0], duration=[0.0, 0.0], description=["left", "right"]
    )
    ann.extras = [{"s": "1"}, {"s": "2"}]
    raw.set_annotations(ann)
    SetRawAnnotations(event_id={"left": 1, "right": 2}, interval=(0, 4)).transform(raw)
    assert len(raw.annotations) >= 1


# ── RawToEvents ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overlap, wl, error, match",
    [
        (50, None, ValueError, "window_length"),
        ("bad", 1.0, TypeError, "must be a number"),
        (100, 1.0, ValueError, "must be in"),
        (-1, 1.0, ValueError, "must be in"),
    ],
)
def test_raw_to_events_init_errors(overlap, wl, error, match):
    with pytest.raises(error, match=match):
        RawToEvents(event_id={"a": 1}, interval=(0, 4), overlap=overlap, window_length=wl)


def test_raw_to_events_valid_overlap():
    rte = RawToEvents(event_id={"a": 1}, interval=(0, 4), overlap=50, window_length=1.0)
    assert rte.overlap == 50.0


@pytest.mark.parametrize(
    "stim, event_id, use_overlap, min_events",
    [
        ([(500, 1), (1500, 2)], {"l": 1, "r": 2}, False, 2),
        ([(500, 99)], {"l": 1}, False, 0),
        ([(500, 1), (1500, 2)], {"l": 1, "r": 2}, True, 2),
    ],
    ids=["stim-match", "stim-no-match", "with-overlap"],
)
def test_raw_to_events_stim(stim, event_id, use_overlap, min_events):
    raw = _raw(stim_events=stim)
    kw = {"overlap": 50, "window_length": 1.0} if use_overlap else {}
    events = RawToEvents(event_id=event_id, interval=(0, 4), **kw).transform(raw)
    assert events.shape[1] == 3 and len(events) >= min_events


@pytest.mark.parametrize(
    "descs, event_id, n_expected",
    [(["left", "right"], {"left": 1, "right": 2}, 2), (["other"], {"left": 1}, 0)],
    ids=["match", "no-match"],
)
def test_raw_to_events_annotations(descs, event_id, n_expected):
    raw = _ann_raw(descs, [float(i + 1) for i in range(len(descs))], event_id)
    events = RawToEvents(event_id=event_id, interval=(0, 4)).transform(raw)
    assert len(events) == n_expected


def test_raw_to_events_reraises():
    raw = _ann_raw(["left"], [1.0], {"left": 1})
    rte = RawToEvents(event_id={"left": 1}, interval=(0, 4))
    with patch(
        "moabb.datasets.preprocessing.mne.events_from_annotations",
        side_effect=ValueError("other"),
    ):
        with pytest.raises(ValueError, match="other"):
            rte.transform(raw)


# ── RawToEventsP300 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event_id, ignore, expected_labels",
    [
        ({"Target": 1, "NonTarget": 2}, False, {1, 2}),
        ({"Target": [1, 3], "NonTarget": [2, 4]}, False, {0, 1}),
        ({"Target": [1, 3], "NonTarget": [2, 4]}, True, {1, 2, 3, 4}),
    ],
    ids=["simple", "relabel", "ignore-relabel"],
)
def test_raw_to_events_p300(event_id, ignore, expected_labels):
    stim = [(500, 1), (750, 3), (1000, 2), (1250, 4)]
    raw = _raw(stim_events=stim)
    events = RawToEventsP300(
        event_id=event_id, interval=(0, 1), ignore_relabelling=ignore
    ).transform(raw)
    assert set(events[:, 2]).issubset(expected_labels)


# ── RawToFixedIntervalEvents ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "stop_offset, marker, expect_events",
    [(None, 1, True), (5.0, 1, True), (None, 42, True)],
    ids=["default", "stop-offset", "custom-marker"],
)
def test_raw_to_fixed_interval_events(stop_offset, marker, expect_events):
    raw = _raw(stim_events=[(0, 0)])
    t = RawToFixedIntervalEvents(
        length=1.0, stride=0.5, start_offset=0, stop_offset=stop_offset, marker=marker
    )
    events = t.transform(raw)
    assert events is not None and np.all(events[:, 2] == marker)


def test_raw_to_fixed_interval_not_raw_raises():
    with pytest.raises(ValueError):
        RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=None
        ).transform("x")


def test_raw_to_fixed_interval_no_events():
    assert (
        RawToFixedIntervalEvents(
            length=1.0, stride=0.5, start_offset=0, stop_offset=None
        ).transform(_raw(duration=0.1, stim_events=[(0, 0)]))
        is None
    )


# ── EpochsToEvents & EventsToLabels ──────────────────────────────────────────


def test_epochs_to_events():
    raw = _raw(stim_events=[(500, 1), (1500, 2)])
    ev = np.array([[500, 0, 1], [1500, 0, 2]], dtype="int32")
    epochs = mne.Epochs(
        raw,
        ev,
        event_id=[1, 2],
        tmin=0,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose=False,
    )
    assert np.array_equal(EpochsToEvents().transform(epochs), epochs.events)


@pytest.mark.parametrize(
    "event_id, codes, expected",
    [
        ({"left": 1, "right": 2}, [1, 2, 1], ["left", "right", "left"]),
        ({"t": [1, 3]}, [1, 3], ["t", "t"]),
    ],
)
def test_events_to_labels(event_id, codes, expected):
    events = np.array([[i * 100, 0, c] for i, c in enumerate(codes)], dtype="int32")
    assert EventsToLabels(event_id=event_id).transform(events) == expected


# ── RawToEpochs ──────────────────────────────────────────────────────────────


def test_raw_to_epochs_basic():
    raw = _raw(stim_events=[(500, 1), (1500, 1)])
    ev = np.array([[500, 0, 1], [1500, 0, 1]], dtype="int32")
    result = RawToEpochs(event_id={"l": 1}, tmin=0, tmax=0.5, baseline=None).transform(
        {"raw": raw, "events": ev}
    )
    assert isinstance(result, mne.Epochs) and len(result) == 2


def test_raw_to_epochs_with_channels():
    raw = _raw(stim_events=[(500, 1)])
    ev = np.array([[500, 0, 1]], dtype="int32")
    result = RawToEpochs(
        event_id={"l": 1}, tmin=0, tmax=0.5, baseline=None, channels=["EEG1", "EEG2"]
    ).transform({"raw": raw, "events": ev})
    assert len(result.ch_names) == 2


@pytest.mark.parametrize(
    "raw_val, events, match",
    [
        ("not_raw", np.array([[0, 0, 1]], dtype="int32"), "raw must be"),
        (None, np.zeros((0, 3), dtype="int32"), "No events"),
    ],
    ids=["not-raw", "no-events"],
)
def test_raw_to_epochs_errors(raw_val, events, match):
    raw_obj = raw_val if raw_val == "not_raw" else _raw(stim_events=[(500, 1)])
    with pytest.raises(ValueError, match=match):
        RawToEpochs(event_id={"l": 1}, tmin=0, tmax=0.5, baseline=None).transform(
            {"raw": raw_obj, "events": events}
        )


def test_raw_to_epochs_interpolate_missing():
    n_samples = int(250 * 10.0)
    info = mne.create_info(
        ch_names=["C3", "Cz", "C4", "STI 014"], sfreq=250, ch_types=["eeg"] * 3 + ["stim"]
    )
    data = np.random.RandomState(42).randn(4, n_samples) * 1e-6
    data[-1, :] = 0
    data[-1, 500] = 1
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
    )
    ev = np.array([[500, 0, 1]], dtype="int32")
    result = RawToEpochs(
        event_id={"l": 1},
        tmin=0,
        tmax=0.5,
        baseline=None,
        channels=["C3", "Cz", "C4", "Fz"],
        interpolate_missing_channels=True,
    ).transform({"raw": raw, "events": ev})
    assert len(result.ch_names) == 4


# ── NamedFunctionTransformer & pipeline helpers ───────────────────────────────


@pytest.mark.parametrize(
    "display_name, expected_repr", [("My Transform", "My Transform"), (None, "my_func")]
)
def test_named_function_transformer(display_name, expected_repr):
    def my_func(x):
        return x

    t = NamedFunctionTransformer(my_func, display_name=display_name)
    assert repr(t) == expected_repr
    assert t._sk_visual_block_() is not NotImplemented


@pytest.mark.parametrize(
    "display_name, expected_name", [("My Transform", "My Transform"), (None, "my_func")]
)
def test_named_function_transformer_repr_html(display_name, expected_name):
    def my_func(x):
        return x

    t = NamedFunctionTransformer(my_func, display_name=display_name)
    html = t._repr_html_()
    assert expected_name in html
    assert "<style>" in html  # sklearn's full HTML repr includes CSS


@pytest.mark.parametrize(
    "factory, args, label",
    [
        (get_filter_pipeline, (8, 30), "Band Pass Filter"),
        (get_crop_pipeline, (0, 4), "Crop"),
        (get_resample_pipeline, (128,), "Resample"),
    ],
)
def test_pipeline_helpers(factory, args, label):
    t = factory(*args)
    assert isinstance(t, NamedFunctionTransformer) and label in repr(t)


def test_fixed_pipeline_repr_html_with_steptype_keys():
    """FixedPipeline._repr_html_ must work with StepType enum keys."""
    pipeline = FixedPipeline(
        steps=[
            (StepType.RAW, FunctionTransformer()),
            (StepType.EPOCHS, FunctionTransformer()),
        ]
    )
    html = pipeline._repr_html_()
    assert "<style>" in html
    assert "FunctionTransformer" in html
    # StepType enums must not leak into the HTML output
    assert "StepType.RAW" not in html
    assert "StepType.EPOCHS" not in html
