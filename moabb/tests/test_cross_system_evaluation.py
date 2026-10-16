"""Tests for the cross-system CrossSystemEvaluation."""

import numpy as np
import pandas as pd
import pytest
import sklearn.base
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn import config_context
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import FunctionTransformer, make_pipeline

from moabb.datasets.fake import FakeDataset
from moabb.evaluations import CrossSubjectMode, CrossSystemEvaluation
from moabb.paradigms import LeftRightImagery


_CAPTURE = []


class _SystemRecorder(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """A system-aware step that records the cross-system metadata it is routed."""

    def fit(self, X, y=None, systems=None, X_target_unlabeled=None):
        _CAPTURE.append(
            {"X": X, "systems": systems, "X_target_unlabeled": X_target_unlabeled}
        )
        return self

    def transform(self, X):
        return X


def _trial_variances(X):
    """One feature per trial, whatever its channels and time samples."""
    return np.array([[np.var(trial)] for trial in X])


@pytest.fixture(autouse=True)
def _clear_capture():
    _CAPTURE.clear()
    yield
    _CAPTURE.clear()


def _datasets(sfreqs=(128, 128), channels=(("C3", "Cz", "C4"), ("Cz", "C4", "Pz", "P4"))):
    """Two fake systems, by default with different channels."""
    return [
        FakeDataset(
            ["left_hand", "right_hand"],
            channels=channels[0],
            n_subjects=2,
            n_sessions=1,
            sfreq=sfreqs[0],
            seed=1,
        ),
        FakeDataset(
            ["left_hand", "right_hand"],
            channels=channels[1],
            n_subjects=3,
            n_sessions=2,
            sfreq=sfreqs[1],
            seed=2,
        ),
    ]


def _recorder_pipeline():
    with config_context(enable_metadata_routing=True):
        step = _SystemRecorder().set_fit_request(systems=True, X_target_unlabeled=True)
    return make_pipeline(step, FunctionTransformer(_trial_variances), LDA())


def _evaluation(tmp_path, datasets, **kwargs):
    kwargs.setdefault("paradigm", LeftRightImagery())
    kwargs.setdefault("overwrite", True)
    return CrossSystemEvaluation(datasets=datasets, hdf5_path=tmp_path, **kwargs)


def _source_and_target(capture, datasets):
    (source,) = np.unique(capture["systems"])
    (target,) = {dataset.code for dataset in datasets} - {source}
    return source, target


def _identity(X):
    return X


def test_trials_keep_the_channels_of_their_system(tmp_path):
    ds_a, ds_b = _datasets()
    evaluation = _evaluation(tmp_path, [ds_a, ds_b])
    X, y, metadata = evaluation._load_data(ds_a, {}, None, None)

    # Systems with different channels give an object array of trials.
    assert X.dtype == object and X.shape == (len(y),)
    n_channels = {ds_a.code: 3, ds_b.code: 4}
    for trial, system in zip(X, metadata["system"]):
        assert trial.shape == (n_channels[system], 3 * 128 + 1)

    results = evaluation.process({"rec": _recorder_pipeline()})
    # One row per target subject/session: 2 x 1 for A, 3 x 2 for B.
    assert results.groupby("dataset").size().to_dict() == {ds_a.code: 2, ds_b.code: 6}
    assert len(_CAPTURE) == 2
    for capture in _CAPTURE:
        source, _ = _source_and_target(capture, [ds_a, ds_b])
        # Sources are trained on their own trials, routed with their system.
        assert {trial.shape[0] for trial in capture["X"]} == {n_channels[source]}
        assert len(capture["systems"]) == len(capture["X"])
        # Default TRAIN mode: no target trial is routed.
        assert capture["X_target_unlabeled"] is None


def test_trials_keep_the_sampling_rate_of_their_system(tmp_path):
    datasets = _datasets(sfreqs=(128, 256), channels=(("C3", "Cz", "C4"),) * 2)
    evaluation = _evaluation(tmp_path, datasets)
    X, _, metadata = evaluation._load_data(datasets[0], {}, None, None)

    assert X.dtype == object
    n_times = {datasets[0].code: 3 * 128 + 1, datasets[1].code: 3 * 256 + 1}
    for trial, system in zip(X, metadata["system"]):
        assert trial.shape == (3, n_times[system])
    assert len(evaluation.process({"rec": _recorder_pipeline()})) == 8


def test_same_channels_and_sampling_rate_give_regular_array(tmp_path):
    datasets = _datasets(channels=(("C3", "Cz", "C4"),) * 2)
    evaluation = _evaluation(tmp_path, datasets)
    X, y, _ = evaluation._load_data(datasets[0], {}, None, None)
    assert X.dtype != object and X.shape == (len(y), 3, 3 * 128 + 1)

    # A standard pipeline then runs as is.
    results = evaluation.process(
        {"csp": make_pipeline(Covariances("oas"), CSP(2), LDA())}
    )
    assert len(results) == 8
    assert results["score"].between(0, 1).all()


def test_calibration_routes_target_slice(tmp_path):
    ds_a, ds_b = _datasets()
    results = _evaluation(
        tmp_path, [ds_a, ds_b], cs_mode=CrossSubjectMode.TRAIN_AND_TARGET_UNLABELED_20P
    ).process({"rec": _recorder_pipeline()})

    (capture,) = [
        c for c in _CAPTURE if _source_and_target(c, [ds_a, ds_b])[1] == ds_a.code
    ]
    X_target = capture["X_target_unlabeled"]
    assert X_target is not None and len(X_target) > 0
    # Target trials keep the channels of the target system.
    assert {trial.shape[0] for trial in X_target} == {3}
    # Calibration and scored target trials partition the target system.
    _, y_a, _ = LeftRightImagery().get_data(ds_a)
    scored = results.loc[results["dataset"] == ds_a.code, "samples_test"].sum()
    assert scored + len(X_target) == len(y_a)


def test_target_datasets_restricts_targets(tmp_path):
    ds_a, ds_b = _datasets()
    ds_c = FakeDataset(
        ["left_hand", "right_hand"],
        channels=("C3", "Pz"),
        n_subjects=2,
        n_sessions=1,
        seed=3,
    )
    results = _evaluation(tmp_path, [ds_a, ds_b, ds_c], target_datasets=[ds_c]).process(
        {"rec": _recorder_pipeline()}
    )

    assert set(results["dataset"]) == {ds_c.code}
    assert len(results) == 2
    (capture,) = _CAPTURE
    # The non-target systems are the sources.
    assert set(capture["systems"]) == {ds_a.code, ds_b.code}


def test_trialwise_mode(tmp_path):
    evaluation = _evaluation(
        tmp_path, _datasets(), cs_mode=CrossSubjectMode.TRAIN_TRIALWISE
    )
    assert len(evaluation.process({"rec": _recorder_pipeline()})) == 8


def test_results_are_cached(tmp_path):
    datasets = _datasets()
    pipelines = {"rec": _recorder_pipeline()}
    first = _evaluation(tmp_path, datasets).process(pipelines)
    n_fits = len(_CAPTURE)

    cached = _evaluation(tmp_path, datasets, overwrite=False).process(pipelines)

    assert len(_CAPTURE) == n_fits
    keys = ["dataset", "subject", "session"]
    pd.testing.assert_frame_equal(
        first.sort_values(keys).reset_index(drop=True),
        cached.sort_values(keys).reset_index(drop=True),
        check_like=True,
    )


def test_evaluate_single_target(tmp_path):
    ds_a, ds_b = _datasets()
    evaluation = _evaluation(tmp_path, [ds_a, ds_b])
    process_pipeline = evaluation.paradigm.make_process_pipelines(ds_b)[0]
    results = list(
        evaluation.evaluate(
            ds_b,
            {"rec": _recorder_pipeline()},
            param_grid=None,
            process_pipeline=process_pipeline,
        )
    )
    assert len(results) == 6
    assert {res["dataset"].code for res in results} == {ds_b.code}


def test_rejects_postprocess_pipeline(tmp_path):
    evaluation = _evaluation(tmp_path, _datasets())
    with pytest.raises(ValueError, match="postprocess_pipeline"):
        evaluation.process(
            {"rec": _recorder_pipeline()},
            postprocess_pipeline=FunctionTransformer(_identity),
        )
