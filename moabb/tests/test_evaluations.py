import os
import os.path as osp
import platform
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import sklearn.base
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.dummy import DummyClassifier as Dummy
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline

from moabb.analysis.results import get_digest, get_string_rep
from moabb.datasets.compound_dataset import compound
from moabb.datasets.fake import FakeDataset
from moabb.evaluations import evaluations as ev
from moabb.evaluations.base import BaseEvaluation, optuna_available
from moabb.evaluations.splitters import LearningCurveSplitter
from moabb.evaluations.utils import _create_save_path as create_save_path
from moabb.evaluations.utils import _save_model_cv as save_model_cv
from moabb.paradigms.motor_imagery import FakeImageryParadigm


def _identity(x):
    """Identity function (replaces lambda to avoid MOABB hash warnings)."""
    return x


try:
    from codecarbon import EmissionsTracker  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


def _expected_result_key_count(extra_columns=0):
    """Return expected number of result-dict keys for this environment."""
    base_keys = 10  # includes n_samples_test and n_classes
    carbon_keys = 2 if _carbonfootprint else 0
    return base_keys + carbon_keys + extra_columns


pipelines = OrderedDict()
pipelines["C"] = make_pipeline(Covariances("oas"), CSP(8), LDA())
dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=2, seed=12)
if not osp.isdir(osp.join(osp.expanduser("~"), "mne_data")):
    os.makedirs(osp.join(osp.expanduser("~"), "mne_data"))


class DummyClassifier(sklearn.base.BaseEstimator):
    __slots__ = "kernel"

    def __init__(self, kernel):
        self.kernel = kernel


class LegacyOnlyEvaluation(BaseEvaluation):
    """Minimal evaluation that intentionally uses legacy process path."""

    def _create_splitter(self):
        return None

    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        for subject in dataset.subject_list:
            for name in pipelines:
                yield {
                    "time": 0.0,
                    "dataset": dataset,
                    "subject": subject,
                    "session": "0",
                    "score": 0.0,
                    "n_samples": 1,
                    "n_channels": 1,
                    "pipeline": name,
                }

    def is_valid(self, dataset):
        return True


class TestWithinSess:
    """This is actually integration testing but I don't know how to do this
    better.

    A paradigm implements pre-processing so it needs files to run MNE
    stuff on. To test the scoring and train/test we need to also have data and
    run it. Putting this on the future docket...
    """

    def setup_method(self):
        self.eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            hdf5_path="res_test",
            save_model=True,
            optuna=False,
        )

    def teardown_method(self):
        path = self.eval.results.filepath
        if os.path.isfile(path):
            os.remove(path)

    def test_mne_labels(self):
        kwargs = dict(paradigm=FakeImageryParadigm(), datasets=[dataset])
        epochs = dict(return_epochs=False, mne_labels=True)
        with pytest.raises(ValueError):
            ev.WithinSessionEvaluation(**epochs, **kwargs)

    def test_eval_results(self):
        process_pipeline = self.eval.paradigm.make_process_pipelines(dataset)[0]
        results = [
            r
            for r in self.eval.evaluate(
                dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
            )
        ]

        # We should get 4 results, 2 sessions 2 subjects
        assert len(results) == 4
        # We should have 9 columns in the results data frame
        assert len(results[0].keys()) == _expected_result_key_count()

    def test_compound_dataset(self):
        ch1 = ["C3", "Cz", "Fz"]
        dataset1 = FakeDataset(
            paradigm="imagery",
            event_list=["left_hand", "right_hand"],
            channels=ch1,
            sfreq=128,
        )
        ch2 = ["C3", "C4", "Cz"]
        dataset2 = FakeDataset(
            paradigm="imagery",
            event_list=["left_hand", "right_hand"],
            channels=ch2,
            sfreq=256,
        )
        merged_dataset = compound(dataset1, dataset2)

        # We want to interpolate channels that are not in common between the two datasets
        self.eval.paradigm.match_all(
            merged_dataset.datasets, channel_merge_strategy="union"
        )

        process_pipeline = self.eval.paradigm.make_process_pipelines(dataset)[0]
        results = [
            r
            for r in self.eval.evaluate(
                dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
            )
        ]

        # We should get 4 results, 2 sessions 2 subjects
        assert len(results) == 4
        # We should have 9 columns in the results data frame
        assert len(results[0].keys()) == _expected_result_key_count()

    def test_eval_grid_search(self):
        # Test grid search
        param_grid = {"C": {"csp__metric": ["euclid", "riemann"]}}
        process_pipeline = self.eval.paradigm.make_process_pipelines(dataset)[0]
        results = [
            r
            for r in self.eval.evaluate(
                dataset,
                pipelines,
                param_grid=param_grid,
                process_pipeline=process_pipeline,
            )
        ]

        # We should get 4 results, 2 sessions 2 subjects
        assert len(results) == 4
        # We should have 9 columns in the results data frame
        assert len(results[0].keys()) == _expected_result_key_count()

    def test_eval_grid_search_optuna(self):
        if not optuna_available:
            pytest.skip("Optuna is not available")

        # Test grid search
        param_grid = {"C": {"csp__metric": ["euclid", "riemann"]}}
        process_pipeline = self.eval.paradigm.make_process_pipelines(dataset)[0]

        self.eval.optuna = True

        results = [
            r
            for r in self.eval.evaluate(
                dataset,
                pipelines,
                param_grid=param_grid,
                process_pipeline=process_pipeline,
            )
        ]

        self.eval.optuna = False

        # We should get 4 results, 2 sessions 2 subjects
        assert len(results) == 4
        # We should have 9 columns in the results data frame
        assert len(results[0].keys()) == _expected_result_key_count()

    def test_within_session_evaluation_save_model(self):
        res_test_path = "./res_test"

        # Get a list of all subdirectories inside 'res_test'
        subdirectories = [
            d
            for d in os.listdir(res_test_path)
            if os.path.isdir(os.path.join(res_test_path, d))
        ]

        # Check if any of the subdirectories contain the partial name 'Model'
        model_folder_exists = any("Model" in folder for folder in subdirectories)

        # Assert that at least one folder with the partial name 'Model' exists
        assert model_folder_exists, (
            "No folder with partial name 'Model' found inside 'res_test' directory",
        )

    def test_lambda_warning(self):
        def explicit_kernel(x):
            return x**3

        c1 = DummyClassifier(kernel=lambda x: x**2)
        c2 = DummyClassifier(kernel=lambda x: 5 * x)

        c3 = DummyClassifier(kernel=explicit_kernel)

        assert repr(c1) != repr(c2)
        if platform.system() != "Windows":
            with pytest.warns(RuntimeWarning):
                assert get_string_rep(c1) == get_string_rep(c2)

        # I do not know an elegant way to check for no warnings
        with warnings.catch_warnings(record=True) as w:
            get_string_rep(c3)
            assert len(w) == 0

    def test_digest_distinguishes_strings_with_spaces(self):
        assert get_digest({"x": "a b"}) != get_digest({"x": "ab"})

    def test_postprocess_pipeline(self):
        cov = Covariances("oas")
        pipelines0 = {
            "CovCspLda": make_pipeline(
                cov,
                CSP(
                    8,
                ),
                LDA(),
            )
        }
        pipelines1 = {"CspLda": make_pipeline(CSP(8), LDA())}

        results0 = self.eval.process(pipelines0)
        results1 = self.eval.process(
            pipelines0, postprocess_pipeline=FunctionTransformer(_identity)
        )
        results2 = self.eval.process(pipelines1, postprocess_pipeline=cov)
        np.testing.assert_allclose(results0.score, results1.score)
        np.testing.assert_allclose(results0.score, results2.score)


class TestWithinSessLearningCurve:
    """Some tests for the learning curve evaluation.

    TODO if we ever extend dataset metadata, e.g. including y for
    example, we could get rid of a lot of issues regarding valid inputs
    for policy per_class as this could be determined at Evaluation
    initialization instead of during running the evaluation
    """

    def test_correct_results_integrity(self):
        learning_curve_eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            cv_class=LearningCurveSplitter,
            cv_kwargs={
                "data_size": {"policy": "ratio", "value": np.array([0.2, 0.5])},
                "n_perms": np.array([2, 2]),
            },
            overwrite=True,
        )
        process_pipeline = learning_curve_eval.paradigm.make_process_pipelines(dataset)[0]
        results = [
            r
            for r in learning_curve_eval.evaluate(
                dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
            )
        ]
        keys = results[0].keys()
        assert len(keys) == _expected_result_key_count(extra_columns=2)
        assert "permutation" in keys
        assert "data_size" in keys

    def test_all_policies_work(self):
        kwargs = dict(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            cv_class=LearningCurveSplitter,
        )
        # The next two should work without issue
        ev.WithinSessionEvaluation(
            cv_kwargs={
                "data_size": {"policy": "per_class", "value": [5, 10]},
                "n_perms": [2, 2],
            },
            **kwargs,
        )
        ev.WithinSessionEvaluation(
            cv_kwargs={
                "data_size": {"policy": "ratio", "value": [0.2, 0.5]},
                "n_perms": [2, 2],
            },
            **kwargs,
        )
        # Invalid policy should raise ValueError when evaluation is run
        # (LearningCurveSplitter is instantiated at evaluation time, not construction)
        evaluation = ev.WithinSessionEvaluation(
            cv_kwargs={
                "data_size": {"policy": "does_not_exist", "value": [0.2, 0.5]},
                "n_perms": [2, 2],
            },
            overwrite=True,
            **kwargs,
        )
        with pytest.raises(ValueError):
            evaluation.process(pipelines)

    def test_data_sanity(self):
        # need this helper to iterate over the generator
        def run_evaluation(eval, dataset, pipelines):
            process_pipeline = eval.paradigm.make_process_pipelines(dataset)[0]
            list(
                eval.evaluate(
                    dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
                )
            )

        # E.g. if number of samples too high -> expect error
        kwargs = dict(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            cv_class=LearningCurveSplitter,
            overwrite=True,
        )
        should_work = ev.WithinSessionEvaluation(
            cv_kwargs={
                "data_size": {"policy": "per_class", "value": [5, 10]},
                "n_perms": [2, 2],
            },
            **kwargs,
        )
        too_many_samples = ev.WithinSessionEvaluation(
            cv_kwargs={
                "data_size": {"policy": "per_class", "value": [5, 100000]},
                "n_perms": [2, 2],
            },
            **kwargs,
        )
        # This one should run
        run_evaluation(should_work, dataset, pipelines)
        with pytest.raises(ValueError):
            run_evaluation(too_many_samples, dataset, pipelines)

    def test_eval_grid_search(self):
        pass

    def test_datasize_parameters(self):
        # Fail if not values are not correctly ordered
        # (LearningCurveSplitter is instantiated at evaluation time, not construction)
        kwargs = dict(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            cv_class=LearningCurveSplitter,
            overwrite=True,
        )
        decreasing_datasize = dict(
            cv_kwargs={
                "data_size": {"policy": "per_class", "value": [5, 4]},
                "n_perms": [2, 1],
            },
            **kwargs,
        )
        constant_datasize = dict(
            cv_kwargs={
                "data_size": {"policy": "per_class", "value": [5, 5]},
                "n_perms": [2, 3],
            },
            **kwargs,
        )
        increasing_perms = dict(
            cv_kwargs={
                "data_size": {"policy": "per_class", "value": [3, 4]},
                "n_perms": [2, 3],
            },
            **kwargs,
        )
        with pytest.raises(ValueError):
            ev.WithinSessionEvaluation(**decreasing_datasize).process(pipelines)
        with pytest.raises(ValueError):
            ev.WithinSessionEvaluation(**constant_datasize).process(pipelines)
        with pytest.raises(ValueError):
            ev.WithinSessionEvaluation(**increasing_perms).process(pipelines)

    def test_postprocess_pipeline(self):
        learning_curve_eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            cv_class=LearningCurveSplitter,
            cv_kwargs={
                "data_size": {"policy": "ratio", "value": np.array([0.2, 0.5])},
                "n_perms": np.array([2, 2]),
            },
        )

        cov = Covariances("oas")
        pipelines0 = {
            "CovCspLda": make_pipeline(
                cov,
                CSP(
                    8,
                ),
                LDA(),
            )
        }
        pipelines1 = {"CspLda": make_pipeline(CSP(8), LDA())}

        results0 = learning_curve_eval.process(pipelines0)
        results1 = learning_curve_eval.process(
            pipelines0, postprocess_pipeline=FunctionTransformer(_identity)
        )
        results2 = learning_curve_eval.process(pipelines1, postprocess_pipeline=cov)
        np.testing.assert_allclose(results0.score, results1.score)
        np.testing.assert_allclose(results0.score, results2.score)


class TestWithinSubj(TestWithinSess):
    def setup_method(self):
        self.eval = ev.WithinSubjectEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            hdf5_path="res_test",
            save_model=True,
            optuna=False,
        )


class Test_CrossSubj(TestWithinSess):
    def setup_method(self):
        self.eval = ev.CrossSubjectEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            hdf5_path="res_test",
            save_model=True,
        )

    def test_compatible_dataset(self):
        # raise
        ds = FakeDataset(["left_hand", "right_hand"], n_subjects=1)
        assert not self.eval.is_valid(dataset=ds)

        # do not raise
        ds = FakeDataset(["left_hand", "right_hand"], n_subjects=2)
        assert self.eval.is_valid(dataset=ds)


class Test_CrossSess(TestWithinSess):
    def setup_method(self):
        self.eval = ev.CrossSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            hdf5_path="res_test",
            save_model=True,
        )

    def test_compatible_dataset(self):
        ds = FakeDataset(["left_hand", "right_hand"], n_sessions=1)
        assert not self.eval.is_valid(ds)

        # do not raise
        ds = FakeDataset(["left_hand", "right_hand"], n_sessions=2)
        assert self.eval.is_valid(dataset=ds)

    def test_incompatibility_error_message(self):
        """Test that incompatibility error message is clear and informative."""
        ds = FakeDataset(["left_hand", "right_hand"], n_sessions=1)
        # Test that the error message includes the dataset code and reason
        with pytest.raises(AssertionError) as exc_info:
            list(self.eval.evaluate(ds, pipelines, None, None))
        error_msg = str(exc_info.value)
        assert "CrossSessionEvaluation" in error_msg
        assert "1 session" in error_msg
        assert "requires at least 2 sessions" in error_msg


class TestUtilEvaluation:
    def test_save_model_cv(self):
        model = Dummy()
        save_path = "test_save_path"
        cv_index = 0

        save_model_cv(model, save_path, cv_index)

        # Assert that the saved model file exists
        assert os.path.isfile(os.path.join(save_path, "fitted_model_0.pkl"))

    def test_create_save_path(self):
        hdf5_path = "base_path"
        code = "evaluation_code"
        subject = 1
        session = "0"
        name = "evaluation_name"
        eval_type = "WithinSession"
        save_path = create_save_path(
            hdf5_path, code, subject, session, name, eval_type=eval_type
        )

        expected_path = os.path.join(
            hdf5_path, "Models_WithinSession", code, "1", "0", "evaluation_name"
        )
        assert save_path == expected_path

        grid_save_path = create_save_path(
            hdf5_path, code, subject, session, name, grid=True, eval_type=eval_type
        )

        expected_grid_path = os.path.join(
            hdf5_path,
            "GridSearch_WithinSession",
            code,
            "1",
            "0",
            "evaluation_name",
        )
        assert grid_save_path == expected_grid_path

    def test_save_model_cv_with_pytorch_model(self):
        try:
            import torch
            from skorch import NeuralNetClassifier
        except ImportError:
            pytest.skip("skorch library not available")

        step = NeuralNetClassifier(module=torch.nn.Linear(10, 2))
        step.initialize()
        model = Pipeline([("step", step)])
        save_path = "."
        cv_index = 0
        save_model_cv(model, save_path, cv_index)

        # Assert that the saved model files exist
        assert os.path.isfile(os.path.join(save_path, "step_fitted_0_model.pkl"))
        assert os.path.isfile(os.path.join(save_path, "step_fitted_0_optim.pkl"))
        assert os.path.isfile(os.path.join(save_path, "step_fitted_0_history.json"))
        assert os.path.isfile(os.path.join(save_path, "step_fitted_0_criterion.pkl"))

    def test_create_save_path_with_cross_session_evaluation(self):
        hdf5_path = "base_path"
        code = "evaluation_code"
        subject = 1
        session = "0"
        name = "evaluation_name"
        eval_type = "CrossSession"
        save_path = create_save_path(
            hdf5_path, code, subject, session, name, eval_type=eval_type
        )

        expected_path = os.path.join(
            hdf5_path, "Models_CrossSession", code, "1", "evaluation_name"
        )
        assert save_path == expected_path

        grid_save_path = create_save_path(
            hdf5_path, code, subject, session, name, grid=True, eval_type=eval_type
        )

        expected_grid_path = os.path.join(
            hdf5_path, "GridSearch_CrossSession", code, "1", "evaluation_name"
        )
        assert grid_save_path == expected_grid_path

    def test_create_save_path_without_hdf5_path(self):
        hdf5_path = None
        code = "evaluation_code"
        subject = 1
        session = "0"
        name = "evaluation_name"
        eval_type = "WithinSession"
        save_path = create_save_path(
            hdf5_path, code, subject, session, name, eval_type=eval_type
        )

        assert save_path is None

    def test_save_model_cv_without_hdf5_path(self):
        model = DummyClassifier(kernel="rbf")
        save_path = None
        cv_index = 0

        # Assert that calling save_model_cv without a save_path does raise an IOError
        with pytest.raises(IOError):
            save_model_cv(model, save_path, cv_index)

    def test_create_save_path_with_cross_subject_evaluation(self):
        hdf5_path = "base_path"
        code = "evaluation_code"
        subject = "1"
        session = ""
        name = "evaluation_name"
        eval_type = "CrossSubject"
        save_path = create_save_path(
            hdf5_path, code, subject, session, name, eval_type=eval_type
        )

        expected_path = os.path.join(
            hdf5_path, "Models_CrossSubject", code, "1", "evaluation_name"
        )
        assert save_path == expected_path

        grid_save_path = create_save_path(
            hdf5_path, code, subject, session, name, grid=True, eval_type=eval_type
        )

        expected_grid_path = os.path.join(
            hdf5_path, "GridSearch_CrossSubject", code, "1", "evaluation_name"
        )
        assert grid_save_path == expected_grid_path

    def test_create_save_path_without_hdf5_path_or_session(self):
        hdf5_path = None
        code = "evaluation_code"
        subject = 1
        session = ""
        name = "evaluation_name"
        eval_type = "WithinSession"
        save_path = create_save_path(
            hdf5_path, code, subject, session, name, eval_type=eval_type
        )

        assert save_path is None

        grid_save_path = create_save_path(
            hdf5_path, code, subject, session, name, grid=True, eval_type=eval_type
        )

        assert grid_save_path is None

    def test_create_save_path_with_special_characters(self):
        hdf5_path = "base_path"
        code = "evaluation_code"
        subject = 1
        session = "0"
        name = "evalu@tion#name"
        eval_type = "WithinSession"
        save_path = create_save_path(
            hdf5_path, code, subject, session, name, eval_type=eval_type
        )

        expected_path = os.path.join(
            hdf5_path, "Models_WithinSession", code, "1", "0", "evalu@tion#name"
        )
        assert save_path == expected_path


class TestBatchNotYetComputed:
    """Tests for Results.batch_not_yet_computed()."""

    def test_matches_per_subject_not_yet_computed(self, tmp_path):
        """batch_not_yet_computed() matches per-subject not_yet_computed()."""
        evaluation = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "batch_test"),
        )
        process_pipeline = evaluation.paradigm.make_process_pipelines(dataset)[0]

        # Check via batch
        batch_result = evaluation.results.batch_not_yet_computed(
            pipelines, dataset, dataset.subject_list, process_pipeline
        )

        # Check per subject
        for subject in dataset.subject_list:
            per_subj = evaluation.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if per_subj:
                assert subject in batch_result
                assert set(per_subj.keys()) == set(batch_result[subject].keys())
            else:
                assert subject not in batch_result

    def test_after_computation(self, tmp_path):
        """batch_not_yet_computed returns empty after results are computed."""
        evaluation = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "batch_test"),
        )
        process_pipeline = evaluation.paradigm.make_process_pipelines(dataset)[0]

        # Run evaluation to populate results
        for res in evaluation.evaluate(
            dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
        ):
            evaluation.push_result(res, pipelines, process_pipeline)

        # Now batch_not_yet_computed should return empty
        batch_result = evaluation.results.batch_not_yet_computed(
            pipelines, dataset, dataset.subject_list, process_pipeline
        )
        assert batch_result == {}

    def test_batch_or_cache_returns_cached_df_when_complete(self, tmp_path):
        """Atomic helper returns cached dataframe when no work remains."""
        evaluation = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "batch_test"),
        )
        process_pipeline = evaluation.paradigm.make_process_pipelines(dataset)[0]

        for res in evaluation.evaluate(
            dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
        ):
            evaluation.push_result(res, pipelines, process_pipeline)

        work_plan, cached_df = evaluation.results.batch_not_yet_computed_or_cached_df(
            pipelines, dataset, dataset.subject_list, process_pipeline
        )
        assert work_plan == {}
        assert cached_df is not None
        assert not cached_df.empty


class TestParallelProcess:
    """Tests for the flattened parallel process() approach."""

    def test_within_session_process_structure(self, tmp_path):
        """WithinSession process() returns correct number of results."""
        evaluation = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "parallel_test"),
        )
        results = evaluation.process(pipelines)
        # 2 subjects × 2 sessions = 4 results
        assert len(results) == 4
        assert "score" in results.columns

    def test_cross_session_process_structure(self, tmp_path):
        """CrossSession process() returns correct number of results."""
        evaluation = ev.CrossSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "parallel_test"),
        )
        results = evaluation.process(pipelines)
        # 2 subjects × 2 sessions (leave-one-out) = 4 results
        assert len(results) == 4

    def test_within_subject_process_structure(self, tmp_path):
        """WithinSubject process() returns correct number of results."""
        evaluation = ev.WithinSubjectEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "parallel_test"),
        )
        results = evaluation.process(pipelines)
        # 2 subjects × 2 sessions = 4 results
        assert len(results) == 4
        assert "score" in results.columns

    def test_cross_subject_process_structure(self, tmp_path):
        """CrossSubject process() returns correct number of results."""
        evaluation = ev.CrossSubjectEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "parallel_test"),
        )
        results = evaluation.process(pipelines)
        # 2 subjects × 2 sessions = 4 results
        assert len(results) == 4

    def test_learning_curve_parallel(self, tmp_path):
        """LearningCurve evaluation via parallel process()."""
        evaluation = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            cv_class=LearningCurveSplitter,
            cv_kwargs={
                "data_size": {"policy": "ratio", "value": np.array([0.2, 0.5])},
                "n_perms": np.array([2, 2]),
            },
            overwrite=True,
            hdf5_path=str(tmp_path / "parallel_test"),
        )
        results = evaluation.process(pipelines)
        assert len(results) > 0
        assert "permutation" in results.columns
        assert "data_size" in results.columns

    def test_process_recovers_from_empty_cached_dataframe(self, tmp_path, monkeypatch):
        """Parallel path recomputes when cache says done but dataframe is empty."""
        evaluation = ev.CrossSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "parallel_test"),
        )
        original = evaluation.results.batch_not_yet_computed_or_cached_df
        state = {"calls": 0}

        def fake_batch_or_cache(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                return {}, pd.DataFrame()
            return original(*args, **kwargs)

        monkeypatch.setattr(
            evaluation.results, "batch_not_yet_computed_or_cached_df", fake_batch_or_cache
        )

        results = evaluation.process(pipelines)
        assert len(results) == 4

    def test_process_warns_on_legacy_fallback(self, tmp_path):
        """Legacy fallback path emits a deprecation warning."""
        evaluation = LegacyOnlyEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            overwrite=True,
            hdf5_path=str(tmp_path / "legacy_warning"),
        )

        with pytest.warns(FutureWarning, match="deprecated"):
            results = evaluation.process(pipelines)
        assert len(results) == len(dataset.subject_list)

    def test_within_session_dataset_order_deterministic(self, tmp_path):
        """Fixed random_state should give same scores regardless dataset order."""
        ds1 = FakeDataset(["left_hand", "right_hand"], n_subjects=2, n_runs=2, seed=12)
        ds2 = FakeDataset(["left_hand", "right_hand"], n_subjects=2, n_runs=3, seed=12)

        eval_ab = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[ds1, ds2],
            random_state=42,
            overwrite=True,
            hdf5_path=str(tmp_path / "order_ab"),
        )
        results_ab = eval_ab.process(pipelines)

        eval_ba = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[ds2, ds1],
            random_state=42,
            overwrite=True,
            hdf5_path=str(tmp_path / "order_ba"),
        )
        results_ba = eval_ba.process(pipelines)

        keys = ["dataset", "subject", "session", "pipeline"]
        left = (
            results_ab[keys + ["score"]]
            .drop_duplicates(subset=keys, keep="first")
            .rename(columns={"score": "score_ab"})
        )
        right = (
            results_ba[keys + ["score"]]
            .drop_duplicates(subset=keys, keep="first")
            .rename(columns={"score": "score_ba"})
        )

        merged = left.merge(right, on=keys, how="inner").sort_values(keys)
        assert len(merged) == len(left) == len(right)
        np.testing.assert_allclose(
            merged["score_ab"].to_numpy(),
            merged["score_ba"].to_numpy(),
            rtol=0,
            atol=0,
        )


class TestParallelLegacyEquivalence:
    """Tests verifying parallel process() produces same results as legacy."""

    def _compare_parallel_vs_legacy(self, eval_class, tmp_path, **kwargs):
        """Helper to compare parallel vs legacy results."""
        paradigm = FakeImageryParadigm()
        ds = FakeDataset(["left_hand", "right_hand"], n_subjects=2, seed=12)

        # Run parallel path
        eval_parallel = eval_class(
            paradigm=paradigm,
            datasets=[ds],
            random_state=42,
            overwrite=True,
            hdf5_path=str(tmp_path / "parallel"),
            **kwargs,
        )
        results_parallel = eval_parallel.process(pipelines)

        # Run legacy path
        eval_legacy = eval_class(
            paradigm=paradigm,
            datasets=[ds],
            random_state=42,
            overwrite=True,
            hdf5_path=str(tmp_path / "legacy"),
            **kwargs,
        )
        results_legacy = eval_legacy._process_legacy(
            pipelines, param_grid=None, postprocess_pipeline=None
        )

        # Compare
        keys = ["subject", "session", "pipeline"]
        left = (
            results_parallel[keys + ["score"]]
            .sort_values(keys)
            .reset_index(drop=True)
            .rename(columns={"score": "score_parallel"})
        )
        right = (
            results_legacy[keys + ["score"]]
            .sort_values(keys)
            .reset_index(drop=True)
            .rename(columns={"score": "score_legacy"})
        )

        assert len(left) == len(
            right
        ), f"Different number of results: parallel={len(left)}, legacy={len(right)}"
        merged = left.merge(right, on=keys, how="inner")
        assert len(merged) == len(
            left
        ), "Not all rows matched between parallel and legacy"
        np.testing.assert_allclose(
            merged["score_parallel"].to_numpy(),
            merged["score_legacy"].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_within_session_equivalence(self, tmp_path):
        """WithinSession parallel matches legacy scores."""
        self._compare_parallel_vs_legacy(ev.WithinSessionEvaluation, tmp_path)

    def test_cross_session_equivalence(self, tmp_path):
        """CrossSession parallel matches legacy scores."""
        self._compare_parallel_vs_legacy(ev.CrossSessionEvaluation, tmp_path)

    def test_cross_subject_equivalence(self, tmp_path):
        """CrossSubject parallel matches legacy scores."""
        self._compare_parallel_vs_legacy(ev.CrossSubjectEvaluation, tmp_path)

    def test_within_subject_equivalence(self, tmp_path):
        """WithinSubject parallel matches legacy scores."""
        self._compare_parallel_vs_legacy(ev.WithinSubjectEvaluation, tmp_path)


class TestAggregateFoldResults:
    """Tests for _aggregate_fold_results score handling."""

    @staticmethod
    def _make_fold(subject, session, pipeline, score, extras=None, is_error=False):
        res = {
            "subject": subject,
            "session": session,
            "pipeline": pipeline,
            "time": 1.0,
            "n_samples": 100,
            "n_samples_total": 200,
            "n_channels": 8,
            "dataset": "FakeDataset",
            "score": score,
            "is_error": is_error,
        }
        if extras:
            res.update(extras)
        return res

    def test_multi_metric_no_double_prefix(self):
        """Averaged multi-metric scores keep correct score_* key names."""
        from moabb.evaluations.base import BaseEvaluation

        folds = [
            self._make_fold(1, "0", "csp", 0.8, {"score_accuracy": 0.8, "score_f1": 0.7}),
            self._make_fold(
                1, "0", "csp", 0.9, {"score_accuracy": 0.9, "score_f1": 0.85}
            ),
        ]
        agg = BaseEvaluation._aggregate_fold_results(folds)
        assert len(agg) == 1
        res = agg[0]
        # Correct keys present
        assert "score" in res
        assert "score_accuracy" in res
        assert "score_f1" in res
        # No double-prefixed keys
        assert not any(k.startswith("score_score_") for k in res)
        np.testing.assert_almost_equal(res["score_accuracy"], 0.85)
        np.testing.assert_almost_equal(res["score_f1"], 0.775)

    def test_error_folds_included_in_average(self):
        """Error folds contribute their error_score to the average."""
        from moabb.evaluations.base import BaseEvaluation

        folds = [
            self._make_fold(1, "0", "csp", 0.9),
            self._make_fold(1, "0", "csp", 0.0, is_error=True),
        ]
        agg = BaseEvaluation._aggregate_fold_results(folds)
        assert len(agg) == 1
        # Average of 0.9 and 0.0 = 0.45, not 0.9 (old behavior dropped errors)
        np.testing.assert_almost_equal(agg[0]["score"], 0.45)

    def test_all_folds_errored_still_produces_result(self):
        """When every fold errors, the group still appears with error_score."""
        from moabb.evaluations.base import BaseEvaluation

        folds = [
            self._make_fold(1, "0", "csp", 0.0, is_error=True),
            self._make_fold(1, "0", "csp", 0.0, is_error=True),
        ]
        agg = BaseEvaluation._aggregate_fold_results(folds)
        # Should produce a result instead of silently dropping the group
        assert len(agg) == 1
        np.testing.assert_almost_equal(agg[0]["score"], 0.0)

    def test_error_folds_multi_metric_use_fallback(self):
        """Error folds use their score value as fallback for missing metrics."""
        from moabb.evaluations.base import BaseEvaluation

        folds = [
            self._make_fold(1, "0", "csp", 0.8, {"score_accuracy": 0.8, "score_f1": 0.7}),
            # Error fold only has "score", missing score_accuracy/score_f1
            self._make_fold(1, "0", "csp", 0.0, is_error=True),
        ]
        agg = BaseEvaluation._aggregate_fold_results(folds)
        assert len(agg) == 1
        res = agg[0]
        # score_accuracy: avg(0.8, 0.0) = 0.4 (error fold uses score=0.0)
        np.testing.assert_almost_equal(res["score_accuracy"], 0.4)
        # score_f1: avg(0.7, 0.0) = 0.35
        np.testing.assert_almost_equal(res["score_f1"], 0.35)
