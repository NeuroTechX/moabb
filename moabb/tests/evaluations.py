import os
import os.path as osp
import platform
import unittest
import warnings
from collections import OrderedDict

import numpy as np
import pytest
import sklearn.base
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.dummy import DummyClassifier as Dummy
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline

from moabb.analysis.results import get_string_rep
from moabb.datasets.compound_dataset import compound
from moabb.datasets.fake import FakeDataset
from moabb.evaluations import evaluations as ev
from moabb.evaluations.utils import create_save_path, save_model_cv, save_model_list
from moabb.paradigms.motor_imagery import FakeImageryParadigm


try:
    from codecarbon import EmissionsTracker  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

pipelines = OrderedDict()
pipelines["C"] = make_pipeline(Covariances("oas"), CSP(8), LDA())
dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=2, seed=12)
if not osp.isdir(osp.join(osp.expanduser("~"), "mne_data")):
    os.makedirs(osp.join(osp.expanduser("~"), "mne_data"))


class DummyClassifier(sklearn.base.BaseEstimator):
    __slots__ = "kernel"

    def __init__(self, kernel):
        self.kernel = kernel


class Test_WithinSess(unittest.TestCase):
    """This is actually integration testing but I don't know how to do this
    better.

    A paradigm implements pre-processing so it needs files to run MNE
    stuff on. To test the scoring and train/test we need to also have data and
    run it. Putting this on the future docket...
    """

    def setUp(self):
        self.eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            hdf5_path="res_test",
            save_model=True,
            optuna=False,
        )

    def test_mne_labels(self):
        kwargs = dict(paradigm=FakeImageryParadigm(), datasets=[dataset])
        epochs = dict(return_epochs=False, mne_labels=True)
        self.assertRaises(ValueError, ev.WithinSessionEvaluation, **epochs, **kwargs)

    def tearDown(self):
        path = self.eval.results.filepath
        if os.path.isfile(path):
            os.remove(path)

    def test_eval_results(self):
        process_pipeline = self.eval.paradigm.make_process_pipelines(dataset)[0]
        results = [
            r
            for r in self.eval.evaluate(
                dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
            )
        ]

        # We should get 4 results, 2 sessions 2 subjects
        self.assertEqual(len(results), 4)
        # We should have 9 columns in the results data frame
        self.assertEqual(len(results[0].keys()), 9 if _carbonfootprint else 8)

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
        self.assertEqual(len(results), 4)
        # We should have 9 columns in the results data frame
        self.assertEqual(len(results[0].keys()), 9 if _carbonfootprint else 8)

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
        self.assertEqual(len(results), 4)
        # We should have 9 columns in the results data frame
        self.assertEqual(len(results[0].keys()), 9 if _carbonfootprint else 8)

    def test_eval_grid_search_optuna(self):
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
        self.assertEqual(len(results), 4)
        # We should have 9 columns in the results data frame
        self.assertEqual(len(results[0].keys()), 9 if _carbonfootprint else 8)

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
        self.assertTrue(
            model_folder_exists,
            "No folder with partial name 'Model' found inside 'res_test' directory",
        )

    def test_lambda_warning(self):
        def explicit_kernel(x):
            return x**3

        c1 = DummyClassifier(kernel=lambda x: x**2)
        c2 = DummyClassifier(kernel=lambda x: 5 * x)

        c3 = DummyClassifier(kernel=explicit_kernel)

        self.assertFalse(repr(c1) == repr(c2))
        if platform.system() != "Windows":
            with self.assertWarns(RuntimeWarning):
                self.assertTrue(get_string_rep(c1) == get_string_rep(c2))

        # I do not know an elegant way to check for no warnings
        with warnings.catch_warnings(record=True) as w:
            get_string_rep(c3)
            self.assertTrue(len(w) == 0)

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
            pipelines0, postprocess_pipeline=FunctionTransformer(lambda x: x)
        )
        results2 = self.eval.process(pipelines1, postprocess_pipeline=cov)
        np.testing.assert_allclose(results0.score, results1.score)
        np.testing.assert_allclose(results0.score, results2.score)


class Test_WithinSessLearningCurve(unittest.TestCase):
    """Some tests for the learning curve evaluation.

    TODO if we ever extend dataset metadata, e.g. including y for
    example, we could get rid of a lot of issues regarding valid inputs
    for policy per_class as this could be determined at Evaluation
    initialization instead of during running the evaluation
    """

    @pytest.mark.skip(reason="This test is not working")
    def test_correct_results_integrity(self):
        learning_curve_eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            data_size={"policy": "ratio", "value": np.array([0.2, 0.5])},
            n_perms=np.array([2, 2]),
        )
        process_pipeline = learning_curve_eval.paradigm.make_process_pipelines(dataset)[0]
        results = [
            r
            for r in learning_curve_eval.evaluate(
                dataset, pipelines, param_grid=None, process_pipeline=process_pipeline
            )
        ]
        keys = results[0].keys()
        self.assertEqual(len(keys), 10)  # 8 + 2 new for learning curve
        self.assertTrue("permutation" in keys)
        self.assertTrue("data_size" in keys)

    def test_all_policies_work(self):
        kwargs = dict(paradigm=FakeImageryParadigm(), datasets=[dataset], n_perms=[2, 2])
        # The next two should work without issue
        ev.WithinSessionEvaluation(
            data_size={"policy": "per_class", "value": [5, 10]}, **kwargs
        )
        ev.WithinSessionEvaluation(
            data_size={"policy": "ratio", "value": [0.2, 0.5]}, **kwargs
        )
        self.assertRaises(
            ValueError,
            ev.WithinSessionEvaluation,
            **dict(data_size={"policy": "does_not_exist", "value": [0.2, 0.5]}, **kwargs),
        )

    @pytest.mark.skip(reason="This test is not working")
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
        kwargs = dict(paradigm=FakeImageryParadigm(), datasets=[dataset], n_perms=[2, 2])
        should_work = ev.WithinSessionEvaluation(
            data_size={"policy": "per_class", "value": [5, 10]}, **kwargs
        )
        too_many_samples = ev.WithinSessionEvaluation(
            data_size={"policy": "per_class", "value": [5, 100000]}, **kwargs
        )
        # This one should run
        run_evaluation(should_work, dataset, pipelines)
        self.assertRaises(
            ValueError,
            run_evaluation,
            too_many_samples,
            dataset,
            pipelines,
        )

    def test_eval_grid_search(self):
        pass

    def test_datasize_parameters(self):
        # Fail if not values are not correctly ordered
        kwargs = dict(paradigm=FakeImageryParadigm(), datasets=[dataset])
        decreasing_datasize = dict(
            data_size={"policy": "per_class", "value": [5, 4]}, n_perms=[2, 1], **kwargs
        )
        constant_datasize = dict(
            data_size={"policy": "per_class", "value": [5, 5]}, n_perms=[2, 3], **kwargs
        )
        increasing_perms = dict(
            data_size={"policy": "per_class", "value": [3, 4]}, n_perms=[2, 3], **kwargs
        )
        self.assertRaises(ValueError, ev.WithinSessionEvaluation, **decreasing_datasize)
        self.assertRaises(ValueError, ev.WithinSessionEvaluation, **constant_datasize)
        self.assertRaises(ValueError, ev.WithinSessionEvaluation, **increasing_perms)
        pass

    def test_postprocess_pipeline(self):
        learning_curve_eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            data_size={"policy": "ratio", "value": np.array([0.2, 0.5])},
            n_perms=np.array([2, 2]),
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
            pipelines0, postprocess_pipeline=FunctionTransformer(lambda x: x)
        )
        results2 = learning_curve_eval.process(pipelines1, postprocess_pipeline=cov)
        np.testing.assert_allclose(results0.score, results1.score)
        np.testing.assert_allclose(results0.score, results2.score)


class Test_AdditionalColumns(unittest.TestCase):
    def setUp(self):
        self.eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            additional_columns=["one", "two"],
        )

    def tearDown(self):
        path = self.eval.results.filepath
        if os.path.isfile(path):
            os.remove(path)


class Test_CrossSubj(Test_WithinSess):
    def setUp(self):
        self.eval = ev.CrossSubjectEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            hdf5_path="res_test",
            save_model=True,
        )

    def test_compatible_dataset(self):
        # raise
        ds = FakeDataset(["left_hand", "right_hand"], n_subjects=1)
        self.assertFalse(self.eval.is_valid(dataset=ds))

        # do not raise
        ds = FakeDataset(["left_hand", "right_hand"], n_subjects=2)
        self.assertTrue(self.eval.is_valid(dataset=ds))


class Test_CrossSess(Test_WithinSess):
    def setUp(self):
        self.eval = ev.CrossSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            hdf5_path="res_test",
            save_model=True,
        )

    def test_compatible_dataset(self):
        ds = FakeDataset(["left_hand", "right_hand"], n_sessions=1)
        self.assertFalse(self.eval.is_valid(ds))

        # do not raise
        ds = FakeDataset(["left_hand", "right_hand"], n_sessions=2)
        self.assertTrue(self.eval.is_valid(dataset=ds))


class UtilEvaluation(unittest.TestCase):
    def test_save_model_cv(self):
        model = Dummy()
        save_path = "test_save_path"
        cv_index = 0

        save_model_cv(model, save_path, cv_index)

        # Assert that the saved model file exists
        self.assertTrue(os.path.isfile(os.path.join(save_path, "fitted_model_0.pkl")))

    def test_save_model_list(self):
        step = Dummy()
        model = Pipeline([("step", step)])
        model_list = [model]
        score_list = [0.8]
        save_path = "test_save_path"
        save_model_list(model_list, score_list, save_path)

        # Assert that the saved model file for best model exists
        self.assertTrue(os.path.isfile(os.path.join(save_path, "fitted_model_best.pkl")))

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
        self.assertEqual(save_path, expected_path)

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
        self.assertEqual(grid_save_path, expected_grid_path)

    def test_save_model_cv_with_pytorch_model(self):
        try:
            import torch
            from skorch import NeuralNetClassifier
        except ImportError:
            self.skipTest("skorch library not available")

        step = NeuralNetClassifier(module=torch.nn.Linear(10, 2))
        step.initialize()
        model = Pipeline([("step", step)])
        save_path = "."
        cv_index = 0
        save_model_cv(model, save_path, cv_index)

        # Assert that the saved model files exist
        self.assertTrue(
            os.path.isfile(os.path.join(save_path, "step_fitted_0_model.pkl"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(save_path, "step_fitted_0_optim.pkl"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(save_path, "step_fitted_0_history.json"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(save_path, "step_fitted_0_criterion.pkl"))
        )

    def test_save_model_list_with_multiple_models(self):
        model1 = Dummy()
        model2 = Dummy()
        model_list = [model1, model2]
        score_list = [0.8, 0.9]
        save_path = "test_save_path"
        save_model_list(model_list, score_list, save_path)

        # Assert that the saved model files for each model exist
        self.assertTrue(os.path.isfile(os.path.join(save_path, "fitted_model_0.pkl")))
        self.assertTrue(os.path.isfile(os.path.join(save_path, "fitted_model_1.pkl")))

        # Assert that the saved model file for the best model exists
        self.assertTrue(os.path.isfile(os.path.join(save_path, "fitted_model_best.pkl")))

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
        self.assertEqual(save_path, expected_path)

        grid_save_path = create_save_path(
            hdf5_path, code, subject, session, name, grid=True, eval_type=eval_type
        )

        expected_grid_path = os.path.join(
            hdf5_path, "GridSearch_CrossSession", code, "1", "evaluation_name"
        )
        self.assertEqual(grid_save_path, expected_grid_path)

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

        self.assertIsNone(save_path)

    def test_save_model_cv_without_hdf5_path(self):
        model = DummyClassifier(kernel="rbf")
        save_path = None
        cv_index = 0

        # Assert that calling save_model_cv without a save_path does raise an IOError
        with self.assertRaises(IOError):
            save_model_cv(model, save_path, cv_index)

    def test_save_model_list_with_single_model(self):
        model = Dummy()
        model_list = model
        score_list = [0.8]
        save_path = "test_save_path"
        save_model_list(model_list, score_list, save_path)

        # Assert that the saved model file for the single model exists
        self.assertTrue(os.path.isfile(os.path.join(save_path, "fitted_model_0.pkl")))

        # Assert that the saved model file for the best model exists
        self.assertTrue(os.path.isfile(os.path.join(save_path, "fitted_model_best.pkl")))

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
        self.assertEqual(save_path, expected_path)

        grid_save_path = create_save_path(
            hdf5_path, code, subject, session, name, grid=True, eval_type=eval_type
        )

        expected_grid_path = os.path.join(
            hdf5_path, "GridSearch_CrossSubject", code, "1", "evaluation_name"
        )
        self.assertEqual(grid_save_path, expected_grid_path)

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

        self.assertIsNone(save_path)

        grid_save_path = create_save_path(
            hdf5_path, code, subject, session, name, grid=True, eval_type=eval_type
        )

        self.assertIsNone(grid_save_path)

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
        self.assertEqual(save_path, expected_path)


if __name__ == "__main__":
    unittest.main()
