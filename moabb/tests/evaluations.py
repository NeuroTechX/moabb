import os
import os.path as osp
import unittest
from collections import OrderedDict

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb.datasets.fake import FakeDataset
from moabb.evaluations import evaluations as ev
from moabb.paradigms.motor_imagery import FakeImageryParadigm


pipelines = OrderedDict()
pipelines["C"] = make_pipeline(Covariances("oas"), CSP(8), LDA())
dataset = FakeDataset(["left_hand", "right_hand"], n_subjects=2)
if not osp.isdir(osp.join(osp.expanduser("~"), "mne_data")):
    os.makedirs(osp.join(osp.expanduser("~"), "mne_data"))


class Test_WithinSess(unittest.TestCase):
    """This is actually integration testing but I don't know how to do this
    better. A paradigm implements pre-processing so it needs files to run MNE
    stuff on. To test the scoring and train/test we need to also have data and
    run it. Putting this on the future docket...

    """

    def setUp(self):
        self.eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(), datasets=[dataset]
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
        results = [r for r in self.eval.evaluate(dataset, pipelines)]

        # We should get 4 results, 2 session 2 subject
        self.assertEqual(len(results), 4)
        # We should have 8 columns in the results data frame
        self.assertEqual(len(results[0].keys()), 8)


class Test_WithinSessLearningCurve(unittest.TestCase):
    """
    Some tests for the learning curve evaluation.

    TODO if we ever extend dataset metadata, e.g. including y for example, we could get rid of a
    lot of issues regarding valid inputs for policy per_class as this could be determined at
    Evaluation initialization instead of during running the evaluation
    """

    def test_correct_results_integrity(self):
        learning_curve_eval = ev.WithinSessionEvaluation(
            paradigm=FakeImageryParadigm(),
            datasets=[dataset],
            data_size={"policy": "ratio", "value": np.array([0.2, 0.5])},
            n_perms=np.array([2, 2]),
        )
        results = [r for r in learning_curve_eval.evaluate(dataset, pipelines)]
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

    def test_data_sanity(self):
        # need this helper to iterate over the generator
        def run_evaluation(eval, dataset, pipelines):
            list(eval.evaluate(dataset, pipelines))

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
            ValueError, run_evaluation, too_many_samples, dataset, pipelines
        )
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

    def test_fails_if_nothing_returned(self):
        self.assertRaises(ValueError, self.eval.process, pipelines)
        # TODO Add custom evaluation that actually returns additional info


class Test_CrossSubj(Test_WithinSess):
    def setUp(self):
        self.eval = ev.CrossSubjectEvaluation(
            paradigm=FakeImageryParadigm(), datasets=[dataset]
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
            paradigm=FakeImageryParadigm(), datasets=[dataset]
        )

    def test_compatible_dataset(self):
        ds = FakeDataset(["left_hand", "right_hand"], n_sessions=1)
        self.assertFalse(self.eval.is_valid(ds))

        # do not raise
        ds = FakeDataset(["left_hand", "right_hand"], n_sessions=2)
        self.assertTrue(self.eval.is_valid(dataset=ds))


if __name__ == "__main__":
    unittest.main()
