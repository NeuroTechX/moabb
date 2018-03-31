from moabb.evaluations import evaluations as ev
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.motor_imagery import FakeImageryParadigm
import unittest
import os

from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline

from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pipelines = OrderedDict()
pipelines['C'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
dataset = FakeDataset(['left_hand', 'right_hand'], n_subjects=2)


class Test_WithinSess(unittest.TestCase):
    '''This is actually integration testing but I don't know how to do this
    better. A paradigm implements pre-processing so it needs files to run MNE
    stuff on. To test the scoring and train/test we need to also have data and
    run it. Putting this on the future docket...

    '''

    def setUp(self):
        self.eval = ev.WithinSessionEvaluation(paradigm=FakeImageryParadigm(),
                                               datasets=[dataset])

    def tearDown(self):
        path = self.eval.results.filepath
        print(path)
        if os.path.isfile(path):
            os.remove(path)

    def test_eval_results(self):
        results = [r for r in self.eval.evaluate(dataset, pipelines)]

        # We should get 4 results, 2 session 2 subject
        self.assertEqual(len(results), 4)


class Test_CrossSubj(Test_WithinSess):

    def setUp(self):
        self.eval = ev.CrossSubjectEvaluation(paradigm=FakeImageryParadigm(),
                                              datasets=[dataset])


class Test_CrossSess(Test_WithinSess):
    def setUp(self):
        self.eval = ev.CrossSessionEvaluation(paradigm=FakeImageryParadigm(),
                                              datasets=[dataset])


if __name__ == "__main__":
    unittest.main()
