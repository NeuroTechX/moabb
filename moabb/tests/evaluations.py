from moabb.evaluations import evaluations as ev
from moabb.datasets.fake import FakeDataset
from moabb.paradigms.motor_imagery import FakeImageryParadigm
import unittest

from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline

from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pipelines = OrderedDict()
pipelines['C'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
dataset = FakeDataset(['left_hand', 'right_hand'])


class Test_CrossSess(unittest.TestCase):
    '''This is actually integration testing but I don't know how to do this
    better. A paradigm implements pre-processing so it needs files to run MNE
    stuff on. To test the scoring and train/test we need to also have data and
    run it. Putting this on the future docket...

    '''

    def return_eval(self):
        return ev.CrossSessionEvaluation(paradigm=FakeImageryParadigm(),
                                         datasets=[dataset])

    def test_eval_results(self):
        e = self.return_eval()
        e.preprocess_data(dataset)
        res = e.evaluate(dataset, 1, pipelines)

        # return 1 results averaged across the 2 sessions
        self.assertEqual(len(res['C']), 1)


class Test_CrossSubj(Test_CrossSess):
    def return_eval(self):
        return ev.CrossSubjectEvaluation(paradigm=FakeImageryParadigm(),
                                         datasets=[dataset])

    def test_eval_results(self):
        e = self.return_eval()
        e.preprocess_data(dataset)
        res = e.evaluate(dataset, 1, pipelines)

        # return 1 results for 1 subject
        self.assertEqual(len(res['C']), 1)


class Test_WithinSess(Test_CrossSess):
    def return_eval(self):
        return ev.WithinSessionEvaluation(paradigm=FakeImageryParadigm(),
                                          datasets=[dataset])

    def test_eval_results(self):
        e = self.return_eval()
        e.preprocess_data(dataset)
        res = e.evaluate(dataset, 1, pipelines)

        # return 2 results for each session
        self.assertEqual(len(res['C']), 2)

if __name__ == "__main__":
    unittest.main()
