from moabb.evaluations import evaluations as ev
from moabb.datasets.bnci import BNCI2014001
from moabb.analysis import Results
from moabb.paradigms.motor_imagery import LeftRightImagery
import unittest

from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline

from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pipelines = OrderedDict()
pipelines['C'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
dataset = BNCI2014001()


class Test_CrossSess(unittest.TestCase):
    '''This is actually integration testing but I don't know how to do this
    better. A paradigm implements pre-processing so it needs files to run MNE
    stuff on. To test the scoring and train/test we need to also have data and
    run it. Putting this on the future docket...

    '''

    def return_eval(self):
        return ev.CrossSessionEvaluation(paradigm=LeftRightImagery(),
                                         datasets=[dataset])

    def test_eval_results(self):
        e = self.return_eval()
        e.preprocess_data(dataset)
        e.evaluate(dataset, 1, pipelines)


class Test_CrossSubj(Test_CrossSess):
    def return_eval(self):
        return ev.CrossSubjectEvaluation(paradigm=LeftRightImagery(),
                                         datasets=[dataset])


class Test_WithinSess(Test_CrossSess):
    def return_eval(self):
        return ev.WithinSessionEvaluation(paradigm=LeftRightImagery(),
                                          datasets=[dataset])


if __name__ == "__main__":
    unittest.main()
