from moabb.evaluations import evaluations as ev
from moabb.datasets.bnci import BNCI2014001
from moabb.analysis import Results
from moabb.paradigms.motor_imagery import LeftRightImagery
import unittest

from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline
import os

from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pipelines = OrderedDict()
pipelines['C'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
dataset = BNCI2014001()
dataset.selected_events = {k: dataset.event_id[k]
                           for k in ['left_hand', 'right_hand']}


class Test_CrossSess(unittest.TestCase):
    '''This is actually integration testing but I don't know how to do this
    better. A paradigm implements pre-processing so it needs files to run MNE
    stuff on. To test the scoring and train/test we need to also have data and
    run it. Putting this on the future docket...

    '''
    def tearDown(self):
        if os.path.isfile('results.hd5'):
            os.remove('results.hd5')

    def return_eval(self):
        return ev.CrossSessionEvaluation(paradigm=LeftRightImagery())

    def test_eval_results(self):
        e = self.return_eval()
        e.preprocess_data(dataset)
        e.evaluate(dataset, 1, pipelines)


class Test_CrossSubj(Test_CrossSess):
    def return_eval(self):
        return ev.CrossSubjectEvaluation(paradigm=LeftRightImagery())


class Test_WithinSess(Test_CrossSess):
    def return_eval(self):
        return ev.WithinSessionEvaluation(paradigm=LeftRightImagery())


if __name__ == "__main__":
    unittest.main()
