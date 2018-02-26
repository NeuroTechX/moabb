from moabb.contexts import evaluations as ev
from moabb.datasets.bnci import BNCI2014001
from moabb.viz import Results
from moabb.contexts.motor_imagery import LeftRightImagery
import unittest

from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline
import os

from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pipelines = OrderedDict()
pipelines['C'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
d = BNCI2014001()
d.selected_events = {k: d.event_id[k] for k in ['left_hand', 'right_hand']}

class Test_CrossSess(unittest.TestCase):

    def tearDown(self):
        if os.path.isfile('results.hd5'):
            os.remove('results.hd5')

    def return_eval(self):
        return ev.CrossSessionEvaluation()

    def test_eval_results(self):
        e = self.return_eval()
        r = Results(e,'results.hd5')
        p = LeftRightImagery(pipelines, e, [d])
        e.preprocess_data(d,p)
        r.add(e.evaluate(d, 1,
                         pipelines, p))

class Test_CrossSubj(Test_CrossSess):
    def return_eval(self):
        return ev.CrossSubjectEvaluation()

class Test_WithinSess(Test_CrossSess):
    def return_eval(self):
        return ev.WithinSessionEvaluation()


if __name__ == "__main__":
    unittest.main()
