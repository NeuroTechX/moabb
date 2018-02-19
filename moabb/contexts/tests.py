from . import evaluations as ev
from ..datasets.bnci import BNCI2014001
from ..viz import Results
from .motor_imagery import LeftRightImagery
import unittest

from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline

from collections import OrderedDict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pipelines = OrderedDict()
pipelines['C'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
d = BNCI2014001()
d.selected_events = {k: d.event_id[k] for k in ['left_hand', 'right_hand']}


class Test_CrossSess(unittest.TestCase):
    def return_eval(self):
        return ev.CrossSessionEvaluation()

    def test_eval_results(self):
        e = self.return_eval()
        r = Results(e, pipelines)
        p = LeftRightImagery(pipelines, e, [d])
        r.add(e.evaluate(d, 1,
                         pipelines['C'], p), 'C')


class Test_CrossSubj(unittest.TestCase):
    def return_eval(self):
        return ev.CrossSubjectEvaluation()

    def test_eval_results(self):
        e = self.return_eval()
        r = Results(e, pipelines)
        p = LeftRightImagery(pipelines, e, [d])
        e.preprocess_data(d, p)
        r.add(e.evaluate(d, 1,
                         pipelines['C'], p), 'C')


class Test_WithinSess(unittest.TestCase):
    def return_eval(self):
        return ev.WithinSessionEvaluation()

    def test_eval_results(self):
        e = self.return_eval()
        r = Results(e, pipelines)
        p = LeftRightImagery(pipelines, e, [d])
        r.add(e.evaluate(d, 1, pipelines['C'], p), 'C')


if __name__ == "__main__":
    unittest.main()
