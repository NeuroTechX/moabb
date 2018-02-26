import unittest
import numpy as np
import moabb.viz.meta_analysis as ma
from moabb.viz import Results
import os
from moabb.datasets.base import BaseDataset
from moabb.contexts.base import BaseEvaluation

# dummy evaluation
class DummyEvaluation(BaseEvaluation):

    def evaluate(self, dataset, subject, clf, paradigm):
        raise NotImplementedError('dummy')


# dummy datasets
class DummyDataset(BaseDataset):

    def __init__(self, code):
        """

        """
        super().__init__(list(range(5)), 2, {'a':1,'b':2},code, [1,2], 'imagery')
        


# Create dummy data for tests
d1 = {'time': 1,
            'dataset': DummyDataset('d1'),
            'id': 1,
            'score': 0.9,
            'n_samples': 100,
            'n_channels':10}

d2 = {'time': 2,
            'dataset': DummyDataset('d1'),
            'id': 2,
            'score': 0.9,
            'n_samples': 100,
            'n_channels':10}


d3 = {'time': 2,
            'dataset': DummyDataset('d2'),
            'id': 2,
            'score': 0.9,
            'n_samples': 100,
            'n_channels':10}

def to_result_input(pnames, dsets):
    return dict(zip(pnames, dsets))

class Test_Stats(unittest.TestCase):

    def test_rmanova(self):
        matrix=np.asarray([[45,50,55],
                           [42,42,45],
                           [36,41,43],
                           [39,35,40],
                           [51,55,59],
                           [44,49,56]])
        f, p = ma._rmanova(matrix)
        self.assertAlmostEqual(f, 12.53, places=2)
        self.assertAlmostEqual(p, 0.002, places=3)

class Test_Results(unittest.TestCase):

    def setUp(self):
        self.obj = Results(evaluation=DummyEvaluation(), path='results.hdf5')

    def tearDown(self):
        if os.path.isfile('results.hd5'):
            os.remove('results.hd5')

    def testCanAddSample(self):
        self.obj.add(to_result_input(['a'],[d1]))
    
    def testRecognizesAlreadyComputed(self):
        _in = to_result_input(['a'],[d1])
        self.obj.add(_in)
        not_yet_computed = self.obj.not_yet_computed({'a':1}, d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed)==0)

    def testCanAddMultiplePipelines(self):
        _in = to_result_input(['a','b','c'],[d1,d1,d2])
        self.obj.add(_in)

    def testCanAddMultipleValuesPerPipeline(self):
        _in = to_result_input(['a','b'],[[d1,d2],[d2,d1]])
        self.obj.add(_in)
        not_yet_computed = self.obj.not_yet_computed({'a':1}, d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed)==0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed({'b':2}, d2['dataset'], d2['id'])
        self.assertTrue(len(not_yet_computed)==0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed({'b':1}, d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed)==0, not_yet_computed)

    def testCanExportToDataframe(self):
        pass

if __name__ == "__main__":
    unittest.main()
