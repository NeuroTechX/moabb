import unittest
import numpy as np
import moabb.analysis.meta_analysis as ma
from moabb.analysis import Results
import os
from moabb.evaluations.base import BaseEvaluation
from moabb.paradigms.base import BaseParadigm
from moabb.datasets.fake import FakeDataset
# dummy evaluation


class DummyEvaluation(BaseEvaluation):

    def evaluate(self, dataset, pipelines):
        raise NotImplementedError('dummy')

    def verify(self, dataset):
        pass


class DummyParadigm(BaseParadigm):

    def __init__(self):
        pass

    @property
    def scoring(self):
        raise NotImplementedError('dummy')

    def verify(self, dataset):
        pass

    def process_raw(raw):
        raise NotImplementedError('dummy')

    @property
    def datasets(self):
        return [FakeDataset(['d1', 'd2'])]


# Create dummy data for tests
d1 = {'time': 1,
      'dataset': FakeDataset(['d1', 'd2']),
      'id': 1,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}

d2 = {'time': 2,
      'dataset': FakeDataset(['d1', 'd2']),
      'id': 2,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}


d3 = {'time': 2,
      'dataset': FakeDataset(['d1', 'd2']),
      'id': 2,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}

d4 = {'time': 2,
      'dataset': FakeDataset(['d1', 'd2']),
      'id': 1,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}


def to_pipeline_dict(pnames):
    return {n: 'pipeline {}'.format(n) for n in pnames}


def to_result_input(pnames, dsets):
    return dict(zip(pnames, dsets))


class Test_Stats(unittest.TestCase):

    def test_rmanova(self):
        matrix = np.asarray([[45, 50, 55],
                             [42, 42, 45],
                             [36, 41, 43],
                             [39, 35, 40],
                             [51, 55, 59],
                             [44, 49, 56]])
        f, p = ma._rmanova(matrix)
        self.assertAlmostEqual(f, 12.53, places=2)
        self.assertAlmostEqual(p, 0.002, places=3)


class Test_Integration(unittest.TestCase):

    def setUp(self):
        self.obj = Results(evaluation_class=DummyEvaluation,
                           paradigm_class=DummyParadigm,
                           suffix='test')

    def tearDown(self):
        path = self.obj.filepath
        if os.path.isfile(path):
            os.remove(path)

    def test_rmanova(self):
        _in = to_result_input(['a', 'b', 'c'], [[d1]*5, [d1]*5, [d4]*5])
        self.obj.add(_in, to_pipeline_dict(['a', 'b', 'c']))
        _in = to_result_input(['a', 'b', 'c'], [[d2]*5, [d2]*5, [d3]*5])
        self.obj.add(_in, to_pipeline_dict(['a', 'b', 'c']))
        df = self.obj.to_dataframe()
        ma.rmANOVA(df)


class Test_Results(unittest.TestCase):

    def setUp(self):
        self.obj = Results(evaluation_class=DummyEvaluation,
                           paradigm_class=DummyParadigm,
                           suffix='test')

    def tearDown(self):
        path = self.obj.filepath
        if os.path.isfile(path):
            os.remove(path)

    def testCanAddSample(self):
        self.obj.add(to_result_input(['a'], [d1]), to_pipeline_dict(['a']))

    def testRecognizesAlreadyComputed(self):
        _in = to_result_input(['a'], [d1])
        self.obj.add(_in, to_pipeline_dict(['a']))
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(['a']), d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed) == 0)

    def testCanAddMultiplePipelines(self):
        _in = to_result_input(['a', 'b', 'c'], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(['a', 'b', 'c']))

    def testCanAddMultipleValuesPerPipeline(self):
        _in = to_result_input(['a', 'b'], [[d1, d2], [d2, d1]])
        self.obj.add(_in, to_pipeline_dict(['a', 'b']))
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(['a']), d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(['b']), d2['dataset'], d2['id'])
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(['b']), d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)

    def testCanExportToDataframe(self):
        _in = to_result_input(['a', 'b', 'c'], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(['a', 'b', 'c']))
        _in = to_result_input(['a', 'b', 'c'], [d2, d2, d3])
        self.obj.add(_in, to_pipeline_dict(['a', 'b', 'c']))
        df = self.obj.to_dataframe()
        self.assertTrue(set(np.unique(df['pipeline'])) == set(
            ('a', 'b', 'c')), np.unique(df['pipeline']))
        self.assertTrue(df.shape[0] == 6, df.shape[0])


if __name__ == "__main__":
    unittest.main()
