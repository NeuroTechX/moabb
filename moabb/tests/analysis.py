import os
import unittest

import numpy as np
import pandas as pd

import moabb.analysis.meta_analysis as ma
from moabb.analysis import Results
from moabb.datasets.fake import FakeDataset
from moabb.evaluations.base import BaseEvaluation
from moabb.paradigms.base import BaseParadigm


# dummy evaluation


class DummyEvaluation(BaseEvaluation):
    def evaluate(self, dataset, pipelines):
        raise NotImplementedError("dummy")

    def is_valid(self, dataset):
        pass


class DummyParadigm(BaseParadigm):
    def __init__(self):
        pass

    @property
    def scoring(self):
        raise NotImplementedError("dummy")

    def is_valid(self, dataset):
        pass

    def process_raw(self, raw, dataset, return_epochs=False):
        raise NotImplementedError("dummy")

    @property
    def datasets(self):
        return [FakeDataset(["d1", "d2"])]


# Create dummy data for tests
d1 = {
    "time": 1,
    "dataset": FakeDataset(["d1", "d2"]),
    "subject": 1,
    "session": "session_0",
    "score": 0.9,
    "n_samples": 100,
    "n_channels": 10,
}

d2 = {
    "time": 2,
    "dataset": FakeDataset(["d1", "d2"]),
    "subject": 2,
    "session": "session_0",
    "score": 0.9,
    "n_samples": 100,
    "n_channels": 10,
}


d3 = {
    "time": 2,
    "dataset": FakeDataset(["d1", "d2"]),
    "subject": 2,
    "session": "session_0",
    "score": 0.9,
    "n_samples": 100,
    "n_channels": 10,
}

d4 = {
    "time": 2,
    "dataset": FakeDataset(["d1", "d2"]),
    "subject": 1,
    "session": "session_0",
    "score": 0.9,
    "n_samples": 100,
    "n_channels": 10,
}


def to_pipeline_dict(pnames):
    return {n: "pipeline {}".format(n) for n in pnames}


def to_result_input(pnames, dsets):
    return dict(zip(pnames, dsets))


class Test_Stats(unittest.TestCase):
    def return_df(self, shape):
        size = shape[0] * shape[1]
        data = np.arange(size).reshape(*shape)
        return pd.DataFrame(data=data)

    def test_wilcoxon(self):
        P = ma.compute_pvals_wilcoxon(self.return_df((60, 5)))
        self.assertTrue(np.allclose(np.tril(P), 0), P)

    def test_perm_exhaustive(self):
        P = ma.compute_pvals_perm(self.return_df((4, 5)))
        Pl = P[np.tril_indices(P.shape[0])]
        self.assertTrue(np.allclose(Pl, (1 / 2 ** 4)), np.tril(P))

    def test_perm_random(self):
        P = ma.compute_pvals_perm(self.return_df((18, 5)))
        Pl = P[np.tril_indices(P.shape[0])]
        self.assertTrue(np.allclose(Pl, 1e-4), np.tril(P))


class Test_Integration(unittest.TestCase):
    def setUp(self):
        self.obj = Results(
            evaluation_class=DummyEvaluation, paradigm_class=DummyParadigm, suffix="test"
        )

    def tearDown(self):
        path = self.obj.filepath
        if os.path.isfile(path):
            os.remove(path)


class Test_Results(unittest.TestCase):
    def setUp(self):
        self.obj = Results(
            evaluation_class=DummyEvaluation, paradigm_class=DummyParadigm, suffix="test"
        )

    def tearDown(self):
        path = self.obj.filepath
        if os.path.isfile(path):
            os.remove(path)

    def testCanAddSample(self):
        self.obj.add(to_result_input(["a"], [d1]), to_pipeline_dict(["a"]))

    def testRecognizesAlreadyComputed(self):
        _in = to_result_input(["a"], [d1])
        self.obj.add(_in, to_pipeline_dict(["a"]))
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["a"]), d1["dataset"], d1["subject"]
        )
        self.assertTrue(len(not_yet_computed) == 0)

    def testCanAddMultiplePipelines(self):
        _in = to_result_input(["a", "b", "c"], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]))

    def testCanAddMultipleValuesPerPipeline(self):
        _in = to_result_input(["a", "b"], [[d1, d2], [d2, d1]])
        self.obj.add(_in, to_pipeline_dict(["a", "b"]))
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["a"]), d1["dataset"], d1["subject"]
        )
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["b"]), d2["dataset"], d2["subject"]
        )
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["b"]), d1["dataset"], d1["subject"]
        )
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)

    def testCanExportToDataframe(self):
        _in = to_result_input(["a", "b", "c"], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]))
        _in = to_result_input(["a", "b", "c"], [d2, d2, d3])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]))
        df = self.obj.to_dataframe()
        self.assertTrue(
            set(np.unique(df["pipeline"])) == set(("a", "b", "c")),
            np.unique(df["pipeline"]),
        )
        self.assertTrue(df.shape[0] == 6, df.shape[0])


if __name__ == "__main__":
    unittest.main()
