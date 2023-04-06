import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.pyplot import Figure

import moabb.analysis.meta_analysis as ma
from moabb import benchmark
from moabb.analysis import Results
from moabb.datasets.fake import FakeDataset
from moabb.evaluations.base import BaseEvaluation
from moabb.paradigms.base import BaseParadigm


try:
    from codecarbon import EmissionsTracker  # noqa

    from moabb.analysis.plotting import codecarbon_plot  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


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

if _carbonfootprint:
    d1["carbon_emission"] = 5
    d2["carbon_emission"] = 10
    d3["carbon_emission"] = 0.2
    d4["carbon_emission"] = 1


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
        self.assertTrue(np.allclose(Pl, (1 / 2**4)), np.tril(P))

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


if _carbonfootprint:

    class TestCodecarbonPlot(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.pp_dir = Path.cwd() / Path("moabb/tests/test_pipelines/")

        @classmethod
        def tearDownClass(cls):
            rep_dir = Path.cwd() / Path("benchmark/")
            shutil.rmtree(rep_dir)

        def setUp(self):
            self.data = benchmark(
                pipelines=str(self.pp_dir),
                evaluations=["WithinSession"],
                include_datasets=["FakeDataset"],
                results="moabb/results",
            )
            self.country = "France"
            self.pipelines = ["pipeline 1", "pipeline 2"]
            self.order_list = ["pipeline 2", "pipeline 1"]

        def test_codecarbon_plot_returns_figure(self):
            fig = codecarbon_plot(self.data)
            self.assertIsInstance(fig, Figure)

        def test_codecarbon_plot_title_includes_country(self):
            fig = codecarbon_plot(self.data, country=self.country)
            self.assertIn(self.country, fig._suptitle.get_text())

        def test_codecarbon_plot_filters_pipelines_correctly(self):
            fig = codecarbon_plot(self.data, pipelines=self.pipelines)
            pipelines_in_plot = set(fig.data["pipeline"].tolist())
            self.assertEqual(pipelines_in_plot, set(self.pipelines))

        def test_codecarbon_plot_orders_pipelines_correctly(self):
            fig = codecarbon_plot(self.data, order_list=self.order_list)
            hue_order_in_plot = fig._legend.get_lines()[0].get_data()[1].tolist()
            self.assertEqual(hue_order_in_plot, self.order_list)

        def test_codecarbon_plot_uses_log_scale_y_axis(self):
            fig = codecarbon_plot(self.data)
            self.assertEqual(fig.axes[0].get_yscale(), "log")


if __name__ == "__main__":
    unittest.main()
