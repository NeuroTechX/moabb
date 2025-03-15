import os
import shutil
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
    "session": "0",
    "score": 0.9,
    "n_samples": 100,
    "n_channels": 10,
}

d2 = {
    "time": 2,
    "dataset": FakeDataset(["d1", "d2"]),
    "subject": 2,
    "session": "0",
    "score": 0.9,
    "n_samples": 100,
    "n_channels": 10,
}

d3 = {
    "time": 2,
    "dataset": FakeDataset(["d1", "d2"]),
    "subject": 2,
    "session": "0",
    "score": 0.9,
    "n_samples": 100,
    "n_channels": 10,
}

d4 = {
    "time": 2,
    "dataset": FakeDataset(["d1", "d2"]),
    "subject": 1,
    "session": "0",
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


class TestStats:
    def return_df(self, shape):
        size = shape[0] * shape[1]
        data = np.arange(size).reshape(*shape)
        return pd.DataFrame(data=data)

    def test_wilcoxon(self):
        P = ma.compute_pvals_wilcoxon(self.return_df((60, 5)))
        assert np.allclose(np.tril(P), 0), P

    def test_perm_exhaustive(self):
        n_samples = 6
        data = (
            self.return_df((n_samples, 5)) * 0
        )  # We provide the exact same data for each pipeline
        n_perms = 2**n_samples
        pvals = ma.compute_pvals_perm(data)
        assert np.all(
            pvals == 1 - 1 / n_perms
        ), f"P-values should be equal to 1 - 1/n_perms {pvals}"

    def test_perm_random(self):
        rng = np.random.RandomState(12)
        data = (
            self.return_df((18, 5)) * 0
        )  # We provide the exact same data for each pipeline
        n_perms = 10000  # hardcoded in _pairedttest_random

        pvals = ma.compute_pvals_perm(data, seed=rng)
        assert np.all(
            pvals == 1 - 1 / n_perms
        ), f"P-values should be equal to 1 - 1/n_perms {pvals}"

    def test_edge_case_one_sample(self):
        data = self.return_df((1, 2))
        n_perms = 2
        pvals = ma.compute_pvals_perm(data)
        assert pvals.shape == (
            2,
            2,
        ), f"Incorrect dimension of p-values array {pvals.shape}"
        assert np.all(
            pvals == 1 - 1 / n_perms
        ), f"P-values should be equal to 1 - 1/n_perms {pvals}"

    def test_compute_pvals_exhaustif_cannot_be_zero(self):
        df = pd.DataFrame({"pipeline_1": [1, 1], "pipeline_2": [0, 0]})
        n_perms = 4
        pvals = ma.compute_pvals_perm(df)
        p1vsp2 = pvals[0, 1]
        assert p1vsp2 == 1 / n_perms, f"P-values cannot be zero {pvals}"

    def test_compute_pvals_random_cannot_be_zero(self):
        rng = np.random.RandomState(12)
        df = pd.DataFrame({"pipeline_1": [1] * 18, "pipeline_2": [0] * 18})
        n_perms = 10000  # hardcoded in _pairedttest_random
        pvals = ma.compute_pvals_perm(df, seed=rng)
        p1vsp2 = pvals[0, 1]
        assert p1vsp2 >= 1 / n_perms, "P-values cannot be zero "


class TestResults:
    def setup_method(self, method):
        self.obj = Results(
            evaluation_class=DummyEvaluation, paradigm_class=DummyParadigm, suffix="test"
        )

    def teardown_method(self, method):
        path = self.obj.filepath
        if os.path.isfile(path):
            os.remove(path)

    def test_add_sample(self):
        self.obj.add(
            to_result_input(["a"], [d1]), to_pipeline_dict(["a"]), "process_pipeline"
        )

    def test_recognizes_already_computed(self):
        _in = to_result_input(["a"], [d1])
        self.obj.add(_in, to_pipeline_dict(["a"]), "process_pipeline")
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["a"]),
            d1["dataset"],
            d1["subject"],
            "process_pipeline",
        )
        assert len(not_yet_computed) == 0

    def test_can_add_multiple_pipelines(self):
        _in = to_result_input(["a", "b", "c"], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]), "process_pipeline")

    def test_can_add_multiple_values_per_pipeline(self):
        _in = to_result_input(["a", "b"], [[d1, d2], [d2, d1]])
        self.obj.add(_in, to_pipeline_dict(["a", "b"]), "process_pipeline")
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["a"]),
            d1["dataset"],
            d1["subject"],
            "process_pipeline",
        )
        assert len(not_yet_computed) == 0, not_yet_computed
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["b"]),
            d2["dataset"],
            d2["subject"],
            "process_pipeline",
        )
        assert len(not_yet_computed) == 0, not_yet_computed
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["b"]),
            d1["dataset"],
            d1["subject"],
            "process_pipeline",
        )
        assert len(not_yet_computed) == 0, not_yet_computed

    def test_can_export_to_dataframe(self):
        _in = to_result_input(["a", "b", "c"], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]), "process_pipeline")
        _in = to_result_input(["a", "b", "c"], [d2, d2, d3])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]), "process_pipeline")
        df = self.obj.to_dataframe()
        assert set(np.unique(df["pipeline"])) == set(("a", "b", "c")), (
            np.unique(df["pipeline"]),
        )
        assert df.shape[0] == 6, df.shape[0]


if _carbonfootprint:

    class TestCodeCarbonPlot:
        @classmethod
        def setup_class(cls):
            cls.pp_dir = Path.cwd() / Path("moabb/tests/test_pipelines/")

        @classmethod
        def teardown_class(cls):
            rep_dir = Path.cwd() / Path("benchmark/")
            shutil.rmtree(rep_dir)

        def setup_method(self):
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
            assert isinstance(fig, Figure)

        def test_codecarbon_plot_title_includes_country(self):
            fig = codecarbon_plot(self.data, country=self.country)
            assert self.country in fig._suptitle.get_text()

        def test_codecarbon_plot_filters_pipelines_correctly(self):
            fig = codecarbon_plot(self.data, pipelines=self.pipelines)
            pipelines_in_plot = set(fig.data["pipeline"].tolist())
            assert pipelines_in_plot == set(self.pipelines)

        def test_codecarbon_plot_orders_pipelines_correctly(self):
            fig = codecarbon_plot(self.data, order_list=self.order_list)
            hue_order_in_plot = fig._legend.get_lines()[0].get_data()[1].tolist()
            assert hue_order_in_plot == self.order_list

        def test_codecarbon_plot_uses_log_scale_y_axis(self):
            fig = codecarbon_plot(self.data)
            assert fig.axes[0].get_yscale() == "log"
