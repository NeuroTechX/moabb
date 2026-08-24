import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
        assert np.all(pvals == 1 - 1 / n_perms), (
            f"P-values should be equal to 1 - 1/n_perms {pvals}"
        )

    def test_perm_random(self):
        rng = np.random.RandomState(12)
        data = (
            self.return_df((18, 5)) * 0
        )  # We provide the exact same data for each pipeline
        n_perms = 10000  # hardcoded in _pairedttest_random

        pvals = ma.compute_pvals_perm(data, seed=rng)
        assert np.all(pvals == 1 - 1 / n_perms), (
            f"P-values should be equal to 1 - 1/n_perms {pvals}"
        )

    def test_edge_case_one_sample(self):
        data = self.return_df((1, 2))
        n_perms = 2
        pvals = ma.compute_pvals_perm(data)
        assert pvals.shape == (2, 2), (
            f"Incorrect dimension of p-values array {pvals.shape}"
        )
        assert np.all(pvals == 1 - 1 / n_perms), (
            f"P-values should be equal to 1 - 1/n_perms {pvals}"
        )

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
        assert p1vsp2 >= 1 / n_perms, f"P-values cannot be zero {pvals}"

    @pytest.mark.parametrize(
        "function", [ma.compute_pvals_wilcoxon, ma.compute_pvals_perm, ma.compute_effect]
    )
    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_score_statistics_reject_nonfinite_values(self, function, invalid):
        df = pd.DataFrame({"pipeline_1": [0.7, invalid], "pipeline_2": [0.6, 0.5]})
        with pytest.raises(ValueError, match="finite"):
            function(df)

    @pytest.mark.parametrize(
        "function", [ma.compute_pvals_wilcoxon, ma.compute_pvals_perm, ma.compute_effect]
    )
    def test_score_statistics_reject_empty_values(self, function):
        df = pd.DataFrame(columns=["pipeline_1", "pipeline_2"], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            function(df)

    @pytest.mark.parametrize("n_subjects", [22, 25, 40])
    def test_wilcoxon_identical_pipelines(self, n_subjects):
        # Wilcoxon is undefined when every paired difference is zero. SciPy
        # returns 1.0 from its exact method but NaN from the normal
        # approximation it uses on larger samples, so the answer used to depend
        # on the number of subjects. See issue #678.
        df = pd.DataFrame(
            {"pipeline_1": [0.7] * n_subjects, "pipeline_2": [0.7] * n_subjects}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pvals = ma.compute_pvals_wilcoxon(df)
        assert np.isfinite(pvals).all(), f"P-values must be finite {pvals}"
        assert pvals[0, 1] == 0.5, f"Indistinguishable pipelines give 0.5 {pvals}"
        assert pvals[1, 0] == 0.5, f"Indistinguishable pipelines give 0.5 {pvals}"

    def test_wilcoxon_stays_inside_unit_interval(self):
        # Stouffer's method maps 0 and 1 to an infinite z-score, so the Wilcoxon
        # branch must keep p strictly inside (0, 1), as the permutation branch
        # already does.
        rng = np.random.RandomState(0)
        n = 60
        base = rng.uniform(0.4, 0.9, size=n)
        df = pd.DataFrame({"pipeline_1": base, "pipeline_2": base + 0.3})
        pvals = ma.compute_pvals_wilcoxon(df)
        offdiag = pvals[~np.eye(2, dtype=bool)]
        assert np.all(offdiag > 0), f"P-values cannot be zero {pvals}"
        assert np.all(offdiag < 1), f"P-values cannot be one {pvals}"

    def test_compute_effect_zero_spread(self):
        # Identical pipelines give 0/0 and a constant offset gives c/0. Neither
        # should come back as NaN. See issue #678.
        # 0.75 - 0.5 is exact in binary, so the paired differences really do
        # have a standard deviation of zero rather than a rounding residue.
        df = pd.DataFrame(
            {
                "pipeline_1": [0.5] * 10,
                "pipeline_2": [0.5] * 10,
                "pipeline_3": [0.75] * 10,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            effect = ma.compute_effect(df)
        assert not np.isnan(effect).any(), f"Effect sizes must not be NaN {effect}"
        assert effect[0, 1] == 0.0, f"Identical pipelines have no effect {effect}"
        assert effect[1, 0] == 0.0, f"Identical pipelines have no effect {effect}"
        assert effect[2, 0] == np.inf, f"A constant gain is unbounded {effect}"
        assert effect[0, 2] == -np.inf, f"A constant loss is unbounded {effect}"

    def test_compute_effect_single_observation_constant_difference(self):
        df = pd.DataFrame(
            {
                "pipeline_1": [0.5],
                "pipeline_2": [0.5],
                "pipeline_3": [0.75],
                "pipeline_4": [0.25],
            }
        )
        effect = ma.compute_effect(df)
        assert effect[0, 1] == 0.0, f"Identical pipelines have no effect {effect}"
        assert effect[1, 0] == 0.0, f"Identical pipelines have no effect {effect}"
        assert effect[2, 0] == np.inf, f"A constant gain is unbounded {effect}"
        assert effect[0, 2] == -np.inf, f"A constant loss is unbounded {effect}"
        assert effect[3, 0] == -np.inf, f"A constant loss is unbounded {effect}"
        assert effect[0, 3] == np.inf, f"A constant gain is unbounded {effect}"

    def test_dataset_statistics_no_nan_for_identical_pipelines(self):
        # End-to-end check of the path reported in issue #678: two pipelines
        # that score identically used to make compute_dataset_statistics emit
        # NaN, and find_significant_differences then printed "NaN" to stdout and
        # left the NaN in place.
        n_subjects = 25  # above perm_cutoff, so the Wilcoxon branch is used
        results = pd.DataFrame(
            [
                {"pipeline": pipeline, "dataset": "D1", "subject": subject, "score": 0.7}
                for subject in range(n_subjects)
                for pipeline in ("pipeline_1", "pipeline_2")
            ]
        )

        stats_df = ma.compute_dataset_statistics(results)
        assert not stats_df["p"].isna().any(), f"NaN p-value {stats_df}"
        assert not stats_df["smd"].isna().any(), f"NaN effect size {stats_df}"

        dfP, dfT = ma.find_significant_differences(stats_df)
        offdiag = ~np.eye(len(dfP), dtype=bool)
        assert np.isfinite(dfP.to_numpy(dtype=float)[offdiag]).all(), dfP
        assert np.isfinite(dfT.to_numpy(dtype=float)[offdiag]).all(), dfT


class TestCorrectedTtest:
    """Tests for the Nadeau & Bengio corrected resampled t-test."""

    n_train, n_test = 80, 20  # a 5-fold split of 100 examples

    def toy_scores(self, seed=42, n_resamples=10, delta=0.05):
        rng = np.random.RandomState(seed)
        base = 0.7 + 0.02 * rng.randn(n_resamples)
        return pd.DataFrame(
            {
                "pipeline_1": base + delta,
                "pipeline_2": base - 0.01 * rng.randn(n_resamples),
            }
        )

    def test_matches_manual_formula_and_scales_ttest_rel(self):
        from scipy import stats

        df = self.toy_scores()
        n = df.shape[0]
        x = (df["pipeline_1"] - df["pipeline_2"]).to_numpy()

        # manual Nadeau & Bengio statistic
        t_corr = x.mean() / np.sqrt((1 / n + self.n_test / self.n_train) * x.var(ddof=1))
        expected = stats.t.sf(t_corr, df=n - 1)

        pvals = ma.compute_pvals_corrected_ttest(df, self.n_train, self.n_test)
        assert pvals.shape == (2, 2)
        assert np.isclose(pvals[0, 1], expected), pvals

        # the corrected statistic is the standard paired t statistic
        # shrunk by sqrt((1/n) / (1/n + n2/n1))
        t_rel = stats.ttest_rel(df["pipeline_1"], df["pipeline_2"]).statistic
        shrink = np.sqrt((1 / n) / (1 / n + self.n_test / self.n_train))
        assert np.isclose(t_corr, t_rel * shrink)

    def test_more_conservative_than_uncorrected(self):
        from scipy import stats

        df = self.toy_scores()
        pvals = ma.compute_pvals_corrected_ttest(df, self.n_train, self.n_test)
        # one-sided p-value of the naive resampled (paired) t-test
        p_naive = stats.ttest_rel(
            df["pipeline_1"], df["pipeline_2"], alternative="greater"
        ).pvalue
        assert pvals[0, 1] > p_naive, (pvals[0, 1], p_naive)

        # the correction grows with the test/train ratio
        p_10fold = ma.compute_pvals_corrected_ttest(df, 90, 10)[0, 1]
        p_2fold = ma.compute_pvals_corrected_ttest(df, 50, 50)[0, 1]
        assert p_naive < p_10fold < pvals[0, 1] < p_2fold

    def test_matrix_properties_and_input_validation(self):
        import pytest

        rng = np.random.RandomState(7)
        df = pd.DataFrame(
            {
                "pipeline_1": 0.8 + 0.02 * rng.randn(10),
                "pipeline_2": 0.7 + 0.02 * rng.randn(10),
                "pipeline_3": 0.7 + 0.02 * rng.randn(10),
            }
        )
        pvals = ma.compute_pvals_corrected_ttest(df, self.n_train, self.n_test)
        assert pvals.shape == (3, 3)
        assert np.allclose(np.diag(pvals), 0)
        # one-tailed complementarity: p[i, j] + p[j, i] == 1
        off = ~np.eye(3, dtype=bool)
        assert np.allclose((pvals + pvals.T)[off], 1)
        # p-values stay strictly inside (0, 1) for Stouffer's method,
        # even for a deterministic (zero-variance) difference
        det = pd.DataFrame({"pipeline_1": [1.0] * 10, "pipeline_2": [0.0] * 10})
        p_det = ma.compute_pvals_corrected_ttest(det, self.n_train, self.n_test)
        assert 0 < p_det[0, 1] < p_det[1, 0] < 1
        # identical pipelines -> t = 0 -> p = 0.5
        same = pd.DataFrame({"pipeline_1": [0.7] * 10, "pipeline_2": [0.7] * 10})
        p_same = ma.compute_pvals_corrected_ttest(same, self.n_train, self.n_test)
        assert np.allclose(p_same[0, 1], 0.5)

        with pytest.raises(ValueError, match="order"):
            ma.compute_pvals_corrected_ttest(
                df, self.n_train, self.n_test, order=["pipeline_1", "wrong"]
            )
        with pytest.raises(ValueError, match="resamples"):
            ma.compute_pvals_corrected_ttest(df.iloc[:1], self.n_train, self.n_test)
        with pytest.raises(ValueError, match="positive"):
            ma.compute_pvals_corrected_ttest(df, 0, self.n_test)

    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_scores(self, invalid):
        df = pd.DataFrame(
            {"pipeline_1": [0.72, 0.74, 0.76], "pipeline_2": [0.68, 0.69, 0.70]}
        )
        df.iloc[0, 0] = invalid
        with pytest.raises(ValueError, match="finite"):
            ma.compute_pvals_corrected_ttest(df, n_train=80, n_test=20)

    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_train_test_sizes(self, invalid):
        df = self.toy_scores()
        with pytest.raises(ValueError, match="finite"):
            ma.compute_pvals_corrected_ttest(df, n_train=invalid, n_test=20)
        with pytest.raises(ValueError, match="finite"):
            ma.compute_pvals_corrected_ttest(df, n_train=80, n_test=invalid)

    def test_rejects_empty_scores(self):
        df = pd.DataFrame(columns=["pipeline_1", "pipeline_2"], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            ma.compute_pvals_corrected_ttest(df, n_train=80, n_test=20)


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
            to_pipeline_dict(["a"]), d1["dataset"], d1["subject"], "process_pipeline"
        )
        assert len(not_yet_computed) == 0

    def test_can_add_multiple_pipelines(self):
        _in = to_result_input(["a", "b", "c"], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]), "process_pipeline")

    def test_can_add_multiple_values_per_pipeline(self):
        _in = to_result_input(["a", "b"], [[d1, d2], [d2, d1]])
        self.obj.add(_in, to_pipeline_dict(["a", "b"]), "process_pipeline")
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["a"]), d1["dataset"], d1["subject"], "process_pipeline"
        )
        assert len(not_yet_computed) == 0, not_yet_computed
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["b"]), d2["dataset"], d2["subject"], "process_pipeline"
        )
        assert len(not_yet_computed) == 0, not_yet_computed
        not_yet_computed = self.obj.not_yet_computed(
            to_pipeline_dict(["b"]), d1["dataset"], d1["subject"], "process_pipeline"
        )
        assert len(not_yet_computed) == 0, not_yet_computed

    def test_can_export_to_dataframe(self):
        _in = to_result_input(["a", "b", "c"], [d1, d1, d2])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]), "process_pipeline")
        _in = to_result_input(["a", "b", "c"], [d2, d2, d3])
        self.obj.add(_in, to_pipeline_dict(["a", "b", "c"]), "process_pipeline")
        df = self.obj.to_dataframe()
        assert set(np.unique(df["pipeline"])) == {"a", "b", "c"}, (
            np.unique(df["pipeline"]),
        )
        assert df.shape[0] == 6, df.shape[0]

    def test_add_results_without_carbon_emission(self):
        """Test adding results that don't have carbon_emission key."""
        # Create result dict without carbon_emission
        d_no_carbon = {
            "time": 1,
            "dataset": FakeDataset(["d1", "d2"]),
            "subject": 1,
            "session": "0",
            "score": 0.9,
            "n_samples": 100,
            "n_channels": 10,
        }
        _in = to_result_input(["a"], [d_no_carbon])
        # Should not raise KeyError
        self.obj.add(_in, to_pipeline_dict(["a"]), "process_pipeline")
        df = self.obj.to_dataframe()
        assert df.shape[0] == 1

    def test_mixed_carbon_emission_results(self):
        """Test adding results where some have carbon_emission and some don't."""
        d_with_carbon = {
            "time": 1,
            "dataset": FakeDataset(["d1", "d2"]),
            "subject": 1,
            "session": "0",
            "score": 0.9,
            "n_samples": 100,
            "n_channels": 10,
        }
        d_without_carbon = {
            "time": 2,
            "dataset": FakeDataset(["d1", "d2"]),
            "subject": 2,
            "session": "0",
            "score": 0.85,
            "n_samples": 100,
            "n_channels": 10,
        }

        if _carbonfootprint:
            d_with_carbon["carbon_emission"] = 5
            d_with_carbon["codecarbon_task_name"] = "task1"

        # Add results with carbon_emission
        _in = to_result_input(["a"], [d_with_carbon])
        self.obj.add(_in, to_pipeline_dict(["a"]), "process_pipeline")

        # Add results without carbon_emission
        _in = to_result_input(["a"], [d_without_carbon])
        self.obj.add(_in, to_pipeline_dict(["a"]), "process_pipeline")

        # Should be able to export to dataframe without errors
        df = self.obj.to_dataframe()
        assert df.shape[0] == 2

    def test_add_batch_list_values(self):
        """Test adding a batch of results as a list for a single pipeline."""
        _in = {"a": [d1, d2, d4]}
        self.obj.add(_in, to_pipeline_dict(["a"]), "process_pipeline")
        df = self.obj.to_dataframe()
        assert df.shape[0] == 3, f"Expected 3 rows, got {df.shape[0]}"
        assert set(df["pipeline"]) == {"a"}

    def test_dataframe_with_missing_codecarbon_dataset(self):
        """Test that to_dataframe works even if codecarbon_task_name dataset doesn't exist."""
        # Add a result without carbon_emission
        d_no_carbon = {
            "time": 1,
            "dataset": FakeDataset(["d1", "d2"]),
            "subject": 1,
            "session": "0",
            "score": 0.9,
            "n_samples": 100,
            "n_channels": 10,
        }
        _in = to_result_input(["a"], [d_no_carbon])
        self.obj.add(_in, to_pipeline_dict(["a"]), "process_pipeline")

        # Should be able to call to_dataframe without KeyError
        df = self.obj.to_dataframe()
        assert df.shape[0] == 1
        # codecarbon_task_name should not be in columns if not present in HDF5
        if _carbonfootprint and "codecarbon_task_name" not in str(df.columns):
            pass  # Expected for old files


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
